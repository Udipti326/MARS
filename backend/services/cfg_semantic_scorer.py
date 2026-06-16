# backend/services/cfg_semantic_scorer.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


class RegressionHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.LayerNorm(hidden // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(hidden // 2, hidden // 4),
            torch.nn.LayerNorm(hidden // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(hidden // 4, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_pair_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    e1, e2: shape [batch, dim]
    returns: shape [batch, dim*7 + 3]
    """
    if e1.ndim != 2 or e2.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {e1.shape=} {e2.shape=}")
    if e1.shape != e2.shape:
        raise ValueError(f"Embedding shapes must match, got {e1.shape=} {e2.shape=}")

    abs_diff = np.abs(e1 - e2)
    prod = e1 * e2
    cosine = np.sum(e1 * e2, axis=1, keepdims=True)
    l1 = np.sum(abs_diff, axis=1, keepdims=True)
    l2 = np.linalg.norm(abs_diff, axis=1, keepdims=True)
    mean_emb = (e1 + e2) / 2.0
    max_emb = np.maximum(e1, e2)
    min_emb = np.minimum(e1, e2)

    return np.concatenate(
        [
            e1,
            e2,
            abs_diff,
            prod,
            mean_emb,
            max_emb,
            min_emb,
            cosine,
            l1,
            l2,
        ],
        axis=1,
    )


class SemanticRelatednessScorer:
    """
    Loads a saved encoder + scaler + regression head from backend/semantic_artifacts.
    Falls back to the base encoder if the fine-tuned encoder folder is missing.
    """

    def __init__(self, artifact_dir: str | Path | None = None, base_model_name: str = "intfloat/e5-large-v2"):
        base = Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parents[1] / "semantic_artifacts"
        self.artifact_dir = base.resolve()
        self.model_dir = self.artifact_dir / "model"
        self.scaler_path = self.artifact_dir / "scaler.joblib"
        self.head_path = self.artifact_dir / "head.pt"
        self.base_model_name = base_model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = self._load_encoder()
        self.scaler = self._load_scaler()
        self.head = self._load_head()

        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.input_dim = embedding_dim * 7 + 3

        # Validate head input shape against the saved feature size
        first_linear = None
        for module in self.head.net:
            if isinstance(module, torch.nn.Linear):
                first_linear = module
                break
        if first_linear is not None and first_linear.in_features != self.input_dim:
            raise ValueError(
                f"Head input mismatch: head expects {first_linear.in_features}, "
                f"but build_pair_features produces {self.input_dim}"
            )

        self.head.eval()

    def _load_encoder(self) -> SentenceTransformer:
        if _safe_exists(self.model_dir):
            try:
                return SentenceTransformer(str(self.model_dir), device=self.device)
            except Exception as exc:
                print(f"[CFG scorer] Failed to load fine-tuned encoder from {self.model_dir}: {exc}")
        print(f"[CFG scorer] Falling back to base model: {self.base_model_name}")
        return SentenceTransformer(self.base_model_name, device=self.device)

    def _load_scaler(self):
        if not _safe_exists(self.scaler_path):
            raise FileNotFoundError(f"Missing scaler file: {self.scaler_path}")
        return joblib.load(self.scaler_path)

    def _load_head(self) -> RegressionHead:
        if not _safe_exists(self.head_path):
            raise FileNotFoundError(f"Missing head file: {self.head_path}")

        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        input_dim = embedding_dim * 7 + 3
        head = RegressionHead(input_dim=input_dim).to(self.device)

        state = torch.load(self.head_path, map_location=self.device)
        head.load_state_dict(state)
        return head

    def _encode(self, text: str) -> np.ndarray:
        # E5-style prefixing
        return self.encoder.encode(
            [f"sentence: {text or ''}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def predict(self, text1: str, text2: str, debug: bool = False) -> float | Dict[str, Any]:
        e1 = self._encode(text1)
        e2 = self._encode(text2)

        feats = build_pair_features(e1, e2)
        feats_scaled = self.scaler.transform(feats)
        x = torch.tensor(feats_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = self.head(x).detach().cpu().numpy()[0]

        score = float(np.clip(pred, 0.0, 1.0))

        if debug:
            return {
                "text1": text1,
                "text2": text2,
                "score": score,
                "encoder_model": getattr(self.encoder, "model_card_data", None).__dict__.get("model_id", None)
                if getattr(self.encoder, "model_card_data", None) is not None
                else None,
                "artifact_dir": str(self.artifact_dir),
                "model_dir_exists": _safe_exists(self.model_dir),
                "scaler_exists": _safe_exists(self.scaler_path),
                "head_exists": _safe_exists(self.head_path),
                "embedding_dim": int(self.encoder.get_sentence_embedding_dimension()),
                "feature_dim": int(feats.shape[1]),
            }

        return score

    def batch_predict(self, pairs: Iterable[tuple[str, str]]) -> List[float]:
        return [self.predict(a, b) for a, b in pairs]

    def health_check(self) -> Dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "model_dir": str(self.model_dir),
            "model_dir_exists": _safe_exists(self.model_dir),
            "scaler_exists": _safe_exists(self.scaler_path),
            "head_exists": _safe_exists(self.head_path),
            "device": self.device,
            "embedding_dim": int(self.encoder.get_sentence_embedding_dimension()),
            "input_dim": int(self.input_dim),
        }


if __name__ == "__main__":
    scorer = SemanticRelatednessScorer()
    print(scorer.health_check())
    print(scorer.predict("A cat sits on the mat.", "A cat is on a mat.", debug=True))