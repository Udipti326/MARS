from __future__ import annotations

from typing import Any, Dict, List, Optional


def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _extract_claim_text(item: Dict[str, Any]) -> str:
    return str(
        item.get("claim")
        or item.get("text")
        or item.get("statement")
        or item.get("title")
        or ""
    ).strip()


class RefinementAgent:
    """
    Consumes already-scored claims and decides which ones should be re-retrieved
    or re-queried.

    Expected input for each item:
      {
        "claim": "...",
        "scores": {
          "evidence_score": 0.0-1.0,
          "credibility_score": 0.0-1.0,
          "contradiction_score": 0.0-1.0,
          "diversity_score": 0.0-1.0,
          "confidence": 0.0-1.0,
          "label": "HIGH|MEDIUM|LOW"
        },
        "judge": {...},
        "factcheck": {...}
      }
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        evidence_threshold: float = 0.60,
        credibility_threshold: float = 0.65,
        contradiction_threshold: float = 0.35,
    ):
        self.confidence_threshold = confidence_threshold
        self.evidence_threshold = evidence_threshold
        self.credibility_threshold = credibility_threshold
        self.contradiction_threshold = contradiction_threshold

    def _needs_refinement(self, claim_item: Dict[str, Any]) -> Dict[str, Any]:
        scores = claim_item.get("scores", {}) if isinstance(claim_item.get("scores", {}), dict) else {}

        evidence = _clamp(scores.get("evidence_score", 0.0))
        credibility = _clamp(scores.get("credibility_score", 0.0))
        contradiction = _clamp(scores.get("contradiction_score", 0.0))
        confidence = _clamp(scores.get("confidence", 0.0))
        label = str(scores.get("label", "")).upper()

        reasons: List[str] = []
        if confidence < self.confidence_threshold:
            reasons.append("low_confidence")
        if evidence < self.evidence_threshold:
            reasons.append("weak_evidence")
        if credibility < self.credibility_threshold:
            reasons.append("weak_credibility")
        if contradiction > self.contradiction_threshold:
            reasons.append("high_contradiction")
        if label == "LOW":
            reasons.append("label_low")

        return {
            "claim": _extract_claim_text(claim_item),
            "scores": {
                "evidence_score": evidence,
                "credibility_score": credibility,
                "contradiction_score": contradiction,
                "confidence": confidence,
                "label": label,
            },
            "needs_refinement": bool(reasons),
            "reasons": reasons,
            "priority": round(
                min(
                    1.0,
                    (1.0 - confidence) * 0.5
                    + (1.0 - evidence) * 0.2
                    + (1.0 - credibility) * 0.2
                    + contradiction * 0.1,
                ),
                3,
            ),
            "recommended_action": self._recommend_action(evidence, credibility, contradiction, confidence),
            "retrieval_hint": _extract_claim_text(claim_item),
        }

    @staticmethod
    def _recommend_action(evidence: float, credibility: float, contradiction: float, confidence: float) -> str:
        if contradiction > 0.35:
            return "seek_counterevidence"
        if evidence < 0.6:
            return "broaden_supporting_sources"
        if credibility < 0.65:
            return "upgrade_source_quality"
        if confidence < 0.75:
            return "recheck_claim"
        return "keep"

    def run(
        self,
        scored_claims: List[Dict[str, Any]],
        max_iterations: int = 2,
    ) -> Dict[str, Any]:
        scored_claims = scored_claims or []

        refinement_plan = [self._needs_refinement(item) for item in scored_claims]
        weak_claims = [item for item in refinement_plan if item["needs_refinement"]]
        strong_claims = [item for item in refinement_plan if not item["needs_refinement"]]

        return {
            "max_iterations": max_iterations,
            "weak_claims": weak_claims,
            "strong_claims": strong_claims,
            "needs_rerun": len(weak_claims) > 0,
            "total_claims": len(scored_claims),
            "weak_count": len(weak_claims),
            "strong_count": len(strong_claims),
            "refinement_plan": refinement_plan,
        }