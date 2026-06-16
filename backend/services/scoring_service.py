import math


class ScoringService:

    # -------------------------
    # 1. Evidence Score
    # -------------------------
    @staticmethod
    def compute_evidence_score(support_chunks, all_chunks):
        if not all_chunks:
            return 0.0

        support_ratio = len(support_chunks) / len(all_chunks)

        # Assume each chunk has similarity score (0–1)
        similarities = [c.get("similarity", 0.5) for c in support_chunks]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        evidence_score = (
            0.6 * avg_similarity +
            0.4 * support_ratio
        )

        return round(min(max(evidence_score, 0), 1), 3)

    # -------------------------
    # 2. Credibility Score
    # -------------------------
    @staticmethod
    def compute_credibility_score(sources):
        if not sources:
            return 0.0

        scores = []

        for s in sources:
            domain = s.get("domain", "")
            citations = s.get("citations", 0)
            recency = s.get("recency", 0.5)  # normalized

            # domain score
            if ".edu" in domain or ".gov" in domain:
                domain_score = 0.9
            elif "arxiv" in domain:
                domain_score = 0.85
            else:
                domain_score = 0.6

            citation_score = min(citations / 100, 1)

            score = (
                0.4 * domain_score +
                0.3 * citation_score +
                0.3 * recency
            )

            scores.append(score)

        credibility_score = sum(scores) / len(scores)

        return round(min(max(credibility_score, 0), 1), 3)

    # -------------------------
    # 3. Contradiction Score
    # -------------------------
    @staticmethod
    def compute_contradiction_score(contradictions, all_chunks):
        if not all_chunks:
            return 0.0

        contradiction_ratio = len(contradictions) / len(all_chunks)

        # stronger penalty if many contradictions
        contradiction_score = min(contradiction_ratio * 1.2, 1)

        return round(contradiction_score, 3)

    # -------------------------
    # 4. Diversity Score
    # -------------------------
    @staticmethod
    def compute_diversity_score(sources):
        if not sources:
            return 0.0

        domains = set([s.get("domain") for s in sources])
        diversity_score = len(domains) / len(sources)

        return round(min(diversity_score, 1), 3)

    # -------------------------
    # 5. Final Confidence (RCS)
    # -------------------------
    @staticmethod
    def compute_confidence(evidence, credibility, contradiction, diversity):
        confidence = (
            0.4 * evidence +
            0.3 * credibility -
            0.2 * contradiction +
            0.1 * diversity
        )

        return round(min(max(confidence, 0), 1), 3)

    # -------------------------
    # 6. Label Mapping
    # -------------------------
    @staticmethod
    def get_label(score):
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    # -------------------------
    # MASTER FUNCTION
    # -------------------------
    @classmethod
    def score_claim(cls, data):
        evidence = cls.compute_evidence_score(
            data.get("support_chunks", []),
            data.get("all_chunks", [])
        )

        credibility = cls.compute_credibility_score(
            data.get("sources", [])
        )

        contradiction = cls.compute_contradiction_score(
            data.get("contradictions", []),
            data.get("all_chunks", [])
        )

        diversity = cls.compute_diversity_score(
            data.get("sources", [])
        )

        confidence = cls.compute_confidence(
            evidence, credibility, contradiction, diversity
        )

        return {
            "evidence_score": evidence,
            "credibility_score": credibility,
            "contradiction_score": contradiction,
            "diversity_score": diversity,
            "confidence": confidence,
            "label": cls.get_label(confidence)
        }