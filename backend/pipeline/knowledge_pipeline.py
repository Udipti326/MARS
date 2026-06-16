from backend.agents.concept_extractor_agent import ConceptExtractorAgent
from backend.agents.dependency_agent import DependencyAgent


class KnowledgePipeline:

    def __init__(self):
        self.concept_extractor = ConceptExtractorAgent()
        self.dependency_agent = DependencyAgent()

    def run(self, summary, claims, topic=""):

        # -------------------------
        # 1. Extract Concepts
        # -------------------------
        concept_data = self.concept_extractor.run(
            summary,
            claims,
            topic
        )

        concepts = concept_data.get("concepts", [])

        # -------------------------
        # 2. Build Dependencies
        # -------------------------
        dependency_data = self.dependency_agent.run(
            concepts,
            topic
        )

        return {
            "concepts": concept_data,
            "dependencies": dependency_data
        }