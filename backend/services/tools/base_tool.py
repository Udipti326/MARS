from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError