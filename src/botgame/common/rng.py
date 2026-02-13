import random
from typing import List, Dict, Any

class PRNG:
    """Deterministic pseudo-random number generator."""
    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, sequence: List[Any]) -> Any:
        return self._rng.choice(sequence)
