import random
from typing import List, Dict, Any


def get_random_subset(
    dataset: List[Dict[str, Any]], subset_size: int, seed: int = 42
) -> List[Dict[str, Any]]:
    """Get a random subset of the dataset."""
    random.seed(seed)  # For reproducibility
    if subset_size >= len(dataset):
        return dataset
    return random.sample(dataset, subset_size)
