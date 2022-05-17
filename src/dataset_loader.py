from pathlib import Path
from typing import Any, Callable, Dict, Optional

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset

DATASETS: Dict[str, Any] = {
    "proteins": TUDataset,
    "imdb-binary": TUDataset,
    "pattern": GNNBenchmarkDataset,
    "cluster": GNNBenchmarkDataset,
}


def get_dataset(
    name: str,
    transform: Optional[Callable] = None,
) -> InMemoryDataset:
    """Returns a dataset by it's name and subset

    Args:
        name (str): Name of a general dataset (for example, Planetoid)
        subset (Optional[str], optional): Name of subset (for example, Cora). Defaults to None.

    Raises:
        KeyError: If a dataset was not found

    Returns:
        Dataset
    """
    try:
        Dataset = DATASETS[name.lower()]

        root = Path.cwd().parent / "data" / name
        return Dataset(root, name, transform=transform() if transform else None)
    except KeyError:
        raise KeyError(f"No dataset named {name}")
