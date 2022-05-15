import argparse
from dataset_loader import get_dataset
from model_loader import get_model_by_name
from torch_geometric.transforms import NormalizeFeatures
from utils import split_in_two

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset name (for example, KarateClub or Planetoid/Cora)",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model name (for example, GCNConv)",
)
parser.add_argument(
    "--epochs",
    type=int,
    required=True,
    help="Number of epochs",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    required=True,
    help="Learning rate",
)


def main():
    args = parser.parse_args()

    dataset_name, subset_name = split_in_two(args.dataset)
    model_name: str = args.model
    epochs_num: int = args.epochs
    learning_rate: float = args.learning_rate

    dataset = get_dataset(
        name=dataset_name,
        subset=subset_name,
        transform=NormalizeFeatures,
    )
    Model = get_model_by_name(model_name)


if __name__ == "__main__":
    main()
