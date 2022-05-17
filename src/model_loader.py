from math import sqrt
from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, MessagePassing


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int):
        super().__init__()
        # Just a safety measure to prevent output layer dimensions be less
        # than actual number of classes
        output_size: int = max(int(sqrt(num_node_features)), num_classes)

        self.conv1 = GCNConv(in_channels=num_node_features, out_channels=output_size)
        self.conv2 = GCNConv(in_channels=output_size, out_channels=num_classes)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int):
        super().__init__()
        # Just a safety measure to prevent output layer dimensions be less
        # than actual number of classes
        output_size: int = max(int(sqrt(num_node_features)), num_classes)

        self.conv1 = GATConv(in_channels=num_node_features, out_channels=output_size)
        self.conv2 = GATConv(in_channels=output_size, out_channels=num_classes)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


MODELS: Dict[str, MessagePassing] = {
    "gcn": GCN,
    "gat": GAT,
}


def get_model_by_name(model_name: str) -> MessagePassing:
    """Returns a model by its name

    Args:
        model_name (str): name of a model

    Raises:
        KeyError: If model was not found

    Returns:
        torch.nn.Module: A model class to be initialized
    """
    try:
        return MODELS[model_name.lower()]
    except KeyError:
        raise KeyError(f"There is no model with name {model_name}")
