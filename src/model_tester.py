from dataclasses import dataclass
from datetime import datetime
from typing import Tuple
from pathlib import Path


import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data


@dataclass
class ModelTester:
    dataset_name: str
    model_name: str
    model: MessagePassing
    epochs_num: int
    learning_rate: float
    data: Data

    losses: npt._ArrayLikeFloat_co = np.array([])
    train_accuracies: npt._ArrayLikeFloat_co = np.array([])
    test_accuracies: npt._ArrayLikeFloat_co = np.array([])

    def __generate_path(self) -> Path:
        tokens = [
            f"dataset-{self.dataset_name}",
            f"model-{self.model_name}",
            f"param1-{self.epochs_num}",
            f"param2-{self.learning_rate}",
            datetime.now().strftime("%Y%m%d_%H%M"),
        ]
        filename = "_".join(tokens) + ".npy"

        path = Path().cwd().parent / "output" / filename
        path.touch()

        return path

    def to_numpy_file(self):
        result = np.array([self.losses, self.train_accuracies, self.test_accuracies])

        path = self.__generate_path()
        np.save(self.__generate_path(), result)
        print("saved to", path)

    def __epoch_train(self, optimizer, criterion) -> Tuple[float, float]:
        self.model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = self.model(
            self.data.x, self.data.edge_index
        )  # Perform a single forward pass.
        loss = criterion(
            out[self.data.train_mask], self.data.y[self.data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        train_correct = (
            pred[self.data.train_mask] == self.data.y[self.data.train_mask]
        )  # Check against ground-truth labels.
        train_acc = int(train_correct.sum()) / int(
            self.data.train_mask.sum()
        )  # Derive ratio of correct predictions.

        return float(loss), train_acc

    def __test(self) -> float:
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = (
            pred[self.data.test_mask] == self.data.y[self.data.test_mask]
        )  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(
            self.data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return test_acc

    def train_and_test(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4,
        )
        criterion = torch.nn.CrossEntropyLoss()

        losses = []
        train_accs = []
        test_accs = []

        for _ in range(self.epochs_num):
            loss, train_acc = self.__epoch_train(optimizer, criterion)
            test_acc = self.__test()

            losses.append(loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        self.losses = np.array(losses)
        self.train_accuracies = np.array(train_accs)
        self.test_accuracies = np.array(test_accs)
