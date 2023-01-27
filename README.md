# GNN Training Script

A small utility to train a given GNN model on a given dataset.

## Usage

```
usage: main.py [-h] --dataset DATASET --model MODEL --epochs EPOCHS --learning_rate LEARNING_RATE

A small utility to train a given GNN model on a given dataset.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name (for example, KarateClub or Planetoid/Cora)
  --model MODEL         Model name (for example, GCNConv)
  --epochs EPOCHS       Number of epochs
  --learning_rate LEARNING_RATE
                        Learning rate
```

### Examples

```sh
python3 main.py --model="GCNConv" --epochs=300 --learning_rate=0.05 --dataset="Planetoid/Cora"
```

```sh
python3 main.py --model="GATConv" --dataset="NELL" --epochs=150 --learning_rate=0.1
```

## Input and Output paths

```
.
├── data
│  ├── ...
├── output
│  ├── ...
└── src
   ├── dataset_loader.py
   ├── main.py
   ├── model_loader.py
   ├── model_tester.
```

When ran from `src/` directory, the script saves data in the `data` folder, and outputs files to `output/` directory.
