from typing import Any, Callable, Dict, Optional

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import (
    DBLP,
    DBP15K,
    FAUST,
    GDELT,
    ICEWS18,
    IMDB,
    MD17,
    NELL,
    OGB_MAG,
    OMDB,
    PPI,
    QM9,
    S3DIS,
    SHREC2016,
    TOSCA,
    UPFD,
    ZINC,
    Actor,
    Airports,
    Amazon,
    AmazonProducts,
    AMiner,
    AttributedGraphDataset,
    BAShapes,
    BitcoinOTC,
    CitationFull,
    Coauthor,
    CoMA,
    CoraFull,
    DeezerEurope,
    DynamicFAUST,
    Entities,
    FacebookPagePage,
    FakeDataset,
    FakeHeteroDataset,
    Flickr,
    GEDDataset,
    GemsecDeezer,
    GeometricShapes,
    GitHub,
    GNNBenchmarkDataset,
    HGBDataset,
    JODIEDataset,
    KarateClub,
    LastFM,
    LastFMAsia,
    MalNetTiny,
    MixHopSyntheticDataset,
    MNISTSuperpixels,
    ModelNet,
    MoleculeNet,
    MovieLens,
    PascalPF,
    PascalVOCKeypoints,
    PCPNetDataset,
    Planetoid,
    PolBlogs,
    QM7b,
    Reddit,
    Reddit2,
    RelLinkPredDataset,
    ShapeNet,
    SNAPDataset,
    SuiteSparseMatrixCollection,
    TUDataset,
    Twitch,
    WebKB,
    WikiCS,
    WikipediaNetwork,
    WILLOWObjectClass,
    WordNet18,
    WordNet18RR,
    Yelp,
)

DATASETS: Dict[str, Any] = {
    "karateclub": KarateClub,
    "tudataset": TUDataset,
    "gnnbenchmarkdataset": GNNBenchmarkDataset,
    "planetoid": Planetoid,
    "fakedataset": FakeDataset,
    "fakeheterodataset": FakeHeteroDataset,
    "nell": NELL,
    "citationfull": CitationFull,
    "corafull": CoraFull,
    "coauthor": Coauthor,
    "amazon": Amazon,
    "ppi": PPI,
    "reddit": Reddit,
    "reddit2": Reddit2,
    "flickr": Flickr,
    "yelp": Yelp,
    "amazonproducts": AmazonProducts,
    "qm7b": QM7b,
    "qm9": QM9,
    "md17": MD17,
    "zinc": ZINC,
    "moleculenet": MoleculeNet,
    "entities": Entities,
    "rellinkpreddataset": RelLinkPredDataset,
    "geddataset": GEDDataset,
    "attributedgraphdataset": AttributedGraphDataset,
    "mnistsuperpixels": MNISTSuperpixels,
    "faust": FAUST,
    "dynamicfaust": DynamicFAUST,
    "shapenet": ShapeNet,
    "modelnet": ModelNet,
    "coma": CoMA,
    "shrec2016": SHREC2016,
    "tosca": TOSCA,
    "pcpnetdataset": PCPNetDataset,
    "s3dis": S3DIS,
    "geometricshapes": GeometricShapes,
    "bitcoinotc": BitcoinOTC,
    "icews18": ICEWS18,
    "gdelt": GDELT,
    "dbp15k": DBP15K,
    "willowobjectclass": WILLOWObjectClass,
    "pascalvockeypoints": PascalVOCKeypoints,
    "pascalpf": PascalPF,
    "snapdataset": SNAPDataset,
    "suitesparsematrixcollection": SuiteSparseMatrixCollection,
    "aminer": AMiner,
    "wordnet18": WordNet18,
    "wordnet18rr": WordNet18RR,
    "wikics": WikiCS,
    "webkb": WebKB,
    "wikipedianetwork": WikipediaNetwork,
    "actor": Actor,
    "ogb_mag": OGB_MAG,
    "dblp": DBLP,
    "movielens": MovieLens,
    "imdb": IMDB,
    "lastfm": LastFM,
    "hgbdataset": HGBDataset,
    "jodiedataset": JODIEDataset,
    "mixhopsyntheticdataset": MixHopSyntheticDataset,
    "upfd": UPFD,
    "github": GitHub,
    "facebookpagepage": FacebookPagePage,
    "lastfmasia": LastFMAsia,
    "deezereurope": DeezerEurope,
    "gemsecdeezer": GemsecDeezer,
    "twitch": Twitch,
    "airports": Airports,
    "bashapes": BAShapes,
    "malnettiny": MalNetTiny,
    "omdb": OMDB,
    "polblogs": PolBlogs,
}


def get_dataset(
    name: str,
    subset: Optional[str] = None,
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
        if subset:
            # If a dataset does not support `root`
            try:
                return Dataset(
                    root=f"../data/{name}",
                    name=subset,
                    transform=transform() if transform else None,
                )
            except TypeError:
                return Dataset(
                    name=subset,
                    transform=transform() if transform else None,
                )
        else:
            # If a dataset does not support `root`
            try:
                return Dataset(
                    root=f"../data/{name}",
                    transform=transform() if transform else None,
                )
            except TypeError:
                return Dataset(
                    transform=transform() if transform else None,
                )
    except KeyError:
        raise KeyError(f"No dataset named {name}")
