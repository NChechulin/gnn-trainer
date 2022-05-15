from typing import Dict

from torch_geometric.nn import (
    APPNP,
    AGNNConv,
    ARMAConv,
    CGConv,
    ChebConv,
    ClusterGCNConv,
    DNAConv,
    ECConv,
    EdgeConv,
    EGConv,
    FAConv,
    FeaStConv,
    GATConv,
    GatedGraphConv,
    GATv2Conv,
    GCN2Conv,
    GCNConv,
    GENConv,
    GeneralConv,
    GINConv,
    GINEConv,
    GMMConv,
    GraphConv,
    LEConv,
    LGConv,
    MessagePassing,
    MFConv,
    NNConv,
    PANConv,
    PDNConv,
    PNAConv,
    PointConv,
    PointNetConv,
    PointTransformerConv,
    PPFConv,
    ResGatedGraphConv,
    SAGEConv,
    SGConv,
    SignedConv,
    SplineConv,
    SuperGATConv,
    TAGConv,
    TransformerConv,
    WLConv,
)

MODELS: Dict[str, MessagePassing] = {
    "gcnconv": GCNConv,
    "chebconv": ChebConv,
    "sageconv": SAGEConv,
    "graphconv": GraphConv,
    "gatedgraphconv": GatedGraphConv,
    "resgatedgraphconv": ResGatedGraphConv,
    "gatconv": GATConv,
    "gatv2conv": GATv2Conv,
    "transformerconv": TransformerConv,
    "agnnconv": AGNNConv,
    "tagconv": TAGConv,
    "ginconv": GINConv,
    "gineconv": GINEConv,
    "armaconv": ARMAConv,
    "sgconv": SGConv,
    "appnp": APPNP,
    "mfconv": MFConv,
    "signedconv": SignedConv,
    "dnaconv": DNAConv,
    "pointnetconv": PointNetConv,
    "pointconv": PointConv,
    "gmmconv": GMMConv,
    "splineconv": SplineConv,
    "nnconv": NNConv,
    "ecconv": ECConv,
    "cgconv": CGConv,
    "edgeconv": EdgeConv,
    "ppfconv": PPFConv,
    "feastconv": FeaStConv,
    "pointtransformerconv": PointTransformerConv,
    "leconv": LEConv,
    "pnaconv": PNAConv,
    "clustergcnconv": ClusterGCNConv,
    "genconv": GENConv,
    "gcn2conv": GCN2Conv,
    "panconv": PANConv,
    "wlconv": WLConv,
    "supergatconv": SuperGATConv,
    "faconv": FAConv,
    "egconv": EGConv,
    "pdnconv": PDNConv,
    "generalconv": GeneralConv,
    "lgconv": LGConv,
}


def get_model_by_name(model_name: str) -> MessagePassing:
    """Returns a model by its name

    Args:
        model_name (str): name of a model

    Raises:
        KeyError: If model was not found

    Returns:
        MessagePassing: A model class to be initialized
    """
    try:
        return MODELS[model_name.lower()]
    except KeyError:
        raise KeyError(f"There is no model with name {model_name}")
