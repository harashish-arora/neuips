from .rdkit_featurizer import RDKitFeaturizer
from .morgan_featurizer import MorganFeaturizer
from .dissolvr_featurizer import DissolvrFeaturizer
from .mordred_featurizer import MordredFeaturizer

FEATURIZER_REGISTRY = {
    "rdkit": RDKitFeaturizer,
    "morgan": MorganFeaturizer,
    "dissolvr": DissolvrFeaturizer,
    "mordred": MordredFeaturizer,
}


def get_featurizer(name: str, **kwargs):
    if name not in FEATURIZER_REGISTRY:
        raise ValueError(f"Unknown featurizer: {name}. Available: {list(FEATURIZER_REGISTRY.keys())}")
    return FEATURIZER_REGISTRY[name](**kwargs)
