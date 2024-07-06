from .rdkit_featurizer import RDKitFeaturizer

FEATURIZER_REGISTRY = {
    "rdkit": RDKitFeaturizer,
}


def get_featurizer(name: str, **kwargs):
    if name not in FEATURIZER_REGISTRY:
        raise ValueError(f"Unknown featurizer: {name}. Available: {list(FEATURIZER_REGISTRY.keys())}")
    return FEATURIZER_REGISTRY[name](**kwargs)
