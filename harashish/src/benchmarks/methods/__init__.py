"""Method registry for SC3 benchmark."""

METHOD_REGISTRY = {}

# Import methods that exist, skip those that are missing
def _safe_import():
    global METHOD_REGISTRY
    try:
        from .analytical import GSEModel, ESOLModel
        METHOD_REGISTRY["gse"] = GSEModel
        METHOD_REGISTRY["esol"] = ESOLModel
    except ImportError:
        pass

    try:
        from .sklearn_models import (
            RandomForestModel, XGBoostModel, LightGBMModel,
            CatBoostModel, MLPModel, DecisionTreeModel, DissolvrModel,
        )
        METHOD_REGISTRY.update({
            "rf_rdkit": RandomForestModel,
            "xgb_rdkit": XGBoostModel,
            "lgb_rdkit": LightGBMModel,
            "catboost_rdkit": CatBoostModel,
            "mlp_rdkit": MLPModel,
            "dt_rdkit": DecisionTreeModel,
            "dissolvr": DissolvrModel,
        })
    except ImportError:
        pass

    try:
        from .fastprop import FastPropModel
        METHOD_REGISTRY["fastprop"] = FastPropModel
    except ImportError:
        pass

    try:
        from .fastsolv import FastSolvModel
        METHOD_REGISTRY["fastsolv"] = FastSolvModel
    except ImportError:
        pass

    try:
        from .tayyebi import TayyebiMordredModel
        METHOD_REGISTRY["tayyebi_mordred"] = TayyebiMordredModel
    except ImportError:
        pass

    try:
        from .gp_tanimoto import GPTanimotoModel
        METHOD_REGISTRY["gp_tanimoto"] = GPTanimotoModel
    except ImportError:
        pass

    try:
        from .abraham import AbrahamMLModel
        METHOD_REGISTRY["abraham_ml"] = AbrahamMLModel
    except ImportError:
        pass

    try:
        from .unifac_method import UNIFACMLModel
        METHOD_REGISTRY["unifac_ml"] = UNIFACMLModel
    except ImportError:
        pass

    try:
        from .chemfm import _ChemFMMethod
        METHOD_REGISTRY["chemfm"] = _ChemFMMethod
    except ImportError:
        pass

_safe_import()
