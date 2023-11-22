from pykeen import models


def model_init(model, dataset, seed=None):
    model_classes = {
        'ComplEx': models.ComplEx,
        'ConvE': models.ConvE,
        'ConvKB': models.ConvKB,
        'DistMult': models.DistMult,
        'ERMLP': models.ERMLP,
        'HolE': models.HolE,
        'KG2E': models.KG2E,
        'MuRE': models.MuRE,
        'NTN': models.NTN,
        'ProjE': models.ProjE,
        'QuatE': models.QuatE,
        'RESCAL': models.RESCAL,
        'RotatE': models.RotatE,
        'SE': models.SE,
        'SimplE': models.SimplE,
        'TransD': models.TransD,
        'TransE': models.TransE,
        'TransH': models.TransH,
        'TransR': models.TransR,
        'TuckER': models.TuckER,
        'UM': models.UM,
    }
    if model in model_classes:
        return model_classes[model](triples_factory=dataset.training, random_seed=seed)
    else:
        raise ValueError(f"Unsupported model: {model}")
