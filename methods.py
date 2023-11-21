from pykeen import models


def model_init(model, dataset, seed=None):
    if model == 'TransE':
        model = models.TransE(triples_factory=dataset.training, random_seed=seed)
    elif model == 'RotatE':
        model = models.RotatE(triples_factory=dataset.training, random_seed=seed)
    return model
