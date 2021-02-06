from epitome.models import *
from epitome.dataset import *
import numpy as np

import os

if __name__ == '__main__':
    apm = np.ones((10000, 130_000))
    rpf = os.getcwd() + '/data/test_regions.bed'

    targets = ['CTCF','RAD21','SMC3']
    celltypes = ['K562', 'A549', 'GM12878']

    dataset = EpitomeDataset(targets=targets, cells=celltypes)

    model = VLP(dataset, test_celltypes = ["K562"]) # cell line reserved for testing
    model.train(1000)
    res = model.score_matrix(apm, rpf)
    print(res.shape)