import numpy as np

from .trajectory import TrajectoryGRU


def new_model(config, spec, anchors=None):
    if spec['architecture'] != 'trajectory':
        raise ValueError('A previsão usa a GRU de trajetórias.')
    if anchors is None:
        shape = (len(config['classes']), spec['n_modes'], max(config['horizons']), 6)
        anchors = np.zeros(shape, dtype=np.float32)
    return TrajectoryGRU(len(config['classes']), anchors, spec['hidden'], config['embedding'])
