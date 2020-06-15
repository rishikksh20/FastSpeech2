import numpy as np
import os
import hparams as hp
from utils.files import get_files

if __name__ == '__main__':
    min_e = []
    min_p = []
    max_e = []
    max_p = []

    energy_path = os.path.join(hp.data_dir, 'energy')
    pitch_path = os.path.join(hp.data_dir, 'pitch')
    energy_files = get_files(energy_path, extension='.npy')
    pitch_files = get_files(pitch_path, extension='.npy')

    assert len(energy_files) == len(pitch_files)

    for f in energy_files:
        e = np.load(f)
        min_e.append(e.min())
        max_e.append(e.max())

    for f in pitch_files:
        p = np.load(f)
        min_p.append(p.min())
        max_p.append(p.max())

    print("Min Energy : {}".format(min(min_e)))
    print("Max Energy : {}".format(max(max_e)))
    print("Min Pitch : {}".format(min(min_p)))
    print("Max Pitch : {}".format(max(max_p)))