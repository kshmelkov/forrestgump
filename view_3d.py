#!/usr/bin/python

import numpy as np
import nibabel as nib
import cortex

import sys

def load_map(nifti, vmin=0.1, vmax=0.4):
    data = nifti.get_data()
    data[np.logical_and(data < 4, data > -4)] = 0
    # data = nib.load(filename).get_data()
    vol = cortex.Volume(data.T, 'fsaverage', 'grp_bold', vmin=vmin, vmax=vmax, cmap='coldwarm')
    # vol = cortex.Volume(data.T, 'fsaverage', 'grp_bold', vmin=vmin, vmax=vmax, cmap='hot')
    ds = cortex.dataset.Dataset(image=vol)
    web = cortex.webgl.show(ds, types=('inflated',), port=8080, open_browser=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        load_map(nib.load(sys.argv[1]), -10, 10)
    raw_input()
