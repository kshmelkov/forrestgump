#!/usr/bin/python

import numpy as np
import nibabel as nib
import cortex
import os

# a script to batch semi-automatic convertion of fsaverage altas Destrieux to pycortex script

def read_label(filename):
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            ss = line.split()
            if len(ss) <= 1:
                continue
            yield int(ss[0])


labels_dir = 'fsaverage'
labels_vertices = dict()
for f in os.listdir(labels_dir):
    labels_vertices[f] = list(read_label(os.path.join(labels_dir, f)))

m = cortex.get_mapper('fsaverage', 'grp_bold')
# print len(labels_vertices)
labels = set(map(lambda x: x.split('.')[1], labels_vertices.keys()))

labels = []
with open('./labels', 'r') as f:
    for line in f:
        labels.append(line.strip())
labels.sort()
print labels

for label in labels:
    lh = np.array(labels_vertices['lh.'+label+'.label'])
    rh = np.array(labels_vertices['rh.'+label+'.label'])
    print label

    vertices = (lh, rh)
    vols = m.backwards(vertices)
    prefix = ('lh', 'rh')
    # print (vols[0] > 0).sum(), lh.shape[0]

    for (hemi, vol) in zip(prefix, vols):
        name = hemi+'.'+label
        print '______', name, vol.sum()
        if vol.sum() > 10:
            volume = cortex.Volume(vol, 'fsaverage', 'grp_bold')
            cortex.add_roi(volume, name=name)
            raw_input()
