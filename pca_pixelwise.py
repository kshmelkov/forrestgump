#!/usr/bin/python
import cross_validation as cv
import numpy as np
import os
import nibabel as nib
from fg_constants import *

from sklearn.decomposition import PCA

if __name__ == '__main__':
    subj_list = [1,2,3,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]
    segnum = 0
    sessions = []
    masker = cv.get_masker()
    for sub_num in subj_list:
        clean_data_dir = os.path.join(PREP_DATA_DIR, 'sub%03d' % sub_num)
        img = os.path.join(clean_data_dir, 'run00%i.nii.gz' % segnum)
        segment = masker.transform(img)
        sessions.append(segment)
    print map(np.shape, sessions)

    voxels = sessions[0].shape[1]
    time = sessions[0].shape[0]
    scores = np.zeros((1, voxels))
    for v in range(voxels):
        if v % 1000 == 0:
            print 'v=%i' % v
        src = np.zeros((time, len(subj_list)), dtype=np.float32)
        for (i, s) in enumerate(sessions):
            src[:, i] = s[:, v]
        pca = PCA(n_components=1)
        pca.fit(src)
        scores[0, v] = pca.explained_variance_ratio_[0]
        print 'score = ', scores[0, v]

    brain = masker.inverse_transform(scores)
    nib.save(brain, './pca_res.nii.gz')
