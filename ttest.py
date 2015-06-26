#!/usr/bin/python2

import sys
import os
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker
from scipy.stats import ttest_rel
from fg_constants import *


def load_cv_map(regressors, subj, masker):
    img = os.path.join(MAPS_DIR, regressors, subj, 'corr_cv.nii.gz')
    return masker.transform(img)


def join_all_subjects(regressor, subjects, masker):
    maps = []
    for s in subjects:
        subj = 'sub%03d' % s
        maps.append(load_cv_map(regressor, subj, masker))
    return np.vstack(maps)


def make_ttest(reg1, reg2):
    masker = NiftiMasker(nib.load(MASK_FILE), standardize=False)
    masker.fit()

    subjects = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    a = np.arctanh(join_all_subjects(reg1, subjects, masker))
    b = np.arctanh(join_all_subjects(reg2, subjects, masker))
    t, prob = ttest_rel(a, b)

    tt = masker.inverse_transform(t)
    pp = masker.inverse_transform(prob)
    return tt, pp

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        reg1 = sys.argv[1]
        reg2 = sys.argv[2]

    tt, pp = make_ttest(reg1, reg2)

    nib.save(tt, 'ttest/ttest_t_%s_vs_%s.nii.gz' % (A, B))
    nib.save(pp, 'ttest/ttest_prob_%s_vs_%s.nii.gz' % (A, B))
