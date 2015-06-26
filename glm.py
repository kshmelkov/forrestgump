#!/usr/bin/python

import os
import numpy as np
import nibabel as nib
from joblib import Memory
from nilearn.input_data import NiftiMasker
import nipy.modalities.fmri.glm as glm

from fg_constants import *
from cross_validation import load_images, split, REGRESSORS
import ttest

memory = Memory(cachedir=JOBLIB_DIR, verbose=1)


def glm_test_volume(subj='sub001'):
    return get_volume(subj)[1]


def glm_predicted_volume(regressor, subj='sub001'):
    masker = get_masker()
    prediction_file = os.path.join(PREDICTED_DIR, regressor, subj, 'predicted_bold.nii.gz')
    return masker.transform(nib.load(predicted_bold))


def glm_volume(subj='sub001'):
    masker = get_masker()
    clean_data_dir = os.path.join(PREP_DATA_DIR, subj)
    Y = np.vstack(split(load_images(masker, clean_data_dir)))
    return Y


def glm_design(name):
    fun = REGRESSORS[name]
    regressors = [fun(j) for j in SEGLIST]
    X = np.vstack(split(regressors))
    return (X-X.mean(0))/X.std(0)


def glm_contrast(contrast, model, show=False):
    masker = get_masker()
    z = model.contrast(contrast).z_score()
    img = masker.inverse_transform(z)
    nib.save(img, 'z.nii.gz')
    return z


def get_masker(std=True):
    masker = NiftiMasker(nib.load(MASK_FILE), standardize=std, memory=memory)
    masker.fit()
    return masker


def load_test_bold(subj='sub001'):
    masker = get_masker()
    bold = nib.load('predicted_bold/%s_bold.nii.gz' % subj)
    return masker.transform(bold)


def load_regressor(regressor, subj='sub001'):
    pred = np.load('./predicted_bold/%s_%s_pred_reg.npy' % (subj, regressor))
    pred = (pred-pred.mean(0))/pred.std(0)
    return pred


def analyse_voxel(regressors, bold):
    model = glm.GeneralLinearModel(regressors)
    model.fit(bold)
    con = model.contrast([1,-1])
    return con.z_score(), con.p_value()


def compare_regressors(names, subj='sub001'):
    if len(names) > 2:
        raise NotImplementedError('Only two regressors are supported')

    masker = get_masker(std=False)
    corr1 = ttest.load_cv_map(names[0], 'sub001', masker)
    corr2 = ttest.load_cv_map(names[1], 'sub001', masker)
    mask = (np.vstack((corr1, corr2)) > 0.08).any(axis=0)
    idx = np.where(mask)[0]
    print idx.shape

    bold = load_test_bold(subj)
    regressors = [load_regressor(name, subj) for name in names]
    regressors = np.dstack(regressors)
    print regressors.shape
    p_values = np.zeros((bold.shape[1], ), dtype=np.float64)
    z_scores = np.zeros((bold.shape[1], ), dtype=np.float64)

    num_voxels = bold.shape[1]
    # for v in range(num_voxels):
    for v in idx:
        if v % 100 == 0:
            print 'Voxel %i' % v
        voxel_predictions = regressors[:, v, :]
        p_values[v], z_scores[v] = analyse_voxel(voxel_predictions, bold[:, v])

    masker = get_masker()
    img = masker.inverse_transform(p_values)
    nib.save(img, './glm2_p_%s_vs_%s.nii.gz' % (names[0], names[1]))
    img = masker.inverse_transform(z_scores)
    nib.save(img, './glm2_z_%s_vs_%s.nii.gz' % (names[0], names[1]))


if __name__ == '__main__':
    compare_regressors(['audio2', 'word2vec'])

