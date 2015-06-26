#!/usr/bin/python

from os import path
from nilearn import image
from nibabel import load, save, Nifti1Image
from joblib import Parallel, delayed

import cortex
import numpy as np

data_dir = '/volatile/accounts/kshmelkov/data'
mask_dir = './masks'


def compute_subj_mask(subj):
    num = 0

    print 'Loading BOLD subject #%i...' % subj
    bold = load(path.join(data_dir, 'sub%03d/BOLD/task001_run00%i/bold_dico.nii.gz' % (subj, num+1)))

    print 'Averaging BOLD...'
    avg = image.mean_img(bold)

    print 'Saving average EPI...'
    avg_path = path.join(mask_dir, 'mean_sub%03d_run00%i.nii.gz' % (subj, num+1))
    affine = avg.get_affine()
    save(avg, avg_path)

    print 'Aligning the surface...'
    cortex.align.automatic('sub%03d' % subj, 'auto01', avg_path)

    m_thick = cortex.db.get_mask('sub%03d' % subj, 'auto01', type='thick')
    print 'Thick mask is computed: %i voxels' % m_thick.sum()
    m_thin = cortex.db.get_mask('sub%03d' % subj, 'auto01', type='thin')
    print 'Thin mask is computed: %i voxels' % m_thin.sum()

# .T is important here, because pycortex and nibabel have different
# assumptions about the axis order
    thick = Nifti1Image(np.float32(m_thick.T), affine)
    thin = Nifti1Image(np.float32(m_thin.T), affine)

    save(thick, path.join(mask_dir, 'sub%03d_mask_thick.nii.gz' % subj))
    save(thin, path.join(mask_dir, 'sub%03d_mask_thin.nii.gz' % subj))

if __name__ == '__main__':
    Parallel(n_jobs=2)(delayed(compute_subj_mask)(i) for i in range(11, 16))
