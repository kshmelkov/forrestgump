#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

def load_image(reg, subj='sub016'):
    path = './maps/%s/%s/corr_cv.nii.gz' % (reg, subj)
    img = nib.load(path)
    return img

if __name__ == '__main__':
    regressors = ['speech', 'word2vec', 'audio2']
    for s in regressors:
        img = load_image(s)
        plotting.plot_glass_brain(img, threshold=None, vmin=0.1, vmax=0.5, colorbar=True, title=s)

    plt.show()


