#!/usr/bin/python

import os
import numpy as np
import nibabel as nib
import cortex
from scipy import stats

from cross_validation import REGRESSORS
from fg_constants import *

import ttest

def load_map(map_name, subj):
    maps_dir = './maps/%s/%s' % (map_name, subj)
    out = nib.load(os.path.join(maps_dir, 'corr_cv.nii.gz'))
    return out.get_data()


def load_maps(map_name, subjects):
    return np.array(map(lambda s: load_map(map_name, s), subjects))


def mean_vol(vol):
    return np.mean(vol, axis=0)


def compute_diff_map(map1, map2):
    diff_m = map1 - map2
    # threshold = 0.1
    threshold = 0.08
    diff_m[np.logical_and(np.abs(map1) < threshold, np.abs(map2) < threshold)] = np.nan
    return diff_m


def get_volume(vol, vmin, vmax, cmap='hot'):
    return cortex.Volume(vol.T, 'fsaverage', 'grp_bold', vmin=vmin, vmax=vmax, cmap=cmap)


def get_ttest_map(map_name):
    subjs = map(lambda n: 'sub%03d' % n, SUBJECTS)
    vols = load_maps(map_name, subjs)
    # se = 1.0/np.sqrt(1199-6)
    popmean = np.arctanh(0.03)
    ttest = stats.ttest_1samp(np.arctanh(vols), popmean, axis=0)[0]
    ttest[ttest < 4] = np.NaN
    return get_volume(ttest, 4, 15)


def get_ttest_contrast(map_name1, map_name2):
    vol = ttest.make_ttest(map_name1, map_name2)[0].get_data()
    vol[np.logical_and(vol < 4, vol > -4)] = np.NaN
    return get_volume(vol, -10, 10, cmap='coldwarm')


def show_maps():
    maps = ['speech', 'word2vec', 'audio8', 'lda2']
    vol_objects = {name + ' ttest': get_ttest_map(name) for name in maps}
    # vol_objects = dict()
    contrasts = [('speech', 'speech_only'), ('log_energy', 'word2vec'), ('audio8', 'audio2'), ('audio8', 'speech_only'),
            ('audio2', 'word2vec'), ('lda2', 'word2vec'), ('lda2', 'audio2')]
    vol_objects_contrasts = {names[0] +' vs ' + names[1] + ' ttest': get_ttest_contrast(names[0], names[1]) for names in contrasts}
    vol_objects.update(vol_objects_contrasts)
    ds = cortex.dataset.Dataset(**vol_objects)
    web = cortex.webgl.show(ds, types=('inflated',), port=8081, open_browser=True)
    raw_input()


def show_map():
    sub_nums = map(int, raw_input('enter subject numbers --> ').split())
    if len(sub_nums) == 0:
        sub_nums = SUBJECTS
    subjs = map(lambda n: 'sub%03d' % n, sub_nums)

    maps = sorted(REGRESSORS)
    for i, m in enumerate(maps):
        print '%d. %s' % (i, m)

    map_nums = map(int, raw_input('Choose map --> ').split())
    if len(map_nums) == 1:
        map_name = maps[map_nums[0]]
        vols = load_maps(map_name, subjs)
        volumes = dict(zip(subjs, vols))
        # vol_objects = dict(map(lambda x: (map_name+' '+x[0], get_volume(x[1], 0.1, 0.4)), volumes.items()))
        vol_objects = dict()
        if len(vols) > 1:
            vols = np.arctanh(vols)
            se = 1.0/np.sqrt(1199-6)  # ~0.03
            popmean = np.arctanh(0.03)
            ttest = stats.ttest_1samp(vols, popmean, axis=0)[0]
            ttest[ttest < 4] = np.NaN
            vol_objects.update(**{map_name + ' ttest': get_volume(ttest, 4, 15)})
            std = np.tanh(np.std(vols, axis=0))
            mean = np.tanh(mean_vol(vols))
            vols = np.tanh(vols)
            vol_objects.update(**{map_name + ' std': get_volume(std, 0.05, 0.15)})
            vol_objects.update(**{map_name + ' mean': get_volume(mean, 0.1, 0.4)})
        ds = cortex.dataset.Dataset(**vol_objects)
    if len(map_nums) == 2:
        mname1 = maps[map_nums[0]]
        mname2 = maps[map_nums[1]]
        maps1 = load_maps(mname1, subjs)
        maps2 = load_maps(mname2, subjs)
        diff_maps = [compute_diff_map(x[0], x[1]) for x in zip(maps1, maps2)]
        vol = np.tanh(compute_diff_map(mean_vol(np.arctanh(maps1)), mean_vol(np.arctanh(maps2))))

        d = get_volume(vol, -0.1, 0.1, cmap='coldwarm')
        diff_vols = [get_volume(vol, -0.1, 0.1, cmap='coldwarm') for vol in diff_maps]

        # vol_objects = dict(zip(subjs, diff_vols))
        vol_objects = dict()
        vol_objects[mname1+'-'+mname2] = d

        ds = cortex.dataset.Dataset(**vol_objects)

    web = cortex.webgl.show(ds, types=('inflated',), port=8081, open_browser=True)
    raw_input()


while True:
    show_map()
    # show_maps()
