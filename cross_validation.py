#!/usr/bin/python

import os

import numpy as np
from scipy.ndimage.filters import convolve1d
from scipy import signal
import scipy.io.wavfile as wav

from sklearn import linear_model, pipeline, preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from joblib import Memory
from joblib import Parallel, delayed

import nibabel as nib
from nilearn.input_data import NiftiMasker

from features import mfcc, logfbank, fbank

from read_subtitles import get_word2vec_vocab, \
    get_random_vocab, build_word_matrix, \
    build_people_matrix, get_bag_of_words_vocab, \
    get_glove_vocab, build_lda_matrix

from fg_constants import *

memory = Memory(cachedir=JOBLIB_DIR, verbose=1)

# TODO if no preprocessed data, preprocess it?

if not os.path.exists(MAPS_DIR):
    os.mkdir(MAPS_DIR)


def make_lags(matrix, num, lags=LAGS, framesTR=20):
    if len(matrix.shape) == 1:
        matrix = matrix[:, np.newaxis]
    matrix = matrix[:(matrix.shape[0]/framesTR*framesTR)]
    dm = [np.roll(matrix, l, axis=0) for l in lags]
    window = np.ones((framesTR))
    # window = signal.hamming(framesTR*2+1)
    # window = gamma_difference_hrf(TR, 20, 16.0, delay=4)
    for i in range(len(lags)):
        dm[i][:lags[i]] = 0
        dm[i] = convolve1d(dm[i], window, mode='constant', axis=0)[::framesTR]/framesTR
    dm = np.hstack(dm)
    if SCANS[num] < dm.shape[0]:
        dm = dm[:SCANS[num]]
    dm = dm[SESSION_BOLD_OFFSET:]
    return dm


def load_audio(num, mono=True):
    filename = os.path.join(AUDIO_DIR, 'fg_ad_seg%i.wav' % num)
    (rate, sig) = wav.read(filename)
    if mono:
        sig = np.array(sig, dtype=np.double).mean(axis=1)
    return rate, sig


@memory.cache
def mfcc_lags(num, window=0.1):
    rate, audio = load_audio(num)
    features = mfcc(audio, rate, winlen=window, winstep=window)
    return make_lags(features, num, framesTR=int(2.0/window))


@memory.cache
def log_energy_lags(num, window=0.1):
    rate, audio = load_audio(num)
    _, energy = fbank(audio, rate, winlen=window, winstep=window)
    energy = np.log(energy)
    return make_lags(energy, num, framesTR=int(2.0/window))


@memory.cache
def audio_lags(num, bins=8):
    bands = np.load(os.path.join(CQ_DIR, str(bins), 'run00%i.npy' % num))
    return make_lags(bands, num)


@memory.cache
def speakers_lags(num, limit=10):
    features = build_people_matrix(num, limit=limit)
    features = features[:(features.shape[0]/20*20)]
    return make_lags(features, num)


@memory.cache
def speech2_lags(num):
    features = build_people_matrix(num, 200)
    features = features[:(features.shape[0]/20*20)]
    features_new = np.zeros((features.shape[0], 2), dtype=np.float32)
    features_new[:, 0] = features[:, 0]
    features_new[:, 1] = features[:, 1:].max(axis=1)
    return make_lags(features_new, num)


@memory.cache
def speech1_lags(num):
    features = build_people_matrix(num, 200)
    features = features[:(features.shape[0]/20*20)]
    features_new = np.zeros((features.shape[0], 1), dtype=np.float32)
    features_new[:, 0] = features.max(axis=1)
    return make_lags(features_new, num)


def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=np.ones(window_len,'d')
    else:
            w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

@memory.cache
def lda_lags(num):
    features = build_lda_matrix(num)
    for i in range(features.shape[1]):
        features[:, i] = smooth(features[:, i], window_len=41)
    return make_lags(features, num)


def lda2_lags(num):
    features1 = build_lda_matrix(num, 'dialogs')
    features2 = build_lda_matrix(num, 'annotations')
    features = np.hstack((features1, features2))
    for i in range(features.shape[1]):
        features[:, i] = smooth(features[:, i], window_len=41)
    return make_lags(features, num)


def embedding_lags(num, vocab_fun):
    matrix = build_word_matrix(num)
    vocab_matrix = vocab_fun()
    features = np.dot(matrix, vocab_matrix)
    return make_lags(features, num)


def combine_regressors(num, funs):
    mats = tuple(REGRESSORS[func_name](num) for func_name in funs)
    return np.hstack(mats)


def load_images(clean_data_dir):
    masker = get_masker()
    imgs = [os.path.join(clean_data_dir, 'run00%i.nii.gz' % j) for j in SEGLIST]
    segments = list(masker.transform_imgs(imgs, n_jobs=4))

    for i in SEGLIST:
        print i, segments[i].shape
        if segments[i].shape[0] > SCANS[i]:
            segments[i] = segments[i][:SCANS[i]]
        segments[i] = segments[i][SESSION_BOLD_OFFSET:]
    return segments


def validate(X, Y, X_test, alpha):
    print 'validate(), X=', X.shape, ' Y=', Y.shape, ' X_test=', X_test.shape
    clf = linear_model.Ridge(alpha=alpha)
    scaler = preprocessing.StandardScaler()
    pp = pipeline.Pipeline([('scaler', scaler), ('classifier', clf)])

    print 'Fitting...'
    pp.fit(X, Y)
    print 'Predicting...'
    Y_pred = pp.predict(X_test)
    print 'Y_pred=', Y_pred.shape
    return Y_pred, pp


def split(segments):
    print 'Splitting segments...'
    print map(lambda x: x.shape, segments)

    splits = map(lambda x: (x[x.shape[0]/3:], x[:x.shape[0]/3]), segments)
    train, test = zip(*splits)
    train_array = np.vstack(train)
    test_array = np.vstack(test)
    return train_array, test_array


def cross_validation(masker, data, regressors, alpha, maps_dir, best_num=10):
    X, X_test = split(regressors)
    Y, Y_test = data
    print X.shape, Y.shape, X_test.shape, Y_test.shape
    Y_pred, _ = validate(X, Y, X_test, alpha)
    Ytc = (Y_test - Y_test.mean(0)) / Y_test.std(0)
    Ypc = (Y_pred - Y_pred.mean(0)) / Y_pred.std(0)
    corr = (Ytc * Ypc).mean(0)
    corr = np.nan_to_num(corr)
    out = masker.inverse_transform(corr)
    nib.save(out, os.path.join(maps_dir, 'corr_cv.nii.gz'))
    best = corr[corr.argsort()[::-1]][:best_num]
    return out, best


REGRESSORS = {
    'audio2': lambda n: audio_lags(n, bins=2),
    'audio8': lambda n: audio_lags(n, bins=8),
    'mfcc': mfcc_lags,
    'log_energy': log_energy_lags,
    'word2vec': lambda n: embedding_lags(n, get_word2vec_vocab),
    'glove': lambda n: embedding_lags(n, get_glove_vocab),
    'random_embedding': lambda n: embedding_lags(n, get_random_vocab),
    'lda': lda_lags,
    'lda2': lda2_lags,
    'speakers': speakers_lags,
    'speech2': speech_lags,
    'speech1': speech_only_lags,
    'audio_w2v_speech_lda2': lambda n: combine_regressors(n, ['audio8', 'lda2', 'word2vec', 'speech2']),
    }


def get_masker():
    masker = NiftiMasker(nib.load(MASK_FILE), standardize=True, memory=memory)
    masker.fit()
    return masker


def compute_model(data, reg_name, sub_num, alpha=ALPHA):
    subj = 'sub%03d'%sub_num
    masker = get_masker()

    fun = REGRESSORS[reg_name]
    print 'Trying the model %s' % reg_name
    regressors = [fun(j) for j in SEGLIST]

    maps_dir = os.path.join(MAPS_DIR, reg_name, subj)
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)

    predicted_dir = os.path.join(PREDICTED_DIR, reg_name, subj)
    if not os.path.exists(predicted_dir):
        os.makedirs(predicted_dir)

    brain, best10 = cross_validation(masker, data, regressors, alpha, maps_dir)
    print best10


def get_subject_bold(sub_num):
    return split(load_images(os.path.join(PREP_DATA_DIR, 'sub%03d' % sub_num)))


def process_subject(sub_num):
    print 'Processing subj %i' % sub_num
    data = get_subject_bold(sub_num)
    print 'Images are loaded'

    regs = REGRESSORS.keys()
    for name in regs:
        compute_model(data, name, sub_num)


if __name__ == '__main__':
    subj_list = SUBJECTS
    for s in subj_list:
        process_subject(s)
