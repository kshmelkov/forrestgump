#!/usr/bin/python

import json
import nltk
import numpy as np
from math import ceil, trunc
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models import Word2Vec
from joblib import Memory
import pandas as pd
import matplotlib.pylab as plt
import unicodedata

from fg_constants import *

memory = Memory('/tmp')

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopword_list = stopwords.words('german')


def parse_json(filename):
    with open(filename, 'r') as annotations:
        data = json.load(annotations)
        for entry in data['annotations']:
            yield entry


def write_word_freqs(filename, word_freq):
    for d in parse_json(filename):
        text = d['parsed']['text']
        for w in parse_sentence(text):
            if w in word_freq:
                word_freq[w] += 1


def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def parse_sentence(text, strip_unicode=False):
    if strip_unicode:
        text = strip_accents(text)
    return tokenizer.tokenize(text.lower())


def words_from_json(filename):
    words = set()
    for d in parse_json(filename):
        text = d['parsed']['text']
        words.update(parse_sentence(text))
    return words


def get_forrest_gump_corpus():
    corpus = extract_text(BLIND_ANNOTATIONS) + extract_text(DIALOGS)
    with open('fg_corpus', 'w') as f:
        f.write(u' '.join(corpus).encode('utf-8'))


def extract_text(filename):
    corpus = []
    for d in parse_json(filename):
        text = d['parsed']['text']
        tokens = parse_sentence(text)
        corpus.extend(tokens)
    return corpus


def extract_corpus(filename):
    corpus = []
    last_time = -1
    for d in parse_json(filename):
        text = d['parsed']['text']
        tokens = parse_sentence(text)#, strip_unicode=True)
        # TODO put it in gensim api
        tokens = [word for word in tokens if word not in stopword_list]

        begin = float(d['begin'])/1000
        end = float(d['end'])/1000
        if len(tokens) == 0:
            continue
        if last_time > 0 and begin-last_time < 6:
            corpus[-1].extend(tokens)
        else:
            corpus.append(tokens)
        last_time = end
    return corpus


@memory.cache
def build_lda_matrix(seg_num, suffix=''):
    if suffix == '':
        df = pd.DataFrame.from_csv('subtitles/fg_ad_seg%i.csv' % seg_num)
    else:
        df = pd.DataFrame.from_csv('subtitles/fg_ad_seg%i_%s.csv' % (seg_num, suffix))
    dictionary = corpora.dictionary.Dictionary.load('./fg_%s.dict' % suffix)
    tfidf = models.tfidfmodel.TfidfModel.load('./fg_gensim_model_%s.tfidf' % suffix)
    lda = models.ldamodel.LdaModel.load('./fg_gensim_model_%s.lda' % suffix)

    matrix = np.zeros((SEGMENTS[seg_num]*10, lda.num_topics), dtype=np.float64)

    for idx, row in df.iterrows():
        begin = int(row['begin'])/100
        end = int(row['end'])/100
        text = row['text']
        text = text.decode('utf-8')
        tokens = parse_sentence(text)
        duration = end-begin
        topics = lda[tfidf[dictionary.doc2bow(tokens)]]
        lda_topics = np.array(map(lambda x: x[1], topics))
        if lda_topics.shape == (lda.num_topics,):
            matrix[begin:end, :] += lda_topics
    return matrix


@memory.cache
def build_word_matrix(seg_num):
    df = pd.DataFrame.from_csv('subtitles/fg_ad_seg%i.csv' % seg_num)

    vocab = build_vocab()

    matrix = np.zeros((SEGMENTS[seg_num]*10, len(vocab)), dtype=np.int)

    for idx, row in df.iterrows():
        begin = int(row['begin'])/100
        end = int(row['end'])/100
        text = row['text']
        tokens = parse_sentence(text)
        letters_count = sum(map(len, tokens))
        duration = end-begin
        shift = 0
        for w in tokens:
            if w in vocab:
                frac = float(len(w))/letters_count
                st = begin+trunc(float(shift)/letters_count*duration)
                matrix[st:(st + ceil(frac*duration)), vocab[w]] = 1
                shift += len(w)
    return matrix


@memory.cache
def build_vocab():
    vocab = (words_from_json(DIALOGS) | words_from_json(BLIND_ANNOTATIONS))
    # vocab = vocab.difference(stopword_list)

    word_freq = dict()
    for w in vocab:
        word_freq[w] = 0

    write_word_freqs(DIALOGS, word_freq)
    write_word_freqs(BLIND_ANNOTATIONS, word_freq)
    words = sorted(word_freq, key=lambda x: word_freq[x], reverse=True)
    print 'Most frequent words: ', \
        sorted([(word_freq[w], w) for w in words[:300]])
    return {word: idx for idx, word in enumerate(words)}


@memory.cache
def get_bag_of_words_vocab(limit=300):
    vocab = build_vocab()
    vocab_m = np.zeros((len(vocab), limit), dtype=np.int32)
    vocab_m[:limit, :limit] = np.eye(limit, dtype=np.int32)
    return vocab_m


@memory.cache
def get_random_vocab():
    print 'Loading random pseudo word2vec model'
    np.random.seed(12345)
    vocab = build_vocab()
    vocab_m = np.random.randn(len(vocab), num_word2vec_features)
    return vocab_m


@memory.cache
def get_word2vec_vocab():
    print 'Loading word2vec model'
    w2v = Word2Vec.load('dewiki.w2v')
    print 'building vocabulary'
    vocab = build_vocab()
    print 'Building vocab matrix from word2vec model'
    vocab_m = np.zeros((len(vocab), num_word2vec_features), dtype=np.float)

    misses = 0
    for word in vocab:
        idx = vocab[word]
        if word in w2v:
            vocab_m[idx] = w2v[word]
        else:
            #print 'word2vec model miss: ', word
            misses += 1

    print '%i misses from %i words' % (misses, len(vocab))

    return vocab_m


@memory.cache
def get_glove_vocab():
    print 'Loading glove model'
    glove = {}
    with open('glove.txt', 'r') as f:
        for line in f:
            a = line.split()
            word = a[0]
            vec = np.array(map(float, a[1:]), dtype=np.float)
            glove[word] = vec
    print 'building vocabulary'
    vocab = build_vocab()
    print 'Building vocab matrix from glove model'
    vocab_m = np.zeros((len(vocab), num_glove_features), dtype=np.float)

    misses = 0
    for word in vocab:
        idx = vocab[word]
        if word in glove:
            vocab_m[idx] = glove[word]
        else:
            #print 'glove model miss: ', word
            misses += 1

    print '%i misses from %i words' % (misses, len(vocab))

    return vocab_m


def write_people_freqs(filename, people_freq):
    for d in parse_json(filename):
        person = d['parsed']['person']
        people_freq[person] += 1


def people_from_json(filename):
    people = set()
    for d in parse_json(filename):
        person = d['parsed']['person']
        people.add(person)
    return people


@memory.cache
def build_people_matrix(seg_num, limit=10):
    df = pd.DataFrame.from_csv('subtitles/fg_ad_seg%i.csv' % seg_num)
    people = build_people_list(most_freq=limit)
    matrix = np.zeros((SEGMENTS[seg_num]*10, len(people)), dtype=np.int)

    for idx, row in df.iterrows():
        begin = int(row['begin'])/100
        end = int(row['end'])/100
        person = row['person']
        if person in people:
            matrix[begin:end, people[person]] = 1
    return matrix


@memory.cache
def build_people_list(most_freq=10):
    parsed_people = people_from_json(DIALOGS)

    people_freq = dict()
    for w in parsed_people:
        people_freq[w] = 0

    write_people_freqs(DIALOGS, people_freq)
    people = sorted(people_freq, key=lambda x: people_freq[x], reverse=True)
    people = people[:most_freq]
    print 'Most frequent people: ', \
        sorted([(people_freq[w], w) for w in people], reverse=True)
    people_list = {word: (idx+1) for idx, word in enumerate(people)}
    people_list['NARRATOR'] = 0
    return people_list
