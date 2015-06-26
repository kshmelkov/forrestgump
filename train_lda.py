#!/usr/bin/python

from read_subtitles import extract_corpus
from fg_constants import *
from gensim import corpora, models

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_lda(corpus, model_suffix, num_topics=5):
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=2)
    dictionary.save('./fg_%s.dict' % model_suffix)

    corpus = [dictionary.doc2bow(text) for text in corpus]
    # corpora.MmCorpus.serialize('./fg.mm', corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    tfidf.save('./fg_gensim_model_%s.tfidf' % model_suffix)

    # lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics) # initialize an LSI transformation
    # lsi.save('./fg_gensim_model.lsi')

    lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=2000, passes=100)
    lda.save('./fg_gensim_model_%s.lda' % model_suffix)
    lda.print_topics(num_topics)

dialogs = extract_corpus(DIALOGS)
annotations = extract_corpus(BLIND_ANNOTATIONS)

train_lda(dialogs, 'dialogs', num_topics=20)
train_lda(annotations, 'annotations', num_topics=20)
train_lda(dialogs+annotations, '', num_topics=20)
