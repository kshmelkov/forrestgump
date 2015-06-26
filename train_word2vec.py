#!/usr/bin/python

from gensim.corpora import WikiCorpus
from gensim.models.word2vec import Word2Vec

corpus = WikiCorpus('dewiki-latest-pages-articles.xml.bz2', dictionary=False, lemmatize=False)

model = Word2Vec(size=300, window=7, min_count=7, workers=4, negative=10, hs=0)
model.build_vocab(corpus.get_texts())
model.train(corpus.get_texts())
model.init_sims(replace=True)
model.save('dewiki.w2v')
