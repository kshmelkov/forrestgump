# What is it?

Here is the source code developed during an internship "Decoding auditory
cortex using high-dimensional regression" at PARIETAL (Inria lab).

It is released under BSD 3-clause license.

# How can I reproduce results?

First of all, you need stimuli: they were communicated privately by Michael
Hanke. So they are absent in this repo. Essentially it is audio files and text
annotations (in JSON).

Then you should download [original dataset](http://studyforrest.org/). I would
recommend to rsync from [here](http://psydata.ovgu.de/forrest_gump/). Put your
paths in `fg_constants.py` and preprocess data using `preprocess.py`. Then look
at `cross_validation.py` to choose what exactly you compute. If audio, run
`convert_audio.py`, if word2vec, download a wikidump (German) and run
`train_word2vec.py`...

After that, run `cross_validation.py` and `view_batch_3d.py` to look at
results.

# Dependencies

Plenty of them. NumPy, SciPy, scikit-learn, gensim, nilearn, nsgt. Optionally,
pycortex or FSL to visualization, glove, FFTW,
[features](https://github.com/jameslyons/python_speech_features).

