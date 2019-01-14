"""
new tests
"""

import numpy as np
np.warnings.filterwarnings('ignore')
import codecs
import os
from custom_types import IntListList, DictList, Dataset
from uralicNLP.ud_tools import UD_collection
from test_sentences import change_ud_morphology, give_all_possibilities
from lang import languages
from tqdm import tqdm
from collections import OrderedDict

# for stats
from scipy.stats import kendalltau
from rbo import rbo

def manhattan_dist(a,b):
    """
    manhattan distance mapped to the range [0,1] by dividing the distance by the maximum possible distance (maxp)
    """
    assert set(list(a)) == set(list(b))
    assert len(a) == len(b)
    maxp = np.abs(np.arange(len(a)) - np.arange(len(a))[::-1]).sum()
    return 1 - (float(np.abs(np.atleast_1d(a) - np.atleast_1d(b)).sum()) / maxp)

def kendall_tau_dist(a,b):
    """
    kendall tau distance mapped to the range [0,1]
    """
    assert set(list(a)) == set(list(b))
    assert len(a) == len(b)
    tau, pvalue = kendalltau(a,b)
    return (tau + 1.) / 2.

def rank_based_overlap(a,b):
    return rbo(list(a),list(b),p=0.9)

# ==============================================================================
# Dataset Functions
# ==============================================================================

def get_sentences(filepaths):
    filepaths = np.unique(np.atleast_1d(filepaths))
    sent = []
    for filepath in filepaths:
        sent += UD_collection(codecs.open(filepath, encoding="utf-8")).sentences
    return sent

def split_sentences(sentences, split=0.9):
    # randomly split into test and train probabilities
    n_train_sentences = int(len(sentences) * split)
    n_test_sentences = len(sentences) - n_train_sentences
    assert n_train_sentences > 0
    assert n_test_sentences > 0
    labels = np.asarray(([0] * n_train_sentences) + ([1] * n_test_sentences))
    np.random.shuffle(labels)
    train_sentences = [sentences[i] for i in np.where(labels==0)[0]]
    test_sentences = [sentences[i] for i in np.where(labels==1)[0]]
    return train_sentences, test_sentences

"""
def get_random_labels(n, split=0.9, seed=689012349):
    np.random.seed(seed)
    n_train_sentences = int(n * split)
    n_test_sentences = int(n - n_train_sentences)
    assert (n_train_sentences > 0) and (n_test_sentences > 0)
    labels = np.asarray(([0] * n_train_sentences) + ([1] * n_test_sentences))
    np.random.shuffle(labels)
    return np.where(labels==0)[0], np.where(labels==1)[0]
"""

def get_random_labels(n, split):
    idx = np.argsort(np.random.rand(n))
    split = np.cumsum((np.atleast_1d(split) * n).astype(np.int32))
    if len(split) == 3:
        return idx[:split[0]], idx[split[0]:split[1]], idx[split[1]:]
    elif len(split) == 2:
        return idx[:split[0]], idx[split[0]:]
    else:
        raise ValueError("SPLIT NOT SUPPORTED {}".format(split))

# dataset then is a collection of target sentences and all possible readings where each word is an intlistlist of possiblities
# how much does each reading have in common with target

def random_reading(readings, tmin, tmax):
    ll = [r.values() for r in readings]
    pad = np.max(list(map(len,ll)))
    z = np.array([i + [-1]*(pad-len(i)) for i in ll])
    s = 0
    idx = {}
    x = np.copy(z)
    trials = 0
    while s < tmin:
        locs = zip(*np.where((x > 0) & (x <= (tmax-s))))
        if len(locs) == 0:
            s = 0
            idx = {}
            x = np.copy(z)
            if trials >= 50:
                return None
            trials += 1
        else:
            i,j = locs[np.random.randint(len(locs))]
            s += x[i,j]
            x[i,:] = -1
            idx[i] = j
    ss = np.sum([z[i,j] for i,j in idx.items()])
    assert (ss >= tmin) & (ss <= tmax)
    return IntListList([r.keys()[idx.get(i,np.argmax(z[i]==0))] for i,r in enumerate(readings)])

def create_dataset(lang, force=False):
    sentences = get_sentences(languages[lang].values())
    filepath = "DATASET_{}_{}.npz".format("MASTER", lang)
    if os.path.exists(filepath) and not force:
        return Dataset(filepath=filepath)

    d = Dataset(["target", "readings"])
    for sentence in tqdm(sentences):
        target = IntListList(sentence)
        readings = []
        for i, word in enumerate(give_all_possibilities(sentence, lang=lang)):
            words = []
            for x in IntListList(word, force_dict=True):
                inter = set(x).intersection(set(target[i]))
                words.append((tuple(sorted(x)), len(target[i]) - len(inter)))
            words.append((tuple(sorted(target[i])), 0))
            readings.append(OrderedDict(words))
        d.add_example({
            "target" : IntListList(sentence),
            "readings" : readings
        })

    d.save(filepath)
    return d

def make_nn_dataset(d, maxlen=200):
    inputs = []
    outputs = []

    for sentence, reading in zip(d["target"], d["readings"]):
        input = reduce(operator.add, [x+[223] for x in sentence])
        inputs.append( input )
        outputs.append( 1 )
        bad_input = random_reading(reading,2,100)
        if bad_input is not None:
            bad_input = reduce(operator.add, [x+[223] for x in bad_input])
            inputs.append( bad_input )
            outputs.append( 0 )

    inputs = pad_sequences(inputs, maxlen)
    outputs = np.array(outputs)
    order = np.argsort(np.random.rand(len(inputs)))

    return inputs[order], outputs[order]

if __name__ == "__main__":

    # evaluate sentences using an RNN

    from scoring import ScoreSentenceByLinearRegression
    from test_sentences import run_spmf_full
    import argparse

    arg_parser = argparse.ArgumentParser(description='Run tests')
    arg_parser.add_argument('--train_lang',type=str, nargs='+', default=["sme"])
    arg_parser.add_argument('--test_lang', type=str, nargs='+', default=["sme"])
    arg_parser.add_argument('--seed', type=int, default=1234)
    args = arg_parser.parse_args()

    assert len(args.test_lang) == 1
    np.random.seed(args.seed)

    train_d = Dataset(keys=["target", "readings"])
    valid_d = Dataset(keys=["target", "readings"])
    test_d = Dataset(keys=["target", "readings"])
    for lang in args.train_lang:
        d = create_dataset(lang)
        if lang in args.test_lang:
            ridx, vidx, tidx = get_random_labels(len(d["target"]), (.8,.1,.1))
            train_d += d.get_subset(ridx)
            test_d += d.get_subset(tidx)
            valid_d += d.get_subset(vidx)
        else:
            ridx, vidx = get_random_labels(len(d["target"]), (.9,.1))
            train_d += d.get_subset(ridx)
            valid_d += d.get_subset(vidx)

    if not args.test_lang[0] in args.train_lang:
        test_d = create_dataset(args.test_lang[0])


    print "TRAIN DATASET SIZE ({})".format(len(train_d["target"]))
    print "TEST DATASET SIZE ({})".format(len(test_d["target"]))
    print "VALID DATASET SIZE ({})".format(len(valid_d["target"]))

    import operator
    from keras.models import Sequential
    from keras import layers
    from keras.preprocessing.sequence import pad_sequences
    from keras.callbacks import ModelCheckpoint, EarlyStopping

    # create some test datasets
    train_inputs, train_outputs = make_nn_dataset(train_d, maxlen=200)
    valid_inputs, valid_outputs = make_nn_dataset(valid_d, maxlen=200)

    num_classes = 224
    embed_dim = 64
    hidden_dim = 128

    model = Sequential()
    model.add(layers.Embedding(num_classes, embed_dim))
    model.add(layers.LSTM(hidden_dim))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)]

    model.fit(x=train_inputs, y=train_outputs, epochs=50, batch_size=32, validation_data=(valid_inputs, valid_outputs), callbacks=callbacks)

    test_inputs, test_outputs = make_nn_dataset(test_d, maxlen=200)
    mean_accuracy = model.evaluate(test_inputs, test_outputs)[1]

    print(mean_accuracy)




#
