"""
new tests
('kpv', 189)
('myv', 1550)
('sme', 3122)
('fin', 13772)
('est', 23564)
"""

#import choix
import json
import functools
import itertools
import numpy as np
np.warnings.filterwarnings('ignore')
import codecs
import random
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

"""
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
"""

def create_dataset(lang, force=False):
    sentences = get_sentences(languages[lang].values())
    filepath = "DATASET_{}_{}.npz".format("MASTER", lang)
    if os.path.exists(filepath) and not force:
        return Dataset(filepath=filepath)

    d = Dataset(["target", "readings"])
    max_diffs = []
    for sentence in tqdm(sentences):
        target = IntListList(sentence)
        readings = []
        max_diff = 0
        for i, word in enumerate(give_all_possibilities(sentence, lang=lang)):
            words = [(tuple(sorted(target[i])), 0)]
            diffs = [0]
            for x in IntListList(word, force_dict=True):
                diff = len(set(x).symmetric_difference(set(target[i])))
                words.append((tuple(sorted(x)), diff))
                diffs.append(diff)
            readings.append(OrderedDict(words))
            max_diff += np.max(diffs)
        d.add_example({
            "target" : IntListList(sentence),
            "readings" : readings
        })
        max_diffs.append( max_diff )

    # to visualize max diff distribution
    #print("[" + ", ".join(list(map(str, np.bincount(max_diffs)))) + "]")

    d.save(filepath)
    return d

def make_nn_dataset(d, maxlen=200):
    inputs = []
    outputs = []
    all_inputs = {i : [] for i in range(11)}

    for sentence, reading in zip(d["target"], d["readings"]):
        input = reduce(operator.add, [x+[223] for x in sentence])
        inputs.append( input )
        outputs.append( 1 )
        bad_input = random_reading(reading,2,100)
        if bad_input is not None:
            bad_input = reduce(operator.add, [x+[223] for x in bad_input])
            inputs.append( bad_input )
            outputs.append( 0 )

        # more comprehensive inputs
        all_inputs[0].append( input )
        for i in range(1,11):
            bad_input = random_reading(reading,i,i)
            if bad_input is not None:
                bad_input = reduce(operator.add, [x+[223] for x in bad_input])
                all_inputs[i].append(bad_input)

    all_inputs = {k:pad_sequences(v,maxlen) for k,v in all_inputs.items()}
    inputs = pad_sequences(inputs, maxlen)
    outputs = np.array(outputs)
    order = np.argsort(np.random.rand(len(inputs)))

    return inputs[order], outputs[order], all_inputs

def make_comparison_dataset(d, maxlen, n=1000, n_readings=10, binsize=10, maxbin=5):
    import random

    def to_vector(x):
        flatx = functools.reduce(operator.add, [sorted(list(xx))+[223] for xx in x])
        return pad_sequences([flatx],maxlen=maxlen)[0]

    def random_reading(readings):
        reading, diff = zip(*[random.choice(list(w.items())) for w in readings])
        return to_vector(reading), int(np.sum(list(diff)))

    sents = {0 : []}
    for sentence, readings in zip(d["target"], d["readings"]):

        sents[0].append( to_vector(sentence) )
        for i in range(n_readings):
            reading, diff = random_reading(readings)
            diff = (diff // binsize) + 1 # CREATING BINS
            if not diff in sents:
                sents[diff] = []
            sents[diff].append(reading)

    # NOTE : this way each distance combination is equally likely
    #idx = list(itertools.combinations(list(sents.keys()),2))
    idx = list(itertools.combinations(range(maxbin),2))

    inputs_a = []
    inputs_b = []
    outputs = []
    ids = []
    for i in range(n):
        a,b = random.sample(random.choice(idx),2)
        inputs_a.append( random.choice(sents[a]) )
        inputs_b.append( random.choice(sents[b]) )
        outputs.append( int(a < b) )
        ids.append((a,b))
    inputs = [np.array(inputs_a), np.array(inputs_b)]
    return inputs, np.array(outputs), np.array(ids)

def make_rank_dataset(d, maxlen, n_readings=10):

    def to_vector(x):
        flatx = functools.reduce(operator.add, [sorted(list(xx))+[223] for xx in x])
        return pad_sequences([flatx],maxlen=maxlen)[0]

    def random_reading(readings):
        reading, diff = zip(*[random.choice(list(w.items())) for w in readings])
        return to_vector(reading), int(np.sum(list(diff)))

    inputs = []
    for sentence, readings in zip(d["target"], d["readings"]):

        input = [to_vector(sentence)]
        for i in range(n_readings):
            reading, _ = random_reading(readings)
            input.append(reading)
        inputs.append(input)
    return inputs

if __name__ == "__main__":

    # evaluate sentences using an RNN
    # print the size of each language
    #for lang in ["kpv", "myv", "sme", "fin", "est"]:
    #    print( lang, len(get_sentences(languages[lang].values())) )

    from scoring import ScoreSentenceByLinearRegression
    from test_sentences import run_spmf_full
    import argparse
    import operator
    import pickle
    from keras.models import Sequential
    from keras import layers
    from keras.models import Model
    from keras.preprocessing.sequence import pad_sequences
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.utils import Sequence
    from keras.models import load_model

    arg_parser = argparse.ArgumentParser(description='Run tests')
    arg_parser.add_argument('--train_lang',type=str, nargs='+', default=["sme"])
    arg_parser.add_argument('--test_lang', type=str, nargs='+', default=["sme"])
    arg_parser.add_argument('--seed', type=int, required=True)
    arg_parser.add_argument('--force', type=int, default=0)
    arg_parser.add_argument('--num_epochs', type=int, default=50)
    arg_parser.add_argument('--maxlen', type=int, default=200)
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--model_filepath', type=str, required=True)
    arg_parser.add_argument('--result_filepath', type=str, required=True)
    arg_parser.add_argument('--binsize', type=int, default=10)
    arg_parser.add_argument('--maxbin', type=int, default=5)
    args = arg_parser.parse_args()

    assert len(args.test_lang) == 1
    np.random.seed(args.seed)

    train_d = Dataset(keys=["target", "readings"])
    valid_d = Dataset(keys=["target", "readings"])
    for lang in args.train_lang:
        train_d += Dataset(
            filepath="DATASET_SPLIT_TRAIN_{}_{}.npz".format(lang, args.seed))
        valid_d += Dataset(
            filepath="DATASET_SPLIT_VALID_{}_{}.npz".format(lang, args.seed))

    test_d = Dataset(
        filepath="DATASET_SPLIT_TEST_{}_{}.npz".format(args.test_lang[0], args.seed))

    print( "TRAIN DATASET SIZE ({})".format(len(train_d["target"])) )
    print( "VALID DATASET SIZE ({})".format(len(valid_d["target"])) )
    print( "TEST DATASET SIZE ({})".format(len(test_d["target"])) )

    # OLD MODEL
    """
    # create some test datasets
    train_inputs, train_outputs, all_train = make_nn_dataset(
        train_d, maxlen=args.maxlen)
    valid_inputs, valid_outputs, all_valid = make_nn_dataset(
        valid_d, maxlen=args.maxlen)
    test_inputs, test_outputs, all_test = make_nn_dataset(
        test_d, maxlen=args.maxlen)

    print "TRAIN LABELS {}".format(np.bincount(train_outputs.flatten()))
    print "VALID LABELS {}".format(np.bincount(valid_outputs.flatten()))

    print {k:len(v) for k,v in all_train.items()}
    print {k:len(v) for k,v in all_valid.items()}
    print {k:len(v) for k,v in all_test.items()}

    # normal model
    num_classes = 224
    embed_dim = 64
    hidden_dim = 128

    model = Sequential()
    model.add(layers.Embedding(num_classes, embed_dim))
    model.add(layers.LSTM(hidden_dim))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    callbacks = [EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)]

    model.fit(x=train_inputs, y=train_outputs, epochs=5, batch_size=32, validation_data=(valid_inputs, valid_outputs), callbacks=callbacks)

    #mean_accuracy = model.evaluate(test_inputs, test_outputs)[1]


    # test with different distances and report mean accuracy
    for i in range(11):
        print(i, np.mean(model.predict(all_test[i]) > (0.5 * (i==0))))
    """

    # NEW MODEL
    class DataGenerator(Sequence):

        def __init__(self, raw_data, batch_size=32, maxlen=200):
            self.raw_data = raw_data
            self.epoch = 0
            self.maxlen = maxlen
            self.batch_size = batch_size
            self.x, self.y, _ = make_comparison_dataset(
                self.raw_data, self.maxlen, n=30*self.batch_size, binsize=args.binsize, maxbin=args.maxbin)

        def __len__(self):
            return int(np.ceil(len(self.y) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = [xx[idx * self.batch_size:(idx + 1) * self.batch_size] for xx in self.x]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

        def on_epoch_end(self):
            self.x, self.y, _ = make_comparison_dataset(
                self.raw_data, self.maxlen, n=30*self.batch_size, binsize=args.binsize, maxbin=args.maxbin)
            self.epoch += 1

    if not os.path.exists(args.model_filepath):

        num_classes = 224
        embed_dim = 64
        hidden_dim = 128

        sentence_a = layers.Input(shape=(None,))
        sentence_b = layers.Input(shape=(None,))

        shared_embedding = layers.Embedding(num_classes, embed_dim)
        shared_lstm = layers.LSTM(hidden_dim)

        a = shared_embedding(sentence_a)
        b = shared_embedding(sentence_b)
        a = shared_lstm(a)
        b = shared_lstm(b)

        merged_vector = layers.concatenate([a, b], axis=-1)
        predictions = layers.Dense(1, activation='sigmoid')(merged_vector)

        model = Model(inputs=[sentence_a, sentence_b], outputs=predictions)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)]

        valid_inputs,valid_outputs,_ = make_comparison_dataset(valid_d, args.maxlen, n=100, binsize=args.binsize, maxbin=args.maxbin)
        data_generator = DataGenerator(
            train_d, batch_size=args.batch_size, maxlen=args.maxlen)
        model.fit_generator(data_generator, epochs=args.num_epochs, validation_data=(valid_inputs, valid_outputs), callbacks=callbacks)

        # save the model for training
        model.save(args.model_filepath)

    else:
        print("LOADING PRETRAINED MODEL")
        model = load_model(args.model_filepath)

    results = {}

    # how high is the target ranked from a collection of n
    test_inputs = make_rank_dataset(
        test_d, args.maxlen, n_readings=10)

    def to_pairwise(x):
        pairwise = []
        for (i,j),v in x.items():
            for _ in range(v):
                pairwise.append([i,j])
            for _ in range(100 - v):
                pairwise.append([j,i])
        return pairwise

    results["RANK"] = []
    for input in tqdm(test_inputs):
        out = {}
        inputs_a = []
        inputs_b = []
        for i,j in itertools.combinations(range(len(input)), 2):
            inputs_a.append( input[i] )
            inputs_b.append( input[j] )
        probs = model.predict([np.array(inputs_a), np.array(inputs_b)])
        for k,(i,j) in enumerate(itertools.combinations(range(len(input)), 2)):
            out[repr((i,j))] = int(probs[k][0] * 1000000)
        #print(json.dumps(out, indent=4))
        results["RANK"].append(out)

    test_inputs, test_outputs, test_ids = make_comparison_dataset(
        test_d, args.maxlen, n=10000, n_readings=10, binsize=args.binsize, maxbin=args.maxbin)

    preds = model.predict(test_inputs).flatten()
    for target, pred, ids in zip(test_outputs, preds, test_ids):
        ids = repr(tuple(sorted(list(ids))))
        if not ids in results:
            results[ids] = []
        results[ids].append( int(np.round(pred) == target) )

    print(np.bincount(np.round(preds).astype(np.int32)))
    print(np.bincount(test_outputs))

    #print(json.dumps({k : np.mean(v) for k,v in sorted(results.items())}, indent=4, sort_keys=True))

    with open(args.result_filepath, 'w') as outfile:
        json.dump(results, outfile)




#
