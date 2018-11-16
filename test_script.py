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

def get_random_labels(n, split=0.9, seed=689012349):
    np.random.seed(seed)
    n_train_sentences = int(n * split)
    n_test_sentences = int(n - n_train_sentences)
    assert (n_train_sentences > 0) and (n_test_sentences > 0)
    labels = np.asarray(([0] * n_train_sentences) + ([1] * n_test_sentences))
    np.random.shuffle(labels)
    return np.where(labels==0)[0], np.where(labels==1)[0]

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

if __name__ == "__main__":

    from scoring import ScoreSentenceByLinearRegression
    from test_sentences import run_spmf_full
    import argparse

    arg_parser = argparse.ArgumentParser(description='Run tests')
    arg_parser.add_argument('--lang', type=str, default="sme")
    arg_parser.add_argument('--pad', type=bool, default=True)
    arg_parser.add_argument('--min-sup', type=int, default=20)
    arg_parser.add_argument('--seed', type=int, default=1234)
    arg_parser.add_argument('--k', type=int, default=276)
    args = arg_parser.parse_args()
    pad_value = 900 if args.pad else None

    np.random.seed(args.seed)

    d = create_dataset(args.lang)

    pattern_filepath = "PATTERNS_PAD={}_MIN_SUP={}.npz".format(
        args.pad, args.min_sup)

    if os.path.exists(pattern_filepath):
        _, sid_dict = np.load(pattern_filepath)["data"]
    else:
        res = run_spmf_full(d["target"], min_sup=args.min_sup, save_results_to="results/tmp/trash_spmf_output.txt", temp_file="results/tmp/trash_tmp_spmf.txt", pad_value=pad_value)
        np.savez(pattern_filepath, data=(res.score_dict, res.sid_dict))
        sid_dict = res.sid_dict

    train_idx, test_idx = get_random_labels(
        len(d["target"]), 0.9, seed=args.seed)

    # prune the sid_dict to only include patterns that occur in train_idx
    sid_dict.prune(train_idx, top_k_support=args.k, min_pattern_items=2, min_pattern_length=2)

    X = []
    Y = []
    for idx in tqdm(train_idx):
        X += [sid_dict.pattern_vector(d["target"][idx])]
        Y += [True]
        wrong = random_reading(d["readings"][idx],2,100)
        if wrong is not None:
            X += [sid_dict.pattern_vector(wrong)]
            Y += [False]

    s = ScoreSentenceByLinearRegression(sid_dict, X, Y)

    # comparison test
    n_results = []
    p_results = []
    for min_diff in tqdm(np.arange(1,20)):
        n_res = []
        for idx in tqdm(test_idx, leave=False):
            wrong = random_reading(d["readings"][idx],min_diff,100)
            if wrong is not None:
                n_res += [(s.score(wrong) < 0.5)]
            if min_diff == 1:
                p_results += [(s.score(d["target"][idx]) > 0.5)]

        n_results += [np.mean(n_res)]
        p_results = np.mean(p_results)

    n_results = np.asarray(n_results)

    print p_results, n_results

    # rank test
    results = []
    diffs = [5, 10, 15, 20]
    for idx in tqdm(test_idx):
        rs = [d["target"][idx]]
        for min_diff, max_diff in zip(diffs, diffs[1:]):
            rs += [random_reading(d["readings"][idx],min_diff,max_diff)]
        if all([r is not None for r in rs]):
            ranks = np.argsort([s.score(r) for r in rs])
            results += [kendall_tau_dist(ranks, np.arange(len(diffs))[::-1])]

    print np.mean(results)




#
