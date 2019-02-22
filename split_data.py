# script to create the datasets
import numpy as np
np.warnings.filterwarnings('ignore')
import codecs
import os
from tqdm import tqdm
from collections import OrderedDict

from lang import languages
from test_sentences import change_ud_morphology, give_all_possibilities
from custom_types import IntListList, DictList, Dataset
from uralicNLP.ud_tools import UD_collection

def get_sentences(filepaths):
    filepaths = np.unique(np.atleast_1d(filepaths))
    sent = []
    for filepath in filepaths:
        sent += UD_collection(codecs.open(filepath, encoding="utf-8")).sentences
    return sent

def create_dataset(lang, force=False):
    sentences = get_sentences(languages[lang].values())
    filepath = "DATASET_{}_{}.npz".format("MASTER", lang)
    if os.path.exists(filepath) and not force:
        try:
            return Dataset(filepath=filepath)
        except:
            pass

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

    d.save(filepath)
    return d

def get_random_labels(n, split):
    idx = np.argsort(np.random.rand(n))
    split = np.cumsum((np.atleast_1d(split) * n).astype(np.int32))
    if len(split) == 3:
        return idx[:split[0]], idx[split[0]:split[1]], idx[split[1]:]
    elif len(split) == 2:
        return idx[:split[0]], idx[split[0]:]
    else:
        raise ValueError("SPLIT NOT SUPPORTED {}".format(split))

if __name__ == "__main__":

    import argparse
    arg_parser = argparse.ArgumentParser(description='Split Data')
    arg_parser.add_argument('--seed', type=int, required=True)
    args = arg_parser.parse_args()
    np.random.seed(args.seed)

    langs = ["kpv", "myv", "sme", "est", "fin"]

    for lang in langs:
        d = create_dataset(lang, force=0)
        ridx, vidx, tidx = get_random_labels(len(d["target"]), (.8,.1,.1))
        d.get_subset(ridx).save("DATASET_SPLIT_TRAIN_{}_{}.npz".format(lang,args.seed))
        d.get_subset(tidx).save("DATASET_SPLIT_TEST_{}_{}.npz".format(lang,args.seed))
        d.get_subset(vidx).save("DATASET_SPLIT_VALID_{}_{}.npz".format(lang,args.seed))


#
