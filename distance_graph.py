# create the graph of distances from the target
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

import itertools
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

if __name__ == "__main__":

    langs = ["kpv", "myv", "sme", "fin"]

    maxdim = 1000
    with PdfPages('distance_graph.pdf') as pdf:
        plt.figure(figsize=(20,4))
        for ii,lang in enumerate(langs):
            d = create_dataset(lang, force=0)
            dists = []
            for reading in tqdm(d["readings"]):
                tmp = [r.values() for r in reading]
                dim = np.prod([len(r) for r in tmp])
                if (dim > maxdim):
                    for _ in range(maxdim):
                        dists.append( np.sum([np.random.choice(r) for r in tmp]) )
                else:
                    for comb in itertools.product(*[r.values() for r in reading]):
                        dists.append( np.sum(comb) )

            plt.subplot(1,4,ii+1)
            plt.hist(dists, bins=np.arange(np.max(dists)), histtype='step', cumulative=True)
            plt.title(lang)
        pdf.savefig()
        #plt.show()
