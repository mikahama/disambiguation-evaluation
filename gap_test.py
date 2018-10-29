import json
import numpy as np
np.warnings.filterwarnings('ignore')
from test_sentences import run_spmf_full
from subprocess import call
import collections
from custom_types import *

import os, json, codecs, pickle
from uralicNLP.ud_tools import UD_collection

# adding a dummy value to each itemset and an additional itemset containing the dummy value allows for patterns with gaps to be found

def print_dict(d):
    print json.dumps(
        collections.OrderedDict([(str(k),str(v)) for k,v in d.items()]),
        indent=4)

def print_txt_file(fp):
    call(["cat", fp])

def encode(x):
    x = x.add_values(99)
    x.append([99])
    return x

def decode(x):
    x = IntListList(x).remove_values(99)
    return x.remove_empty_margin_gaps().to_tuple()

ud = UD_collection(codecs.open("ud/fi-ud-train.conllu", encoding="utf-8"))
X = [UD_sentence_to_list(sentence) for sentence in ud.sentences[:10]]

#X = [
#    IntListList([[4,5], [3], [6], [2]]),
#    IntListList([[4,5], [7], [6]])
#]
X = list(map(encode, X))

res = run_spmf_full(
    X, min_sup=0, algorithm="BIDE+", max_pattern_length=10, max_gap=1)

print_txt_file("tmp_spmf.txt")
print("\n")
print_txt_file("tmp_spmf_output.txt")
print_dict(res.score_dict.map_keys(decode).filter(
    lambda k,v : (v > 1) and (len(k) < 5) and (IntListList(k).max_gap() < 3)) )
#print_dict(res.sid_dict.map_keys(decode))
