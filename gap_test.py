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
    print(json.dumps(
        collections.OrderedDict([(str(k),str(v)) for k,v in d.items()]),
        indent=4))

def print_txt_file(fp):
    call(["cat", fp])

X = [
    IntListList([[4,5], [3], [6], [2]]),
    IntListList([[4,5], [7], [6]]),
    IntListList([[4,5], [5], [6]])
]

#ud = UD_collection(codecs.open("ud/fi-ud-train.conllu", encoding="utf-8"))
#X = [UD_sentence_to_list(sentence) for sentence in ud.sentences[:10]]

for algorithm in ["BIDE+", "VMSP"]:
    for pad_value in [None, 2**16]:

        print(("ALGORITHM : ", algorithm, "PAD_VALUE : ", pad_value))

        res = run_spmf_full(
            X,
            min_sup=0,
            algorithm=algorithm,
            min_pattern_length=2,
            max_pattern_length=5,
            max_gap=1,
            min_pattern_span=2,
            max_pattern_span=5,
            max_span_gap=0,
            pad_value=pad_value)

        print("INPUT : ")
        print_txt_file("tmp_spmf.txt")
        print("\n")

        print("OUTPUT : ")
        print_txt_file("tmp_spmf_output.txt")
        print("\n")

        print("SCORE DICT : ")
        print_dict(res.score_dict)
        print(len(res.score_dict))
        print("\n")
