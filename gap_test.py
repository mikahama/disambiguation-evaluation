import json
import numpy as np
np.warnings.filterwarnings('ignore')
from test_sentences import run_spmf_full
from subprocess import call
import collections

def print_dict(d):
    print json.dumps(
        collections.OrderedDict([(str(k),str(v)) for k,v in d.items()]),
        indent=4)

def print_txt_file(fp):
    call(["cat", fp])

X = [[[4,5], [3], [6], [4,5], [7], [6]]]
print_dict( run_spmf_full(
    X, min_sup=0, algorithm="VMSP", max_pattern_length=3, max_gap=1) )

print_txt_file("tmp_spmf_output.txt")

print_dict( run_spmf_full(
    X, min_sup=0, algorithm="VMSP", max_pattern_length=3, max_gap=2) )

print_txt_file("tmp_spmf_output.txt")
