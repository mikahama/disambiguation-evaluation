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

X = [
    [[4,5], [3], [6], [2]],
    [[4,5], [6], [3]]
]
score_dict, sid_dict = run_spmf_full(
    X, min_sup=0, algorithm="VMSP", max_pattern_length=3, max_gap=1)

print X
print_txt_file("tmp_spmf.txt")
print("\n")
print_txt_file("tmp_spmf_output.txt")
print_dict(score_dict)
print_dict(sid_dict)

score_dict, sid_dict = run_spmf_full(
    X, min_sup=0, algorithm="VMSP", max_pattern_length=3, max_gap=2)

print_txt_file("tmp_spmf.txt")
print("\n")
print_txt_file("tmp_spmf_output.txt")
print_dict(score_dict)
print_dict(sid_dict)


def contains_pattern(X,pattern,verbose=False):
    # does x contain pattern y
    # contains_pattern([[3, 5], [3], [4, 6, 7]], [[3, 5], [], [6]]) == True
    # contains_pattern([[3, 5], [3], [4, 6, 7]], [[4,5], [], [], []]) == False
    assert all([isinstance(x,list) for x in X])
    assert all([isinstance(p,list) for p in pattern])
    for i in range(len(X)-len(pattern)+1):
        j = 0
        while j < len(pattern) and i+j < len(X) and set(pattern[j]).issubset(set(X[i+j])):
            j += 1
        if j == len(pattern):
            if verbose:
                print "{} contains {}".format(X,pattern)
            return True
    if verbose:
        print "{} does NOT contain {}".format(X,pattern)
    return False

def possible_patterns(X,max_gap=1):
    # create all possible patterns with gaps from X (without gaps)
    # ex.
    import operator
    import itertools
    from combinatorics import unlabeled_balls_in_labeled_boxes
    max_gap -= 1
    if max_gap == 0:
        return [X]
    gapped_patterns = [((0,)*(len(X)-1), X)]
    box_sizes = [max_gap]*(len(X)-1)
    ball_range = range(1,max_gap*(len(X)-1)+1)
    for counts in itertools.chain.from_iterable(
    unlabeled_balls_in_labeled_boxes(b,box_sizes) for b in ball_range):
        gapped_patterns += [(counts, reduce(operator.add, [[X[i]] + [[]] * s for i,s in enumerate(counts)]) + [X[-1]])]
    return collections.OrderedDict( gapped_patterns )

def find_gap(data,ungapped_pattern,idx,max_gap=1):
    # idx is the location of the patterns in the data
    # note that this will return all gapped patterns that can be found
    gapped_patterns = possible_patterns(ungapped_pattern,max_gap=max_gap)
    actual_patterns = []
    for pattern in gapped_patterns:
        if all([contains_pattern(data[i],pattern) for i in idx]):
            actual_patterns += [pattern]
    return actual_patterns

# find the distributions for these patterns
def calculate_gap_distribution(data,ungapped_pattern,idx,max_gap=1):
    gapped_patterns = possible_patterns(ungapped_pattern,max_gap=max_gap)
    gap_distribution = []
    for counts, pattern in gapped_patterns.items():
        frequency = 0
        for i in idx:
            frequency += int(contains_pattern(data[i], pattern))
        gap_distribution += [(counts, frequency)]
    return collections.OrderedDict( gap_distribution )

def calculate_gap_distributions(data,sid_dict,max_gap=1):
    gap_distributions = []
    for k,v in sid_dict.items():
        kl = [[__ for __ in _] for _ in k]
        gap_distributions += [
            (k, calculate_gap_distribution(data,kl,v,max_gap=max_gap))]
    return collections.OrderedDict( gap_distributions )

assert contains_pattern([[3, 5], [3], [4, 6, 7]], [[3, 5], [], [6]])
assert not contains_pattern([[3, 5], [3], [4, 6, 7]], [[4,5], [], [], [], []])

from uralicNLP.ud_tools import UD_collection
from common import UD_sentence_to_list
import os, json, codecs

UD_PATH = "ud/fi-ud-test.conllu"
ud = UD_collection(codecs.open(UD_PATH, encoding="utf-8"))
X = [UD_sentence_to_list(sentence,w=3) for sentence in ud.sentences]

_, sid_dict = run_spmf_full(X, min_sup=50, algorithm="VMSP", max_pattern_length=20, max_gap=3)

for k,v in calculate_gap_distributions(X,sid_dict,max_gap=3).items():
    print k, v

#
