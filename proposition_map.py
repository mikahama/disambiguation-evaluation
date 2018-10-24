import numpy as np
import collections
from fw_master_map import fw_map
from bw_master_map import bw_map

map_begin = np.sum([len(v) for v in fw_map.values()])

def keys_match(x,y,k):
    return (k in x) and (k in y) and (x == y)

props = [
    lambda x,y : keys_match(x,y,"Abbr"),
    lambda x,y : keys_match(x,y,"AdpType"),
    lambda x,y : keys_match(x,y,"AdvType"),
    lambda x,y : keys_match(x,y,"Animacy"),
    lambda x,y : keys_match(x,y,"Aspect"),
    lambda x,y : keys_match(x,y,"Case"),
    lambda x,y : keys_match(x,y,"Clitic"),
    lambda x,y : keys_match(x,y,"Connegative"),
    lambda x,y : keys_match(x,y,"Definite"),
    lambda x,y : keys_match(x,y,"Degree"),
    lambda x,y : keys_match(x,y,"Derivation"),
    lambda x,y : keys_match(x,y,"Evident"),
    lambda x,y : keys_match(x,y,"Foreign"),
    lambda x,y : keys_match(x,y,"Gender"),
    lambda x,y : keys_match(x,y,"InfForm"),
    lambda x,y : keys_match(x,y,"Mood"),
    lambda x,y : keys_match(x,y,"NameType"),
    lambda x,y : keys_match(x,y,"NegationType"),
    lambda x,y : keys_match(x,y,"NumType"),
    lambda x,y : keys_match(x,y,"Number"),
    lambda x,y : keys_match(x,y,"Number[obj]"),
    lambda x,y : keys_match(x,y,"Number[psor]"),
    lambda x,y : keys_match(x,y,"Number[subj]"),
    lambda x,y : keys_match(x,y,"POS"),
    lambda x,y : keys_match(x,y,"PartType"),
    lambda x,y : keys_match(x,y,"Person"),
    lambda x,y : keys_match(x,y,"Person[obj]"),
    lambda x,y : keys_match(x,y,"Person[psor]"),
    lambda x,y : keys_match(x,y,"Person[subj]"),
    lambda x,y : keys_match(x,y,"Polarity"),
    lambda x,y : keys_match(x,y,"PronType"),
    lambda x,y : keys_match(x,y,"Reflex"),
    lambda x,y : keys_match(x,y,"Style"),
    lambda x,y : keys_match(x,y,"Tense"),
    lambda x,y : keys_match(x,y,"Typo"),
    lambda x,y : keys_match(x,y,"Valency"),
    lambda x,y : keys_match(x,y,"Variant"),
    lambda x,y : keys_match(x,y,"VerbForm"),
    lambda x,y : keys_match(x,y,"VerbType"),
    lambda x,y : keys_match(x,y,"Voice")
]

props = collections.OrderedDict([(p,i+map_begin) for i,p in enumerate(props)])

if __name__ == "__main__":

    import os, json, codecs
    from uralicNLP.ud_tools import UD_collection
    from common import parse_feature_to_dict

    ud = UD_collection(codecs.open("ud/fi-ud-test.conllu", encoding="utf-8"))
    for sentence in ud.sentences:
        X = [parse_feature_to_dict(node.feats) for node in sentence.find()]

        for a,b in zip(X,X[1:]):
            print [v for k,v in props.items() if  k(a,b)]
