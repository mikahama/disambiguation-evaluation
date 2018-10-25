import numpy as np
import collections
from fw_master_map import fw_map

map_begin = np.sum([len(v) for v in fw_map.values()])

def last_match(a,b,c,k):
    return (k in b) and (k in c) and (b[k]==c[k])

def last_last_match(a,b,c,k):
    return (k in a) and (k in c) and (a[k]==c[k])

def all_match(a,b,c,k):
    return (k in a) and (k in b) and (k in c) and (a[k]==b[k]) and (b[k]==c[k])

props_list = [
    lambda a,b,c : last_match(a,b,c,"Abbr"),
    lambda a,b,c : last_match(a,b,c,"AdpType"),
    lambda a,b,c : last_match(a,b,c,"AdvType"),
    lambda a,b,c : last_match(a,b,c,"Animacy"),
    lambda a,b,c : last_match(a,b,c,"Aspect"),
    lambda a,b,c : last_match(a,b,c,"Case"),
    lambda a,b,c : last_match(a,b,c,"Clitic"),
    lambda a,b,c : last_match(a,b,c,"Connegative"),
    lambda a,b,c : last_match(a,b,c,"Definite"),
    lambda a,b,c : last_match(a,b,c,"Degree"),
    lambda a,b,c : last_match(a,b,c,"Derivation"),
    lambda a,b,c : last_match(a,b,c,"Evident"),
    lambda a,b,c : last_match(a,b,c,"Foreign"),
    lambda a,b,c : last_match(a,b,c,"Gender"),
    lambda a,b,c : last_match(a,b,c,"InfForm"),
    lambda a,b,c : last_match(a,b,c,"Mood"),
    lambda a,b,c : last_match(a,b,c,"NameType"),
    lambda a,b,c : last_match(a,b,c,"NegationType"),
    lambda a,b,c : last_match(a,b,c,"NumType"),
    lambda a,b,c : last_match(a,b,c,"Number"),
    lambda a,b,c : last_match(a,b,c,"Number[obj]"),
    lambda a,b,c : last_match(a,b,c,"Number[psor]"),
    lambda a,b,c : last_match(a,b,c,"Number[subj]"),
    lambda a,b,c : last_match(a,b,c,"POS"),
    lambda a,b,c : last_match(a,b,c,"PartType"),
    lambda a,b,c : last_match(a,b,c,"Person"),
    lambda a,b,c : last_match(a,b,c,"Person[obj]"),
    lambda a,b,c : last_match(a,b,c,"Person[psor]"),
    lambda a,b,c : last_match(a,b,c,"Person[subj]"),
    lambda a,b,c : last_match(a,b,c,"Polarity"),
    lambda a,b,c : last_match(a,b,c,"PronType"),
    lambda a,b,c : last_match(a,b,c,"Reflex"),
    lambda a,b,c : last_match(a,b,c,"Style"),
    lambda a,b,c : last_match(a,b,c,"Tense"),
    lambda a,b,c : last_match(a,b,c,"Typo"),
    lambda a,b,c : last_match(a,b,c,"Valency"),
    lambda a,b,c : last_match(a,b,c,"Variant"),
    lambda a,b,c : last_match(a,b,c,"VerbForm"),
    lambda a,b,c : last_match(a,b,c,"VerbType"),
    lambda a,b,c : last_match(a,b,c,"Voice"),

    lambda a,b,c : last_last_match(a,b,c,"Abbr"),
    lambda a,b,c : last_last_match(a,b,c,"AdpType"),
    lambda a,b,c : last_last_match(a,b,c,"AdvType"),
    lambda a,b,c : last_last_match(a,b,c,"Animacy"),
    lambda a,b,c : last_last_match(a,b,c,"Aspect"),
    lambda a,b,c : last_last_match(a,b,c,"Case"),
    lambda a,b,c : last_last_match(a,b,c,"Clitic"),
    lambda a,b,c : last_last_match(a,b,c,"Connegative"),
    lambda a,b,c : last_last_match(a,b,c,"Definite"),
    lambda a,b,c : last_last_match(a,b,c,"Degree"),
    lambda a,b,c : last_last_match(a,b,c,"Derivation"),
    lambda a,b,c : last_last_match(a,b,c,"Evident"),
    lambda a,b,c : last_last_match(a,b,c,"Foreign"),
    lambda a,b,c : last_last_match(a,b,c,"Gender"),
    lambda a,b,c : last_last_match(a,b,c,"InfForm"),
    lambda a,b,c : last_last_match(a,b,c,"Mood"),
    lambda a,b,c : last_last_match(a,b,c,"NameType"),
    lambda a,b,c : last_last_match(a,b,c,"NegationType"),
    lambda a,b,c : last_last_match(a,b,c,"NumType"),
    lambda a,b,c : last_last_match(a,b,c,"Number"),
    lambda a,b,c : last_last_match(a,b,c,"Number[obj]"),
    lambda a,b,c : last_last_match(a,b,c,"Number[psor]"),
    lambda a,b,c : last_last_match(a,b,c,"Number[subj]"),
    lambda a,b,c : last_last_match(a,b,c,"POS"),
    lambda a,b,c : last_last_match(a,b,c,"PartType"),
    lambda a,b,c : last_last_match(a,b,c,"Person"),
    lambda a,b,c : last_last_match(a,b,c,"Person[obj]"),
    lambda a,b,c : last_last_match(a,b,c,"Person[psor]"),
    lambda a,b,c : last_last_match(a,b,c,"Person[subj]"),
    lambda a,b,c : last_last_match(a,b,c,"Polarity"),
    lambda a,b,c : last_last_match(a,b,c,"PronType"),
    lambda a,b,c : last_last_match(a,b,c,"Reflex"),
    lambda a,b,c : last_last_match(a,b,c,"Style"),
    lambda a,b,c : last_last_match(a,b,c,"Tense"),
    lambda a,b,c : last_last_match(a,b,c,"Typo"),
    lambda a,b,c : last_last_match(a,b,c,"Valency"),
    lambda a,b,c : last_last_match(a,b,c,"Variant"),
    lambda a,b,c : last_last_match(a,b,c,"VerbForm"),
    lambda a,b,c : last_last_match(a,b,c,"VerbType"),
    lambda a,b,c : last_last_match(a,b,c,"Voice"),

    lambda a,b,c : all_match(a,b,c,"Abbr"),
    lambda a,b,c : all_match(a,b,c,"AdpType"),
    lambda a,b,c : all_match(a,b,c,"AdvType"),
    lambda a,b,c : all_match(a,b,c,"Animacy"),
    lambda a,b,c : all_match(a,b,c,"Aspect"),
    lambda a,b,c : all_match(a,b,c,"Case"),
    lambda a,b,c : all_match(a,b,c,"Clitic"),
    lambda a,b,c : all_match(a,b,c,"Connegative"),
    lambda a,b,c : all_match(a,b,c,"Definite"),
    lambda a,b,c : all_match(a,b,c,"Degree"),
    lambda a,b,c : all_match(a,b,c,"Derivation"),
    lambda a,b,c : all_match(a,b,c,"Evident"),
    lambda a,b,c : all_match(a,b,c,"Foreign"),
    lambda a,b,c : all_match(a,b,c,"Gender"),
    lambda a,b,c : all_match(a,b,c,"InfForm"),
    lambda a,b,c : all_match(a,b,c,"Mood"),
    lambda a,b,c : all_match(a,b,c,"NameType"),
    lambda a,b,c : all_match(a,b,c,"NegationType"),
    lambda a,b,c : all_match(a,b,c,"NumType"),
    lambda a,b,c : all_match(a,b,c,"Number"),
    lambda a,b,c : all_match(a,b,c,"Number[obj]"),
    lambda a,b,c : all_match(a,b,c,"Number[psor]"),
    lambda a,b,c : all_match(a,b,c,"Number[subj]"),
    lambda a,b,c : all_match(a,b,c,"POS"),
    lambda a,b,c : all_match(a,b,c,"PartType"),
    lambda a,b,c : all_match(a,b,c,"Person"),
    lambda a,b,c : all_match(a,b,c,"Person[obj]"),
    lambda a,b,c : all_match(a,b,c,"Person[psor]"),
    lambda a,b,c : all_match(a,b,c,"Person[subj]"),
    lambda a,b,c : all_match(a,b,c,"Polarity"),
    lambda a,b,c : all_match(a,b,c,"PronType"),
    lambda a,b,c : all_match(a,b,c,"Reflex"),
    lambda a,b,c : all_match(a,b,c,"Style"),
    lambda a,b,c : all_match(a,b,c,"Tense"),
    lambda a,b,c : all_match(a,b,c,"Typo"),
    lambda a,b,c : all_match(a,b,c,"Valency"),
    lambda a,b,c : all_match(a,b,c,"Variant"),
    lambda a,b,c : all_match(a,b,c,"VerbForm"),
    lambda a,b,c : all_match(a,b,c,"VerbType"),
    lambda a,b,c : all_match(a,b,c,"Voice"),

]


props_list = [
    lambda a,b,c : last_match(a,b,c,"Case"),
    lambda a,b,c : last_last_match(a,b,c,"Case"),
    lambda a,b,c : all_match(a,b,c,"Case"),
    lambda a,b,c : last_match(a,b,c,"Number"),
    lambda a,b,c : last_last_match(a,b,c,"Number"),
    lambda a,b,c : all_match(a,b,c,"Number"),
]

props_list = []

print "THERE ARE {} PROPOSITIONAL VARIABLES".format(len(props_list))

props = collections.OrderedDict([(p,i+map_begin) for i,p in enumerate(props_list)])

if __name__ == "__main__":

    import os, json, codecs
    from uralicNLP.ud_tools import UD_collection
    from common import parse_feature_to_dict, full_window

    ud = UD_collection(codecs.open("ud/fi-ud-test.conllu", encoding="utf-8"))
    for sentence in ud.sentences:
        X = [parse_feature_to_dict(node.feats) for node in sentence.find()]

        for args in full_window(X,3):
            print [v for k,v in props.items() if  k(*args)]
