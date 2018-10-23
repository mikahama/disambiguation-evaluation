import os, json, codecs
import numpy as np
from uralicNLP.ud_tools import UD_collection

np.warnings.filterwarnings('ignore')

def parse_feature_to_dict(s):
	if s == "_":
		return {}
	return {_.split("=")[0] : _.split("=")[1] for _ in s.split("|")}

_partial_keys = ["Case", "Connegative", "VerbForm", "Mood", "Number", "Person", "Tense", "Voice"]

def cache_wrapper(func):
	def call(*args, **kwargs):
		cache = kwargs.get("cache", "")
		overwrite = kwargs.get("overwrite", False)
		if not overwrite and os.path.exists(cache):
			return np.load(cache)["data"]
		else:
			out = func(*args, **kwargs)
			np.savez(cache, data=out)
			return out
	return call

def dict_to_json(filepath, d):
	with open(filepath, 'w') as fp:
		json.dump(d, fp, indent=4, sort_keys=True)

# will have to write a different parser for different conllu files probably
def parse_feature(s):
	if s == "_":
		return []
	return [tuple(_.split("=")) for _ in s.split("|")]

@cache_wrapper
def UD_tree_to_mapping(input_filepath, **kwargs):

	_map = {}
	ud = UD_collection(codecs.open(input_filepath, encoding="utf-8"))
	for sentence in ud.sentences:
		for node in sentence.find(): # for each node get all children
			for type, value in parse_feature(node.feats):
				if not type in _map:
					_map[type] = [value]
				else:
					_map[type] += [value]

	_map = {k:np.unique(v) for k,v in _map.items()}
	fw_map = {k:{x:i for i,x in enumerate(v)} for k,v in _map.items()}
	bw_map = {k:{i:x for i,x in enumerate(v)} for k,v in _map.items()}
	return (fw_map, bw_map)

