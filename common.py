import os, json, codecs
import numpy as np
from uralicNLP.ud_tools import UD_collection
from maps import ud_pos

np.warnings.filterwarnings('ignore')

def parse_feature_to_dict(s):
	if s == "_":
		return {}
	return {_.split("=")[0] : _.split("=")[1] for _ in s.split("|")}

def parse_feature(s):
	if s == "_":
		return []
	return [tuple(_.split("=")) for _ in s.split("|")]

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

def dict_to_py(filepath, d, name):
	with open(filepath, 'w') as f:
		f.write("import collections\n")
		f.write("{} = collections.OrderedDict(".format(name))
		json.dump(d, f, indent=4, sort_keys=True)
		f.write(")")

@cache_wrapper
def UD_trees_to_mapping(input_filepaths, **kwargs):

	#NOTE : now this will also include pos tags

	if not isinstance(input_filepaths, list):
		input_filepaths = [input_filepaths]

	_map = {}
	for input_filepath in input_filepaths:
		print(input_filepath)
		ud = UD_collection(codecs.open(input_filepath, encoding="utf-8"))
		for sentence in ud.sentences:
			for node in sentence.find(): # for each node get all children
				for type, value in parse_feature(node.feats):
					if not type in _map:
						_map[type] = [value]
					else:
						_map[type] += [value]
				if not "POS" in _map:
					_map["POS"] = [ud_pos[node.xpostag]]
				else:
					_map["POS"] += [ud_pos[node.xpostag]]

	_map = {k:np.sort(np.unique(v)) for k,v in _map.items()}
	fw_map = {}
	bw_map = {}
	fw_count = 0
	bw_count = 0
	for k in np.sort(_map.keys()):
		fw_map[k] = {x:(i,fw_count+i) for i,x in enumerate(_map[k])}
		bw_map[k] = {x:(i,bw_count+i) for i,x in enumerate(_map[k])}
		fw_count += len(_map[k])
		bw_count += len(_map[k])

	return (fw_map, bw_map)

def UD_sentence_to_list(sentence):
	tmp = sentence.find()
	tmp.sort()
	return parse_features_to_dict(node) for node in tmp
