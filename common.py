import os, json, codecs
import numpy as np
from uralicNLP.ud_tools import UD_collection
from maps import ud_pos
from proposition_map import props
from fw_master_map import fw_map
import operator

# TODO : try transfer learning
# TODO : try just learning on props
# TODO : easy way to filter list of list

np.warnings.filterwarnings('ignore')

def parse_feature_to_dict(s):
	if s == "_":
		return {}
	return {_.split("=")[0] : _.split("=")[1] for _ in s.split("|")}

def parse_feature(s):
	if s == "_":
		return []
	return [tuple(_.split("=")) for _ in s.split("|")]

def parse_node_to_dict(node):
	d = parse_feature_to_dict(node.feats)
	d["POS"] = ud_pos[node.xpostag]
	return d

# tuple of tuple to list of list
def tt_ll(tt):
	return [[_ for _ in t] for t in tt]

# fill the gaps for sending into smpf again
def fill_gaps(ll,x):
	nl = []
	for l in ll:
		nl += [l]
		nl += [[x]]
	return reduce(operator.add, nl)

def apply_forward_map_to_dict(d,fw_map,index=1):
	#print d
	d["POS"] = ud_pos[d["POS"]]
	return [fw_map[k][v][index] for k,v in d.items()]

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

# return a full window of w items
def full_window(x,w,blank={}):
	x = list(x)
	return zip(*[[{}]*i + (x[:-i] if i > 0 else x) for i in range(w-1,-1,-1)])

def UD_sentence_to_list(sentence,w=3):
	tmp = sentence.find()
	tmp.sort()
	ds = [parse_node_to_dict(node) for node in tmp]
	return [apply_forward_map_to_dict(args[-1],fw_map) + [v for k,v in props.items() if k(*args)] for args in full_window(ds,w)]


if __name__ == "__main__":

	import os, json, codecs
	from uralicNLP.ud_tools import UD_collection
	from common import parse_feature_to_dict
	from test_sentences import spmf_format_to_file, read_spmf_output, run_spmf_full
	import operator
	from tqdm import tqdm
	from test_sentences import get_readings, __change_ud_morphology, __give_all_possibilities
	import itertools
	from backward_map import bw_map

	np.random.seed(1234)

	UD_PATH = "ud/fi-ud-test.conllu"
	#UD_PATH = "ud/sme_giella-ud-test.conllu"
	#LANG = "sme"
	LANG = "fin"
	MAX_WINDOW = 3

	# VMSP performs pretty well min_sup=5, max_pattern_length=20, max_gap=1
	# 120 props (min_sup=20) = 37.78
	# 6 props (min_sup=20) = 35.75
	# 2 props (min_sup=20) = 34.44
	# 0 props (min_sup=20) = 33.89
	# 0 props (min_sup=20,count_data) = 32.54
	# 0 props (min_sup=10) = 26.79
	# 0 props (min_sup=10,count_data) = 25.71
	# 0 props (min_sup=5) ~= 23
	# 0 props (min_sup=2) = 22.71 (22.56 with 20 instead of 10 choices)

	ud = UD_collection(codecs.open(UD_PATH, encoding="utf-8"))
	X = [UD_sentence_to_list(sentence,w=MAX_WINDOW) for sentence in ud.sentences]
	result_dict, sid_dict = run_spmf_full(X, min_sup=10, algorithm="VMSP", max_pattern_length=20, max_gap=1)
	#results = read_spmf_output("tmp_spmf_output.txt")
	results = result_dict.keys()
	print len(results)

	# how many variables does the longest pattern contain
	nested_len = lambda x : np.sum([len(_) for _ in x])
	pattern_lengths = np.asarray([nested_len(_) for _ in results])
	print "MAX PATTERN LENGTH : {}".format(np.max(pattern_lengths))
	print "MEAN PATTERN LENGTH : {}".format(np.mean(pattern_lengths))

	# how many patterns include a propositional variable?

	# summarize the patterns using bw_map


	def score_sentence(results, Y):
		match_count = 0
		for patt in results:
			for i in range(len(Y) - len(patt)):
				match = True
				for j in range(len(patt)):
					if not set(patt[j]).issubset(set(Y[i+j])):
						match = False
						break
				if match:
					match_count += result_dict[patt] # for counts
					#match_count += 1 # for bool

		return match_count

	SCORE_FUNC = score_sentence
	test_results = []
	for sentence in ud.sentences:
		all_readings = __give_all_possibilities(sentence, lang=LANG)
		all_encoded = [[apply_forward_map_to_dict(_,fw_map) for _ in word] for word in all_readings]

		target = UD_sentence_to_list(sentence,w=MAX_WINDOW)

		N = np.product([len(_) for _ in all_encoded])
		#print [len(_) for _ in all_encoded]
		if N < 1000 and N > 0:
			all_poss = list(itertools.product(*all_encoded))
			all_poss_read = list(itertools.product(*all_readings))
			subset = np.random.choice(
				np.arange(len(all_poss)),
				size=(min(10,len(all_poss)),),
				replace=False)
			subset_poss = [all_poss[i] for i in subset]
			subset_poss_read = [all_poss_read[i] for i in subset]
			wrong_scores = []
			for reading, ds in zip(subset_poss, subset_poss_read):
				prop_vars = [[v for k,v in props.items() if k(*args)] for args in full_window(ds,MAX_WINDOW,blank={})]
				reading = [a + b for a,b in zip(reading, prop_vars)]
				if not reading == target:
					wrong_scores += [SCORE_FUNC(results, reading)]

			target_score = SCORE_FUNC(results, target)

			if len(wrong_scores) > 0:
				test_results += [np.mean(np.asarray(wrong_scores) >= target_score)]

				print np.mean(test_results)

	from matplotlib import pyplot as plt
	from scipy.stats import gaussian_kde
	x = np.asarray(test_results)
	x_grid = np.linspace(0.,1.,1000)
	bandwidth=0.01
	kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
	y = kde.evaluate(x_grid)
	y /= np.sum(y)
	y = np.cumsum(y)

	plt.plot(x_grid,y,color='tab:blue',alpha=0.75)
	plt.xticks(np.linspace(0.,1.,11))
	plt.show()





#
