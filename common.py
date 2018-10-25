import os, json, codecs, pickle
import numpy as np
from uralicNLP.ud_tools import UD_collection
from maps import ud_pos
from proposition_map import props
from fw_master_map import fw_map
import operator
import argparse

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
	if "POS" in d:
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

arg_parser = argparse.ArgumentParser(description='Run tests')
def make_arg_parser():
	arg_parser.add_argument('train_language',help='Language code for training')
	arg_parser.add_argument('test_language',help='Language code for testing')
	arg_parser.add_argument('--spmf-algorithm',help='SPMF algorithm to be used', default="VMSP")
	arg_parser.add_argument('--min-sup',help='SPMF minimum support', type=int, default=10)
	arg_parser.add_argument('--max-pattern-length',help='SPMF max pattern pattern length', type=int, default=20)
	arg_parser.add_argument('--max-gap',help='SPMF max gap', type=int, default=1)
	arg_parser.add_argument('--max-window',help='Max window', type=int, default=3)
	arg_parser.add_argument('--save-results',help='Saves results to a file', type=bool, default=False)


# return a full window of w items
def full_window(x,w,blank={}):
	x = list(x)
	return zip(*[[{}]*i + (x[:-i] if i > 0 else x) for i in range(w-1,-1,-1)])

def UD_sentence_to_list(sentence,w=3):
	tmp = sentence.find()
	tmp.sort()
	ds = [parse_node_to_dict(node) for node in tmp]
	return [apply_forward_map_to_dict(args[-1],fw_map) + [v for k,v in props.items() if k(*args)] for args in full_window(ds,w)]

def __make_filename_from_args(args):
	d = {}
	for arg in vars(args):
		d[arg]=  getattr(args, arg)
	keys = list(d.keys())
	keys.sort()
	name_parts = []
	for key in keys:
		name_parts.append(str(d[key]))
	return "_".join(name_parts)


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
	import sys

	make_arg_parser()

	languages = {"fin":{"test":"ud/fi-ud-test.conllu", "train":"ud/fi-ud-train.conllu"}, "kpv":{"test":"ud/kpv_lattice-ud-test.conllu", "train":"ud/kpv_lattice-ud-test.conllu"}, "sme":{"test":"ud/sme_giella-ud-test.conllu", "train":"ud/sme_giella-ud-train.conllu"}, "myv":{"test":"ud/myv-ud.conllu", "train":"ud/myv-ud.conllu"}}
	np.random.seed(1234)

	train_lang = "fin"
	test_lang = "sme"
	MAX_WINDOW = 3

	arg = sys.argv
	args = arg_parser.parse_args()
	save_plot = False
	filename = __make_filename_from_args(args)
	if len(arg)>1:
		MAX_WINDOW = args.max_window
		test_lang = args.test_language
		train_lang = args.train_language
		if args.save_results:
			sys.stdout = codecs.open("results/" + filename + ".log", "w", encoding="utf-8")
			save_plot = True
	else:
		print "No arguments, using variables"

	

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

	ud = UD_collection(codecs.open(languages[train_lang]["train"], encoding="utf-8"))
	X = [UD_sentence_to_list(sentence,w=MAX_WINDOW) for sentence in ud.sentences]
	result_dict, sid_dict = run_spmf_full(X, min_sup=args.min_sup, algorithm=args.spmf_algorithm, max_pattern_length=args.max_pattern_length, max_gap=args.max_gap, save_results_to="results/tmp/" + filename +"_spmf_output.txt", temp_file="results/tmp/" + filename +"_tmp_spmf.txt")
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

	test_ud = UD_collection(codecs.open(languages[test_lang]["test"], encoding="utf-8"))
	SCORE_FUNC = score_sentence
	test_results = []
	for sentence in test_ud.sentences:
		all_readings = __give_all_possibilities(sentence, lang=test_lang)
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

	if save_plot:
		pickle.dump(test_results, open('results/' + filename + ".pickle", "wb"))
		plt.savefig('results/' + filename + ".png")
	else:
		plt.show()





#
