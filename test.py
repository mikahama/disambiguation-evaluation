import codecs
from uralicNLP.ud_tools import UD_collection
import os
import numpy as np
np.warnings.filterwarnings('ignore')
import json
import itertools
from tqdm import tqdm
import operator
from test_sentences import get_readings, __change_ud_morphology, __give_all_possibilities
from common import _partial_keys
from common import *

def increment_dict(d,k):
	d[k] = d.get(k,0) + 1

def encode(d, pos):
	assert type(pos) == unicode
	return (pos,) + tuple([fw_map[k].get(d[k],-1) if k in d else -1 for k in _keys])

def partial_encode(d, pos):
	assert type(pos) == unicode
	return (pos,) + tuple([fw_map[k].get(d[k],-1)  if k in d else -1 for k in _partial_keys])

def node_to_rep(node, encode_func=encode):
	feats = parse_feature_to_dict(node.feats)
	return encode_func(feats, node.xpostag)

#@cache_wrapper
def learn_from_UD_tree(input_filepath, encode_func=encode, include_reverse=False, mode="bigram", **kwargs):
	D = {}
	ud = UD_collection(codecs.open(input_filepath, encoding="utf-8"))
	total_trans = 0
	token_dict = {}

	print("training from UD tree ...")

	if mode == "dependencies":
		for sentence in tqdm(ud.sentences):
			for node in sentence.find():
				for child in node.children:
					parent_rep = node_to_rep(node,encode_func=encode_func)
					child_rep = node_to_rep(child.node,encode_func=encode_func)
					tran_rep = COMBINE_FUNC(parent_rep, child_rep)

					token_dict[parent_rep] = True
					token_dict[child_rep] = True

					increment_dict(D, tran_rep)

					if include_reverse:
						rtran_rep = COMBINE_FUNC(child_rep, parent_rep)
						increment_dict(D, rtran_rep)

					total_trans += 1

	elif mode == "bigram":
		for sentence in tqdm(ud.sentences):
			tmp = sentence.find()
			tmp.sort()
			for a,b in zip(tmp,tmp[1:]):
				arep = node_to_rep(a,encode_func=encode_func)
				brep = node_to_rep(b,encode_func=encode_func)
				increment_dict(D, arep + brep)
				token_dict[arep + brep] = True
				total_trans += 1
	elif mode == "comb":
		for sentence in tqdm(ud.sentences):
			reps = [node_to_rep(node, encode_func=encode_func) for node in sentence.find()]
			for a,b in itertools.product(*([range(len(reps))] * 2)):
				increment_dict(D, reps[a] + reps[b])
				token_dict[reps[a] + reps[b]] = True
				total_trans += 1
	elif mode == "comb23":
		for sentence in tqdm(ud.sentences):
			reps = [node_to_rep(node, encode_func=encode_func) for node in sentence.find()]
			for a,b in itertools.product(*([range(len(reps))] * 2)):
				increment_dict(D, reps[a] + reps[b])
				token_dict[reps[a]] = True
				token_dict[reps[b]] = True
				total_trans += 1
			for a,b,c in itertools.product(*([range(len(reps))] * 3)):
				increment_dict(D, reps[a] + reps[b] + reps[c])
				token_dict[reps[a]] = True
				token_dict[reps[b]] = True
				token_dict[reps[c]] = True
				total_trans += 1

	print "total trans : {} unique trans : {} unique tokens {} -> {}".format(total_trans, len(D), len(token_dict), len(token_dict)**2)

	return D

def check_for_answer(ll, target):
	for x in itertools.product(*ll):
		if list(x) == target:
			return True
	return False

def sort_dict_by_value(d):
	return sorted(d.items(), key=operator.itemgetter(1))

def make_hist(x):
	from matplotlib import pyplot as plt
	plt.hist(x)
	plt.show()

# different scoring functions
bigram_bool_score = lambda x,_ : np.sum([1 if COMBINE_FUNC(a,b) in valid_transitions else 0 for a,b in zip(x,x[1:])])

bigram_count_score = lambda x,_ : np.sum([valid_transitions.get(COMBINE_FUNC(a,b),0) for a,b in zip(x,x[1:])])

comb_bool_score = lambda x,_ : np.sum([1 if COMBINE_FUNC(x[a],x[b]) in valid_transitions else 0 for a,b in itertools.combinations(range(len(x)),2)])

comb_count_score = lambda x,_ : np.sum([valid_transitions.get(COMBINE_FUNC(x[a],x[b]),0) for a,b in itertools.combinations(range(len(x)),2)])

relation_count_score = lambda x,r : np.sum([valid_transitions[x[a] + x[b]] if x[a] + x[b] in valid_transitions else 0 for a,b in r])

relation_bool_score = lambda x,r : np.sum([1 if x[a] + x[b] in valid_transitions else 0 for a,b in r])

# different tuple combiners
default_combine = lambda a,b : a + b

case_agree_combine = lambda a,b : (a[0] == b[0],) + a[1:] + b[1:]

all_case_agree_combine = lambda a,b : tuple([i == j for i,j in zip(a,b)])

if __name__ == "__main__":

	FIN_UD_PATH = "ud/fi-ud-train.conllu"
	UD_PATH = "ud/kpv_lattice-ud-test.conllu"
	ENCODE_FUNC = partial_encode
	COMBINE_FUNC = default_combine
	SCORE_FUNC = comb_bool_score
	LEARN_MODE = "dependencies"

	ALL_UD_PATHS = [
		"ud/fi-ud-train.conllu",
		"ud/fi-ud-test.conllu",
		"ud/kpv_lattice-ud-test.conllu",
		#"ud/myv-ud.conllu",
		"ud/sme_giella-ud-train.conllu",
		"ud/sme_giella-ud-test.conllu",
	]

	fw_map, bw_map = UD_trees_to_mapping(ALL_UD_PATHS, cache="master_map.npz")
	dict_to_json("fw_master_map.json", fw_map)
	dict_to_json("bw_master_map.json", bw_map)

	exit()

	_keys = fw_map.keys() # limit to just the keys we want though


	valid_transitions = learn_from_UD_tree(
		UD_PATH, encode_func=ENCODE_FUNC, mode=LEARN_MODE)

	final_results = []
	ud = UD_collection(codecs.open(UD_PATH, encoding="utf-8"))

	"""
	# test that an incorrect reading gets a lower score
	# N_CHANGES = 1 ~= 72 %
	# N_CHANGES = 2 ~= 92 %
	N_CHANGES = 1
	results = []
	for sentence in ud.sentences:
		wrong_reading = __change_ud_morphology(sentence, N_CHANGES)
		wrong = [partial_encode(_,_["pos"]) for _ in wrong_reading]
		tmp = sentence.find()
		tmp.sort()
		target = [node_to_rep(node,encode_func=ENCODE_FUNC) for node in tmp]

		target_score = SCORE_FUNC(target,[])
		wrong_score = SCORE_FUNC(wrong,[])
		results += [target_score > wrong_score]

		print(np.mean(results), target_score, wrong_score)
	"""

	from matplotlib import pyplot as plt
	from scipy.stats import gaussian_kde
	colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
	funcs = [bigram_bool_score, bigram_count_score, comb_bool_score, comb_count_score]
	labels = ["bigram bool", "bigram count", "comb bool", "comb count"]
	handles = []

	for color, SCORE_FUNC, label in zip(colors, funcs, labels):

		# test that the correct reading gets the highest score
		results = []
		for sentence in tqdm(ud.sentences):
			all_readings = __give_all_possibilities(sentence, lang="kpv")
			all_encoded = [[partial_encode(_,_["pos"]) for _ in word] for word in all_readings]

			tmp = sentence.find()
			tmp.sort()
			target = [node_to_rep(node,encode_func=ENCODE_FUNC) for node in tmp]

			N = np.product([len(_) for _ in all_encoded])
			if N < 10000:
				# remove target from all_encoded if it is there
				wrong_scores = []
				for reading in itertools.product(*all_encoded):
					if not reading == target:
						wrong_scores += [SCORE_FUNC(reading,[])]

				target_score = SCORE_FUNC(target,[])

				if len(wrong_scores) > 0:
					results += [np.mean(np.asarray(wrong_scores) > target_score)]
					#print(np.mean(results), target_score, np.max(wrong_scores), np.mean(np.asarray(wrong_scores) > target_score), np.sum(np.asarray(wrong_scores) > target_score) )

			if len(results) > 1000:
				break


		# plot the results
		print results

		x = np.asarray(results)
		x_grid = np.linspace(0.,1.,1000)
		bandwidth=0.05
		kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
		y = kde.evaluate(x_grid)
		y /= np.sum(y)
		y = np.cumsum(y)

		plt.plot(x_grid,y,color=color,label=label)
		plt.xticks(np.linspace(0.,1.,11))

	plt.legend(loc='upper left')
	plt.show()

	exit()


	# older ranking test.
	for sentence in ud.sentences:
		tmp = sentence.find()
		tmp.sort()
		input = [node.form.encode('utf-8') for node in tmp]
		target = [node_to_rep(node) for node in tmp]

		encoded_readings = []
		partial_encoded_readings = []
		for word in get_readings(input):
			encoded_readings += [[encode(_,_["pos"]) for _ in word]]
			partial_encoded_readings += [[partial_encode(_,_["pos"]) for _ in word]]

		if np.max([len(_) for _ in encoded_readings]) > 1 and len(encoded_readings) > 1 and check_for_answer(encoded_readings, target):

			relations = []
			"""
			for node in tmp:
				for child in node.children:
					#print (node.form.encode('utf-8') == np.asarray(input))
					try:
						nidx = np.where(node.form.encode('utf-8') == np.asarray(input))[0][0]
						cidx = np.where(child.node.form.encode('utf-8') == np.asarray(input))[0][0]
						relations += [(nidx,cidx)]
					except:
						pass
			"""

			scores = np.asarray([SCORE_FUNC(x,relations) for x in itertools.product(*partial_encoded_readings)])
			choices = list(itertools.product(*encoded_readings))
			best_idx = np.where(np.max(scores) == scores)[0]

			# assertions for sanity checks
			assert len(scores) == np.product([len(_) for _ in encoded_readings])
			assert len(scores) > 1
			assert np.product([len(_) for _ in encoded_readings]) == len(list(itertools.product(*encoded_readings)))
			assert np.max([len(_) for _ in encoded_readings]) > 1

			# only include if we have made some distinction
			if len(np.unique(scores)) > 1:

				match = False
				for i in best_idx:
					if target == list(choices[i]):
						match = True
						break

				correct_idx = 0
				for i in range(len(choices)):
					if target == list(choices[i]):
						correct_idx = i
						break

				final_results += [match]

				# print some info to show what is happening
				print "SENTENCE : {}".format(" ".join(input))

				tmp = partial_encoded_readings
				print "word 0: {}".format(tmp[0])
				for i,(cur,next) in enumerate(zip(tmp,tmp[1:])):
					tran_exist = [1 if operator.add(*x) in valid_transitions else 0 for x in itertools.product(cur,next)]
					print "word {}: {} {}".format(i+1, next, tran_exist)

				print "{} : ACC {} SCORES {} CORRECT {} CORRECT_IDX {}\n\n".format(
					len(final_results), np.mean(np.asarray(final_results)), scores, match, correct_idx)

#
