import codecs
from uralicNLP.ud_tools import UD_collection
import os
import numpy as np
np.warnings.filterwarnings('ignore')
import json
import itertools
from tqdm import tqdm
import operator
from test_sentences import get_readings, __change_ud_morphology
from common import *





def increment_dict(d,k):
	d[k] = d.get(k,0) + 1






def encode(d, pos):
	assert type(pos) == unicode
	return (pos,) + tuple([fw_map[k][d[k]] if k in d else -1 for k in _keys])

def partial_encode(d, pos):
	assert type(pos) == unicode
	return (pos,) + tuple([fw_map[k][d[k]] if k in d else -1 for k in _partial_keys])

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
					tran_rep = parent_rep + child_rep

					token_dict[parent_rep] = True
					token_dict[child_rep] = True

					increment_dict(D, tran_rep)

					if include_reverse:
						rtran_rep = child_rep + parent_rep
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
bigram_bool_score = lambda x,_ : np.sum([1 if a + b in valid_transitions else 0 for a,b in zip(x,x[1:])])

bigram_count_score = lambda x,_ : np.sum([valid_transitions[a + b] if a + b in valid_transitions else 0 for a,b in zip(x,x[1:])])

comb_bool_score = lambda x,_ : np.sum([1 if x[a] + x[b] in valid_transitions else 0 for a,b in itertools.combinations(range(len(x)),2)])

relation_count_score = lambda x,r : np.sum([valid_transitions[x[a] + x[b]] if x[a] + x[b] in valid_transitions else 0 for a,b in r])

relation_bool_score = lambda x,r : np.sum([1 if x[a] + x[b] in valid_transitions else 0 for a,b in r])

if __name__ == "__main__":

	UD_PATH = "ud/fi-ud-train.conllu"
	ENCODE_FUNC = partial_encode
	SCORE_FUNC = comb_bool_score
	LEARN_MODE = "dependencies"

	fw_map, bw_map = UD_tree_to_mapping(UD_PATH, cache="test.npz")
	dict_to_json("fw_map.json", fw_map)
	dict_to_json("bw_map.json", bw_map)

	_keys = fw_map.keys() # limit to just the keys we want though


	valid_transitions = learn_from_UD_tree(
		UD_PATH, encode_func=ENCODE_FUNC, mode=LEARN_MODE) #cache="valid_transitions_{}_{}.npz".format(ENCODE_FUNC.__name__, LEARN_MODE))

	final_results = []
	ud = UD_collection(codecs.open(UD_PATH, encoding="utf-8"))

	# test whether an incorrect reading gets a lower score or not
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
