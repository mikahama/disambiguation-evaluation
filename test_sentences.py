#encoding: utf-8
from uralicNLP.cg3 import Cg3
from uralicNLP import uralicApi
from uralicNLP.ud_tools import UD_collection
from nltk.tokenize import word_tokenize
#from common import *
import numpy as np
import random
import codecs
from subprocess import call
from maps import ud_pos
import collections
from custom_types import *
from fw_master_map import fw_map


order = ["Case", "Number", "Person", "Tense", "Connegative", "Voice"]
import json
mappings = json.load(open("fi_mappings.json", "r"))
poses = ["N", "A", "V", "Adv", "CC", "CS", "Pron", "Pr", "Po", "Num", "Interj", "Punct", "Det", "Pcle"]

master_keys = ["POS"] + mappings.keys()
master_keys.sort()

def run_spmf(algorithm, input_file, output_file, min_sup=50, spmf_path="spmf.jar"):
	call(["java", "-jar", spmf_path, "run", algorithm, input_file, output_file, str(min_sup)+"%"])

def __spmf_format_sentence(sent):
	sentence = []
	for num_list in sent:
		w = []
		for num in num_list:
			w.append(str(num))
		sentence.append(" ".join(w))
	line = " -1 ".join(sentence)
	return line

def spmf_format_sentences(sentences):
	output = []
	for sentence in sentences:
		output.append(__spmf_format_sentence(sentence))
	return " -2\n".join(output) + " -2"

def spmf_format_to_file(sentences, file_path):
	f = open(file_path, "w")
	f.write(spmf_format_sentences(sentences))
	f.close()

def encode(x,pad_value=2**16):
	x = x.add_values(pad_value)
	x.append([pad_value])
	return x

def decode(x,pad_value=2**16):
	x = IntListList(x).remove_values(pad_value)
	return x.remove_empty_margin_gaps()

def __parse_spmf_line(line):
	line = line.replace("\n", "")
	# replace possible multiple spaces to single to prevent parsing from failing
	line = " ".join(line.split())
	try:
		if "#SID:" in line:
			numbers, score_and_sid = line.split(" -1 #SUP: ")
			score, sid = score_and_sid.split(" #SID: ")
			sid = list(map(int, sid.split()))
		else:
			numbers, score = line.split(" -1 #SUP: ")
			sid = []
	except Exception as e:
		print(e)
		return None
	patterns = numbers.split("-1")
	s = []
	for pattern in patterns:
		parts = pattern.split()
		parts = tuple(map(int, parts))
		s.append(parts)
	return IntListList(s), int(score), sid

def read_spmf_output(file_path, pad_value=None):
	f = open(file_path, "r")
	score_dict = []
	sid_dict = []
	for line in f:
		_out = __parse_spmf_line(line)
		if _out is not None:
			key, score, sid = _out
			if pad_value is not None:
				key = decode(key, pad_value=pad_value)
			score_dict += [(key.to_tuple(), score)]
			sid_dict += [(key.to_tuple(), sid)]
	return Results(ResultDict(score_dict), ResultDict(sid_dict))

def run_spmf_full(X, algorithm="VMSP", min_sup=5, spmf_path="spmf.jar", max_pattern_length=100, max_gap=1, save_results_to="tmp_spmf_output.txt", temp_file="tmp_spmf.txt", pad_value=None):
	"""
	min_pattern_length : the min number of integers in a pattern,
	max_pattern_length : the max number of integers in a pattern,
	min_pattern_span : the min number of itemsets in a pattern,
	max_pattern_span : the max number of itemsets in a pattern,
	max_span_gap : the max number of consecutive empty itemsets in a pattern,
	pad_value : the value to pad each itemset with (for patterns with gaps)
	"""

	if pad_value is not None:
		X = list(map(lambda x : encode(x,pad_value=pad_value), X))

	spmf_format_to_file(X, temp_file)
	basic_call = ["java", "-jar", spmf_path, "run", algorithm, temp_file, save_results_to, str(min_sup)+"%"]
	if algorithm == "MaxSP":
		call(basic_call + ["false"])
	elif algorithm in ["VMSP", "VGEN"]:
		call(basic_call + [str(max_pattern_length), str(max_gap), "true"])
	elif algorithm in ["FEAT", "FSGP"]:
		call(basic_call + [str(max_pattern_length), "false"])
	else:
		call(basic_call)
	return read_spmf_output(
		save_results_to,pad_value=pad_value)

def __disambiguate(udsentence, lang="fin"):
	tmp = udsentence.find()
	tmp.sort()
	cg = Cg3(lang)
	sentence = [x.form.encode('utf-8') for x in tmp]
	return cg.disambiguate(sentence)

def __parse_morphology(morphology):
	#CC,CS -> C
	#Pr, Po -> Adp
	reading = {}
	for pos in poses:
		if pos in morphology:
			if pos == "CS" or pos == "CC":
				reading["POS"] = "C"
				break
			elif pos == "Pr" or pos == "Po":
				reading["POS"] = "Adp"
				break
			else:
				reading["POS"] = pos
				break
	if "POS" not in reading:
		reading["POS"] = "X"
	reading["POS"] = unicode(reading["POS"])
	for mapping, map_dict in mappings.iteritems():
		for item in map_dict:
			if item in morphology:
				reading[mapping] = map_dict[item]
	return reading


def __parse_fst_morphologies(morphology_list):
	output = []
	for morphology in morphology_list:
		m = morphology[0].replace("@@","+").replace("@","+")
		m = m.split("+")[1:]
		morph = __parse_morphology(m)
		if len(morph["POS"]) != 0:
			output.append(morph)
	return output

def __change_ud_morphology(sentence, change_x_times, lang="fin"):
	sent = []
	random.seed()
	nodes = sentence.find()
	nodes.sort()
	if len(nodes) <= change_x_times:
		replace = range(len(nodes))
	else:
		replace = range(len(nodes))
		replace = random.sample(replace, change_x_times)
	for i in range(len(nodes)):
		node = nodes[i]
		morphology = parse_feature_to_dict(node.feats)
		morphology["POS"] = node.xpostag
		if i in replace:
			replacements = __parse_fst_morphologies(uralicApi.analyze(node.form.encode('utf-8'), lang))
			was_changed = False
			for replacement in replacements:
				for k in fw_map.keys():
					if k in replacement:
						if (k in morphology and replacement[k] != morphology[k]) or k not in morphology:
							morphology = replacement
							was_changed = True
							break
				if was_changed:
					break
			if not was_changed:
				morphology["POS"] = unicode(random.choice(poses))
		sent.append(morphology)
	return sent

def change_ud_morphology(sentence, n, lang="fin"):
	poss = __give_all_possibilities(sentence, lang=lang)
	current = UD_sentence_to_dict(sentence)
	valid_idx = np.where([len(_) > 1 for _ in poss])[0]
	if len(valid_idx) < n:
		return None
	idx = np.random.choice(valid_idx, size=(n,), replace=False)
	out = []
	for i in range(len(current)):
		if not i in idx:
			out += [current[i]]
		else:
			out += [random.choice([p for p in poss[i] if not p == current[i]])]
	return out

def give_all_possibilities(ud_sentence, lang="fin"):
	nodes = ud_sentence.find()
	nodes.sort()
	sent = []
	for node in nodes:
		fst_output = uralicApi.analyze(node.form.encode('utf-8'), lang)
		forms = __parse_fst_morphologies(fst_output)
		if len(forms) == 0:
			forms.append({})
		sent.append([dict(t) for t in {tuple(d.items()) for d in forms}])
	return sent

def __give_all_possibilities(ud_sentence, lang="fin"):
	#method was renamed, but it's still used elsewhere...
	return give_all_possibilities(ud_sentence, lang)





def produce_tests():
	ud = UD_collection(codecs.open("ud/fi-ud-train.conllu", encoding="utf-8"))
	for sentence in ud.sentences[3:]:
		print(unicode(sentence))
		morphs = __give_all_possibilities(sentence)
		print(morphs)
		quit()


def get_readings(sentence, lang):
	disambiguations = __disambiguate(sentence, lang)
	results = []
	for disambiguation in disambiguations:
		possible_words = disambiguation[1]
		word_readings = []
		for possible_word in possible_words:
			morph = __parse_morphology(possible_word.morphology)
			word_readings.append(morph)
		results.append(word_readings)
	return results


if __name__ == '__main__':
	"""
	disambiguations = __disambiguate(u"alan laulamaan")
	for disambiguation in disambiguations:
		possible_words = disambiguation[1]
		for possible_word in possible_words:
			print possible_word.morphology
	print get_readings(s)
	"""
	#UD_PATH = "ud/sme_giella-ud-train.conllu"
	#fw_map, bw_map = UD_trees_to_mapping(UD_PATH, cache="test_sme.npz")
	#sentences = [[[54, 34], [89,78]], [[0,8]]]
	#print spmf_format_sentences(sentences)
	#print __parse_spmf_line("2 3 -1 1 4 -1 #SUP: 2")
	#print read_spmf_output("test.txt")
	#run_spmf("SPADE", "test_spmf.txt", "test_spmf_out.txt")
	#spmf_format_to_file(sentences, "test.txt")
	#produce_tests()
	#dict_to_json("bw_map_sme.json", bw_map)
	#dict_to_json("fw_map_sme.json", fw_map)
