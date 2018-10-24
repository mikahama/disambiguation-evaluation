#encoding: utf-8
from uralicNLP.cg3 import Cg3
from uralicNLP import uralicApi
from uralicNLP.ud_tools import UD_collection
from nltk.tokenize import word_tokenize
#from mikatools import *
from common import *
import random
import codecs


order = ["Case", "Number", "Person", "Tense", "Connegative", "Voice"]
#mappings = json_load("fi_mappings.json")
import json
mappings = json.load(open("fi_mappings.json", "r"))
poses = ["N", "A", "V", "Adv", "CC", "CS", "Pron", "Pr", "Po", "Num", "Interj", "Punct", "Det", "Pcle"]

master_keys = ["pos"] + mappings.keys()
master_keys.sort()

def __spmf_format_sentence(sent):
	sentence = []
	for word in sent:
		w = []
		for key in master_keys:
			if key in word:
				w.append(str(word[key]))
			else:
				w.append("0")
		sentence.append(" ".join(w))
	line = " -1 ".join(sentence)
	return line

def spmf_format_sentences(sentences):
	output = []
	for sentence in sentences:
		output.append(__spmf_format_sentence(sentence))
	return " -2\n".join(output)



def __disambiguate(sentence, lang="fin"):
	#if type(sentence) == unicode:
	#	sentence = sentence.encode('utf-8')
	#tokens = word_tokenize(sentence)
	#print tokens
	cg = Cg3(lang)
	return cg.disambiguate(sentence)

def __parse_morphology(morphology):
	#CC,CS -> C
	#Pr, Po -> Adp
	reading = {}
	for pos in poses:
		if pos in morphology:
			if pos == "CS" or pos == "CC":
				reading["pos"] = "C"
				break
			elif pos == "Pr" or pos == "Po":
				reading["pos"] = "Adp"
				break
			else:
				reading["pos"] = pos
				break
	if "pos" not in reading:
		reading["pos"] = ""
	reading["pos"] = unicode(reading["pos"])
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
		if len(morph["pos"]) != 0:
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
		morphology["pos"] = node.xpostag
		if i in replace:
			replacements = __parse_fst_morphologies(uralicApi.analyze(node.form.encode('utf-8'), lang))
			was_changed = False
			for replacement in replacements:
				for k in _partial_keys:
					if k in replacement:
						if (k in morphology and replacement[k] != morphology[k]) or k not in morphology:
							morphology = replacement
							was_changed = True
							break
				if was_changed:
					break
			if not was_changed:
				morphology["pos"] = unicode(random.choice(poses))
		sent.append(morphology)
	return sent

def __give_all_possibilities(ud_sentence, lang="fin"):
	nodes = ud_sentence.find()
	nodes.sort()
	sent = []
	for node in nodes:
		fst_output = uralicApi.analyze(node.form.encode('utf-8'), lang)
		forms = __parse_fst_morphologies(fst_output)
		sent.append([dict(t) for t in {tuple(d.items()) for d in forms}])
	return sent





def produce_tests():
	ud = UD_collection(codecs.open("ud/fi-ud-train.conllu", encoding="utf-8"))
	for sentence in ud.sentences[3:]:
		print unicode(sentence)
		morphs = __give_all_possibilities(sentence)
		print morphs
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
	sentences = [[{"pos":"N", "Case": "Nom"}, {"pos": "A", "Number":"Sg"}], [{"pos":"Adv", "Mood":"Ind"}]]
	print spmf_format_sentences(sentences)
	#produce_tests()
	#dict_to_json("bw_map_sme.json", bw_map)
	#dict_to_json("fw_map_sme.json", fw_map)

