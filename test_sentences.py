#encoding: utf-8
from uralicNLP.cg3 import Cg3
from uralicNLP import uralicApi
from uralicNLP.ud_tools import UD_collection
from nltk.tokenize import word_tokenize
#from mikatools import *
from common import parse_feature_to_dict
import random
import codecs

order = ["Case", "Number", "Person", "Tense", "Connegative", "Voice"]
#mappings = json_load("fi_mappings.json")
import json
mappings = json.load(open("fi_mappings.json", "r"))


def __disambiguate(sentence):
	#if type(sentence) == unicode:
	#	sentence = sentence.encode('utf-8')
	#tokens = word_tokenize(sentence)
	#print tokens
	cg = Cg3("fin")
	return cg.disambiguate(sentence)

def __parse_morphology(morphology):
	#CC,CS -> C
	#Pr, Po -> Adp
	reading = {}
	poses = ["N", "A", "V", "Adv", "CC", "CS", "Pron", "Pr", "Po", "Num", "Interj", "Punct"]
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
		m = morphology[0].split("+")[1:]
		output.append(__parse_morphology(m))
	return output

def __change_ud_morphology(sentence, change_x_times):
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
		if i in replace:
			morphology = parse_feature_to_dict(node.feats)
			morphology["pos"] = node.xpostag
			print morphology
			replacements = __parse_fst_morphologies(uralicApi.analyze(node.form.encode('utf-8'), "fin"))
			print replacements
			quit()


def produce_tests():
	ud = UD_collection(codecs.open("ud/fi-ud-train.conllu", encoding="utf-8"))
	for sentence in ud.sentences:
		__change_ud_morphology(sentence, 1)


def get_readings(sentence):
	disambiguations = __disambiguate(sentence)
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
	produce_tests()

