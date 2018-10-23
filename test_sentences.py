#encoding: utf-8
from uralicNLP.cg3 import Cg3
from nltk.tokenize import word_tokenize
#from mikatools import *

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
	for mapping, map_dict in mappings.iteritems():
		for item in map_dict:
			if item in morphology:
				reading[mapping] = map_dict[item]
	return reading

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

#print get_readings(u"Kyseisen vaatimustenmukaisuutta arvioivien elinten toimintaa voidaan parantaa huomattavasti.")

if __name__ == '__main__':
	disambiguations = __disambiguate(u"alan laulamaan")
	for disambiguation in disambiguations:
		possible_words = disambiguation[1]
		for possible_word in possible_words:
			print possible_word.morphology
