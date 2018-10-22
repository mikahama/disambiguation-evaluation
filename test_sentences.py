#encoding: utf-8
from uralicNLP.cg3 import Cg3
from nltk.tokenize import word_tokenize

order = [pos, case, number, person, tense, conneg, voice]

def __disambiguate(sentence):
	if type(sentence) == unicode:
		sentence = sentence.encode('utf-8')
	tokens = word_tokenize(sentence)
	cg = Cg3("fin")
	return cg.disambiguate(tokens)

def __parse_morphology(morphology):
	#CC,CS -> C
	#Pr, Po -> Adp
	pass

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

disambiguations = __disambiguate(u"paitsi poliittisesti huolestuttavaa myös tieteellisesti pätemätöntä")

for disambiguation in disambiguations:
	possible_words = disambiguation[1]
	for possible_word in possible_words:
		print possible_word.lemma, possible_word.morphology