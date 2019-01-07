import random
import numpy as np
from custom_types import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

#from test_sentences import __change_ud_morphology as cud
from test_sentences import change_ud_morphology

class Scoring(object):
	"""docstring for Scoring"""
	def __init__(self, results):
		#assert isinstance(results, Results)
		self.results = results

class ScoreSentenceByLinearRegression(Scoring):
	def __init__(self, sid_dict, X, Y, **kwargs):
		self.clf = LogisticRegression()
		self.clf.fit(X,Y)
		super(ScoreSentenceByLinearRegression, self).__init__(sid_dict)

	def score(self, x):
		return self.clf.predict_proba(
			self.results.pattern_vector(x)[None,:] )[0,1]

# write a class to try to seperate based on
class ScoreSentenceByLinearRegressionOLD(Scoring):
	def __init__(self, res, UD=None, train_langs=[], **kwargs):
		assert UD is not None
		# create a dataset with labels
		X = []
		Y = []
		print("CREATING DATA SET")
		for ud,lang in zip(UD,train_langs):
			#if lang == "sme":
			for sentence in tqdm(ud.sentences):
				#wrong_reading = cud(sentence, np.random.randint(5)+1, lang=lang)
				target = UD_sentence_to_list(sentence)
				X += [res.pattern_vector(target)]
				Y += [1]
				for i in range(4):
					wrong_reading = change_ud_morphology(
						sentence, np.random.randint(5)+1, lang=lang)
					if wrong_reading is not None:
						wrong = IntListList(DictList(*wrong_reading))
						X += [res.pattern_vector(wrong)]
						Y += [0]

		# train the decision tree
		print("TRAINING LINEAR REGRESSION")
		#self.clf = LinearRegression()
		#self.clf = GradientBoostingClassifier()
		self.clf = LogisticRegression()
		self.clf.fit(np.asarray(X), np.asarray(Y))
		super(ScoreSentenceByLinearRegression, self).__init__(res)

	def score(self, x):
		#return self.clf.predict( self.results.pattern_vector(x)[None,:] )[0]
		return self.clf.predict_proba( self.results.pattern_vector(x)[None,:] )[0,1]


class ScoreSentence(Scoring):
	"""docstring for ScoreSentence"""
	def __init__(self, *args, **kwargs):
		super(ScoreSentence, self).__init__(*args)

	def score(self, x):
		count = 0
		for pattern in self.results.patterns:
			count += x.contains(pattern)
		return count

class ScoreSentenceByCounts(Scoring):
	def __init__(self, *args, **kwargs):
		super(ScoreSentenceByCounts, self).__init__(*args)

	def score(self, x):
		count = 0
		for pattern,score in self.results.score_dict.items():
			count += (score * x.contains(pattern))
		return count

class ScoreSentenceByGapFreq(Scoring):
	"""docstring for ScoreSentenceByGapFreq"""
	def __init__(self, res, data=None, max_gap=1, min_value=0., max_value=1., **kwargs):
		assert data is not None
		self.max_gap = max_gap
		res.calculate_gap_distributions(
			data, max_gap=max_gap, min_value=min_value, max_value=max_value)
		#for k,v in res.gap_distributions.items():
		#	print k,v
		super(ScoreSentenceByGapFreq, self).__init__(res)

	def score(self,x):
		count = 0
		for p in self.results.patterns:
			for gc, gp in p.all_gapped_versions(max_gap=self.max_gap).items():
				n = x.contains(gp)
				count += (self.results.gap_distributions[p.to_tuple()][gc] * n)
		return count

class ScoreSentenceByDependencies(Scoring):
	def __init__(self, res, data=None, UD=None, min_value=0., max_value=1., **kwargs):
		assert data is not None
		assert UD is not None
		res.calculate_dependency_scores(
			data, UD, min_value=min_value, max_value=max_value)
		super(ScoreSentenceByDependencies, self).__init__(res)

	def score(self, x):
		count = 0
		for k,v in self.results.dep_scores.items():
			count += (v * x.contains(k))
		return count

class RandomScore(Scoring):
	"""docstring for ScoreSentence"""
	def __init__(self, *args):
		super(RandomScore, self).__init__(*args)

	def score(self):
		random.seed()
		match_count = 0
		for patt in self.results:
			for i in range(len(self.Y) - len(patt)):
				match = random.choice([True, False])
				if match:
					match_count += self.result_dict[patt] # for counts
					#match_count += 1 # for bool

		return match_count
