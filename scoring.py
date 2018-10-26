import random
from custom_types import *

class Scoring(object):
	"""docstring for Scoring"""
	def __init__(self, results):
		assert isinstance(results, Results)
		self.results = results

class ScoreSentence(Scoring):
	"""docstring for ScoreSentence"""
	def __init__(self, *args):
		super(ScoreSentence, self).__init__(*args)

	def score(self, x):
		count = 0
		for pattern in self.results.patterns:
			count += int(x.contains(pattern))
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
