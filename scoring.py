import random
from custom_types import *

class Scoring(object):
	"""docstring for Scoring"""
	def __init__(self, results):
		assert isinstance(results, Results)
		self.results = results

class ScoreSentence(Scoring):
	"""docstring for ScoreSentence"""
	def __init__(self, *args, **kwargs):
		super(ScoreSentence, self).__init__(*args)

	def score(self, x):
		count = 0
		for pattern in self.results.patterns:
			count += int(x.contains(pattern))
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
				if x.contains(gp):
					count += self.results.gap_distributions[p.to_tuple()][gc]
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
			if x.contains(k):
				count += v
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
