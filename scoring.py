import random

class Scoring(object):
	"""docstring for Scoring"""
	def __init__(self, results, Y, result_dict):
		self.results = results
		self.Y = Y
		self.result_dict = result_dict


class ScoreSentence(Scoring):
	"""docstring for ScoreSentence"""
	def __init__(self, *args):
		super(ScoreSentence, self).__init__(*args)

	def score(self):
		match_count = 0
		for patt in self.results:
			for i in range(len(self.Y) - len(patt)):
				match = True
				for j in range(len(patt)):
					if not set(patt[j]).issubset(set(self.Y[i+j])):
						match = False
						break
				if match:
					match_count += self.result_dict[patt] # for counts
					#match_count += 1 # for bool

		return match_count
		
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
