import collections
import operator
import itertools
import copy
import numpy as np
np.warnings.filterwarnings('ignore')
from combinatorics import unlabeled_balls_in_labeled_boxes
from fw_master_map import fw_map
from backward_map import bw_map
from maps import ud_pos

class DictList(list):
	def __init__(self,*args):
		if len(args) == 1: # then we could be casting
			if type(args[0]) in [list, tuple, IntListList]:
				args = [{bw_map[_][0]:bw_map[_][1] for _ in x} for x in args[0]]
		assert all([isinstance(arg, dict) for arg in args])
		for d in args:
			if "POS" in d:
				d["POS"] = ud_pos[d["POS"]]
		super(DictList,self).__init__(args)

	def __append__(self,arg):
		assert isinstance(arg, dict)
		super(DictList,self).__append(arg)

	def to_tuple(self):
		return IntListList(self).to_tuple()

class IntListList(list):
	def __init__(self,*args):
		if len(args) == 1: # then we could be casting
			if isinstance(args[0], DictList):
				args = [[fw_map[k][v][1] for k,v in x.items()] for x in args[0]]
			elif type(args[0]) in [list, tuple, IntListList]:
				args = [[_ for _ in x] for x in args[0]]
		assert all([isinstance(arg, list) for arg in args])
		super(IntListList,self).__init__(args)

	def __append__(self,arg):
		assert isinstance(arg, IntList)
		super(IntListList,self).__append(arg)

	def to_tuple(self):
		return tuple([tuple([_ for _ in x]) for x in self])

	def nested_len(self):
		return int(np.sum([len(_) for _ in self]))

	def contains(self,x,verbose=False):
		# does the IntListList contain x
		# contains_pattern([[3, 5], [3], [4, 6, 7]], [[3, 5], [], [6]]) == True
		# contains_pattern([[3, 5], [3], [4, 6, 7]], [[4,5], [], []]) == False
		x = IntListList(x)
		for i in range(len(self)-len(x)+1):
			j = 0
			while j < len(x) and i+j < len(self) and set(x[j]).issubset(set(self[i+j])):
				j += 1
			if j == len(x):
				if verbose:
					print "{} contains {}".format(X,pattern)
				return True
		if verbose:
			print "{} does NOT contain {}".format(X,pattern)
		return False

	def insert_gaps(self,sizes):
		# insert x which could be an item or a list
		assert len(sizes) == (len(self)-1)
		n_gaps = len(self)-1
		add_count = 0
		x = copy.deepcopy(self)
		for i,size in zip(range(n_gaps),sizes):
			for _ in range(size):
				x.insert(add_count+i+1,[] * size)
				add_count += 1
		return x

	def all_gapped_versions(self,max_gap=1):
		max_gap -= 1
		gappedversions = [((0,)*(len(self)-1), self)]
		if max_gap > 0:
			box_sizes = [max_gap]*(len(self)-1)
			ball_range = range(1,max_gap*(len(self)-1)+1)
			for counts in itertools.chain.from_iterable(
			unlabeled_balls_in_labeled_boxes(b,box_sizes) for b in ball_range):
				gappedversions += [(counts, self.insert_gaps(counts))]
		return ResultDict(gappedversions)

class ResultDict(collections.OrderedDict):
	def map_keys(self, func):
		newdict = []
		for k,v in self.items():
			newdict += [(func(k),v)]
		return ResultDict(newdict)

	def map_vals(self, func):
		newdict = []
		for k,v in self.items():
			newdict += [(k,func(v))]
		return ResultDict(newdict)

	def map_keys_to_list(self, func):
		return [func(k) for k in self.keys()]

	def map_vals_to_list(self, func):
		return [func(v) for v in self.values()]

	def norm_vals(self, min_value=0., max_value=1.):
		# return values on the normalized range 0.,1.
		_min = float(np.min(list(self.values())))
		_max = float(np.max(list(self.values())))
		if (_min - _max) == 0:
			return self.map_vals(lambda x : max_value)
		return self.map_vals(lambda x : (float(x) - _min) / (_max - _min) * (max_value - min_value) + min_value)

	def prob_vals(self):
		sum = float(np.sum(list(self.values())))
		return self.map_vals(lambda x : float(x) / sum)

class Results(object):
	def __init__(self,score_dict,sid_dict):
		assert isinstance(score_dict, ResultDict)
		assert isinstance(sid_dict, ResultDict)
		self.score_dict = score_dict
		self.sid_dict = sid_dict
		self.patterns = self.sid_dict.map_keys_to_list(IntListList)
		self.gap_distributions = None

	def calculate_stats(self):
		pattern_lengths = self.score_dict.map_keys_to_list(
			lambda x : IntListList(x).nested_len())
		print "NUMBER OF PATTERNS : {}".format(len(self.patterns))
		print "MAX PATTERN LENGTH : {}".format(np.max(pattern_lengths))
		print "MEAN PATTERN LENGTH : {}".format(np.mean(pattern_lengths))

	def calculate_gap_distribution(self,data,ungapped_pattern,idx,max_gap=1):
		gapped_patterns = ungapped_pattern.all_gapped_versions(max_gap=max_gap)
		gap_distribution = []
		for counts, pattern in gapped_patterns.items():
			frequency = 0
			for i in idx:
				frequency += int(data[i].contains(pattern))
			gap_distribution += [(counts, frequency)]
		return ResultDict( gap_distribution )

	def calculate_gap_distributions(self,X,max_gap=1,min_value=0.,max_value=1.):
		if self.gap_distributions is None:
			gds = []
			for k,v in self.sid_dict.items():
				gd = self.calculate_gap_distribution(
					X,IntListList(k),v,max_gap=max_gap)
				gds += [(k,gd)]
			gds = ResultDict(gds).map_vals(lambda x : x.norm_vals())
			self.gap_distributions = gds
		return self.gap_distributions


if __name__ == "__main__":

	# initializing dictlist
	dl = DictList({"VerbType" : "Aux", "Voice" : "Act"}, {"POS" : "N*"})
	print dl

	# check IntListList casting
	print IntListList( dl )
	print IntListList( [[34, 56], [7], [], [45]] )
	print IntListList( ((220, 56), (), (40,)) )
	print IntListList( ((220, 56), (), (40,)) ).to_tuple()

	# check DictList casting
	print DictList( IntListList([[45, 20], [30]]) )
	print DictList( ((220, 56), (), (40,)) )
	print DictList( [[34, 56], [7], [], [45]] )
	print DictList( [[34, 56], [7], [], [45]] ).to_tuple()

	# check back and forth
	print IntListList( DictList( [[34, 56], [7], [], [45]] ) )

	# check contains
	print IntListList([[3, 5], [3], [4, 6, 7]]).contains([[3, 5], [], [6]])
	print IntListList([[3, 5], [3], [4, 6, 7]]).contains([[4,5], [], []])

	print IntListList(dl).contains(DictList({"VerbType" : "Aux"}))

	print IntListList([[3, 5],[3],[4, 6, 7]]).insert_gaps([2,3])

	print IntListList([[3, 5],[3],[4, 6, 7]]).insert_gaps([2,3]).to_tuple()

	d = ResultDict([(4,5), (6,20), (7,6), (8,9), (5,10)])
	print d.map_vals(lambda x : x ** 2)
	print d.norm_vals(min_value=0.25,max_value=1.)
	print d.prob_vals()

	dd = ResultDict([(1,d),(2,d)])
	print dd.map_vals(lambda x : x.prob_vals())

	print IntListList([[3, 5], [3], [4, 6, 7]]).all_gapped_versions(max_gap=3)

	print ResultDict([(((4,), (5,6)), 5), (((3,), ()), 7)]).map_keys_to_list(IntListList)

	print IntListList([[123]])[0]

	print IntListList([[3, 5], [3], [4, 6, 7]]).nested_len()

#
