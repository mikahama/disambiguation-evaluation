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
from tqdm import tqdm

def parse_feature_to_dict(s):
	if s == "_":
		return {}
	return {_.split("=")[0] : _.split("=")[1] for _ in s.split("|")}

def parse_feature(s):
	if s == "_":
		return []
	return [tuple(_.split("=")) for _ in s.split("|")]

def parse_node_to_dict(node):
	d = parse_feature_to_dict(node.feats)
	d["POS"] = ud_pos[node.xpostag]
	return d

def UD_sentence_to_list(sentence, match_empty_nodes=False):
	tmp = sentence.find(match_empty_nodes=match_empty_nodes)
	tmp.sort()
	return IntListList(DictList(*[parse_node_to_dict(node) for node in tmp]))

def UD_sentence_to_dict(sentence, match_empty_nodes=False):
	tmp = sentence.find(match_empty_nodes=match_empty_nodes)
	tmp.sort()
	return [parse_node_to_dict(node) for node in tmp]

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

	def remove_values(self,vals):
		if not isinstance(vals, list):
			vals = [vals]
		return IntListList([[_ for _ in x if not _ in vals] for x in self])

	def add_values(self,vals):
		if not isinstance(vals, list):
			vals = [vals]
		return IntListList([x + vals for x in self])

	def remove_empty_margin_gaps(self):
		# remove empty lists at beginning and end of IntListList
		x = copy.deepcopy(self)
		while len(x) > 0 and len(x[0]) == 0:
			x.pop(0)
		while len(x) > 0 and len(x[-1]) == 0:
			x.pop(-1)
		return x

	def count_gaps(self):
		i = 0
		counts = []
		while i < len(self):
			if len(self[i]) == 0:
				count = 0
				while len(self[i]) == 0:
					count += 1
					i += 1
				counts += [count]
			else:
				i += 1
		return counts

	def max_gap(self):
		counts = self.count_gaps()
		if len(counts) == 0:
			return 0
		elif len(counts) == 1:
			return counts[0]
		return max(*counts)

	def n_gaps(self):
		return len(self.count_gaps())

	def intersection(self,x):
		# measures the intersection of two IntListList
		x = IntListList(x)
		inter = 0
		for i in range(min(len(self), len(x))):
			inter += len(set(self[i]).intersection(set(x[i])))
		return float(inter) / float(max(self.nested_len(),x.nested_len()))

	def contains(self,x,verbose=False,return_count=True):
		# does the IntListList contain x
		# contains_pattern([[3, 5], [3], [4, 6, 7]], [[3, 5], [], [6]]) == True
		# contains_pattern([[3, 5], [3], [4, 6, 7]], [[4,5], [], []]) == False
		x = IntListList(x)
		count = 0
		for i in range(len(self)-len(x)+1):
			j = 0
			while j < len(x) and i+j < len(self) and set(x[j]).issubset(set(self[i+j])):
				j += 1
			if j == len(x):
				if not return_count:
					return True
				count += 1
		if not return_count:
			return False
		return count

	def contains_and_length_match(self,x,verbose=False):
		# does the IntListList contain x and match the length
		if len(self)==len(x) and self.contains(x,verbose=verbose):
			return True
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

	def filter_keys(self, func):
		new_dict = []
		for k,v in self.items():
			if func(k):
				new_dict += [(k,v)]
		return ResultDict(new_dict)

	def filter_vals(self, func):
		new_dict = []
		for k,v in self.items():
			if func(v):
				new_dict += [(k,v)]
		return ResultDict(new_dict)

	def filter(self, func):
		new_dict = []
		for k,v in self.items():
			if func(k,v):
				new_dict += [(k,v)]
		return ResultDict(new_dict)

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
		self.dep_scores = None

	def extend(self, results_object):
		self.score_dict.update(results_object.score_dict)
		self.sid_dict.update(results_object.sid_dict)
		self.patterns = self.sid_dict.map_keys_to_list(IntListList)
		self.gap_distributions = None
		self.dep_scores = None

	def pattern_vector(self,x):
		return np.asarray([x.contains(pattern,return_count=False) for pattern in self.patterns])

	def calculate_stats(self):
		pattern_lengths = self.score_dict.map_keys_to_list(
			lambda x : IntListList(x).nested_len())
		pattern_spans = self.score_dict.map_keys_to_list(
			lambda x : len(x))
		print "NUMBER OF PATTERNS : {}".format(len(self.patterns))
		print "MAX PATTERN LENGTH : {}".format(np.max(pattern_lengths))
		print "MEAN PATTERN LENGTH : {}".format(np.mean(pattern_lengths))
		print "MIN PATTERN SPAN : {}".format(np.min(pattern_spans))
		print "MAX PATTERN SPAN : {}".format(np.max(pattern_spans))

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
			for k,v in tqdm(self.sid_dict.items()):
				gd = self.calculate_gap_distribution(
					X,IntListList(k),v,max_gap=max_gap)
				gds += [(k,gd)]
			gds = ResultDict(gds).map_vals(lambda x : x.norm_vals(
				min_value=min_value, max_value=max_value))
			self.gap_distributions = gds
		return self.gap_distributions

	def calculate_dependency_scores(self,data,UD,min_value=0.,max_value=1.):
		# make data into dependencies
		# data unused now
		if self.dep_scores is None:
			dep_data = []
			for ud in UD:
				for udsentence, sentence in zip(ud.sentences, data):
					tmp = [node for node in udsentence.find(match_empty_nodes = True)]
					tmp.sort()
					sentence = UD_sentence_to_list(
						udsentence, match_empty_nodes=True)
					mapping = {}
					for node in tmp:
						mapping[node.id] = len(mapping)

					for node in tmp:
						for child in node.children:
							deps = [mapping[node.id], mapping[child.node.id]]
							dep_data += [IntListList(
								[sentence[i] for i in deps])]

			dep_scores = []
			for pattern in self.patterns:
				count = int(np.sum([dep.contains_and_length_match(pattern) for dep in dep_data]))
				dep_scores += [(pattern.to_tuple(), count)]

			self.dep_scores = ResultDict(dep_scores).norm_vals(
				min_value=min_value, max_value=max_value)

		return self.dep_scores







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

	print IntListList([[], [], [], [123], [], [3,4], [5,6], []]).remove_empty_margin_gaps()

	print IntListList([[123], [], [3,4], [5,6], [], [], [], ]).remove_empty_margin_gaps()

	print IntListList([[123], [], [], [34], [], [], [], [], [45]]).count_gaps()

#
