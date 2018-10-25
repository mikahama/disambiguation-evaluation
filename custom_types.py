# custom classes for managing the stuff
# if "POS" in d:
# d["POS"] = ud_pos[d["POS"]]
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
	def totuple(self):
		return IntListList(self).totuple()

class IntListList(list):
	def __init__(self,*args):
		if len(args) == 1: # then we could be casting
			if isinstance(args[0], DictList):
				args = [[fw_map[k][v][1] for k,v in x.items()] for x in args[0]]
			elif type(args[0]) in [list, tuple]:
				args = [[_ for _ in x] for x in args[0]]
		assert all([isinstance(arg, list) for arg in args])
		super(IntListList,self).__init__(args)
	def __append__(self,arg):
		assert isinstance(arg, IntList)
		super(IntListList,self).__append(arg)
	def totuple(self):
		return tuple([tuple([_ for _ in x]) for x in self])
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



if __name__ == "__main__":

	# initializing dictlist
	dl = DictList({"VerbType" : "Aux", "Voice" : "Act"}, {"POS" : "N*"})
	print dl

	# check IntListList casting
	print IntListList( dl )
	print IntListList( [[34, 56], [7], [], [45]] )
	print IntListList( ((220, 56), (), (40,)) )
	print IntListList( ((220, 56), (), (40,)) ).totuple()

	# check DictList casting
	print DictList( IntListList([[45, 20], [30]]) )
	print DictList( ((220, 56), (), (40,)) )
	print DictList( [[34, 56], [7], [], [45]] )
	print DictList( [[34, 56], [7], [], [45]] ).totuple()

	# check back and forth
	print IntListList( DictList( [[34, 56], [7], [], [45]] ) )

	# check contains
	print IntListList([[3, 5], [3], [4, 6, 7]]).contains([[3, 5], [], [6]])
	print IntListList([[3, 5], [3], [4, 6, 7]]).contains([[4,5], [], []])

	print IntListList(dl).contains(DictList({"VerbType" : "Aux"}))


#
