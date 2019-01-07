#encoding: utf-8
import itertools
from subprocess import call

train_langs = ["fin", "sme"]
test_langs = ["fin", "sme", "kpv", "myv"]
min_support = ["5"]
max_pattern = ["20"]
scorings = ["dep", "gap"]
max_gaps = ["1", "2"]



for args in itertools.product(train_langs, test_langs, min_support, max_pattern,scorings, max_gaps):
	print args
	try:
		call(["python", "common.py", args[0], args[1], "--min-sup", args[2], "--max-pattern-length", args[3], "--save-results", "true", "--scoring-method", args[4], "--max-gap", args[5]])
		print "ok"
	except:
		print "fail"
