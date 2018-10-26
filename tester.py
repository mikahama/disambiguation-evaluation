import itertools
from subprocess import call

train_langs = ["fin", "sme"]
test_langs = ["fin", "sme", "kpv", "myv"]
min_support = ["5", "10", "20"]
max_pattern = ["5", "20"]

for args in itertools.product(train_langs, test_langs, min_support, max_pattern):
	print args
	try:
		call(["python", "common.py", args[0], args[1], "--min-sup", args[2], "--max-pattern-length", args[3], "--save-results", "true"])
		print "ok"
	except:
		print "fail"
