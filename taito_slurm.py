import itertools

lang = ["kpv", "sme", "fin","est","myv"]

langs = list(itertools.product(lang, lang))
langs += list(zip([x for x in langs if x[0] != x[1]], [(x[1],) for x in langs if x[0] != x[1]]))

seed = 4568

template = open("slurm_template.sh", "r").read()
for train_lang, test_lang in langs:
	if type(train_lang) == tuple:
		train = " ".join(train_lang)
	else:
		train = train_lang
	if type(test_lang) == tuple:
		test = " ".join(test_lang)
	else:
		test = test_lang
	output = template.replace("TRAIN", train).replace("TEST", test)
	output = output.replace("SEED", str(seed))
	f = open("slurm/" + train.replace(" ","_") + "-" + test + "_" + str(seed) + ".sh", "w")
	f.write(output)
	f.close()


