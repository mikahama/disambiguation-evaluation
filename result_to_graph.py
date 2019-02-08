# get the results of the stuff
import os
import re
import json
import numpy as np
from matplotlib import pyplot as plt
import itertools

from matplotlib.backends.backend_pdf import PdfPages

lang = ["kpv", "sme", "fin", "est", "myv"]

train_langs = [" ".join(x) for x in itertools.product(lang, lang)] + lang

with PdfPages('results_feb_7_2019.pdf') as pdf:

    for test_lang in lang:

        target_a = r'[a-zA-Z]{{3}} [a-zA-Z]{{3}}-{x}-\d{{4}}_\d'.format(x=test_lang)
        target_b = r'[a-zA-Z]{{3}}-{x}-\d{{4}}_\d'.format(x=test_lang)

        root = "./results"
        paths = [os.path.join(root,path) for path in os.listdir(root) if (re.match(target_a, path) or re.match(target_b, path)) and (not path.endswith(".txt"))]

        data = {}
        for tl in train_langs:
            valid_paths = [p for p in paths if tl in p]
            if len(valid_paths) > 0:
                data[tl] = {}
            for path in valid_paths:
                with open(path, "r") as f:
                    x = json.load(f)
                    for k,v in x.items():
                        k = eval(k)
                        if np.all(np.array(list(k)) < 6):
                            data[tl][k] = data[tl].get(k,[]) + [np.mean(v)]



        plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')

        width = 0.1

        for i,(k,v) in enumerate(sorted(data.items())):
            keys, values = zip(*sorted(v.items()))
            x = np.arange(len(values))

            plt.bar(x+(i-4)*width, [np.mean(_) for _ in values], width=width, align='center', label=k, alpha=0.7)

            """
            if test_lang == k:
                plt.bar(x+(i-4)*width, [np.mean(_) for _ in values], width=width, align='center', label=k, color='blue')
            elif test_lang in k:
                plt.bar(x+(i-4)*width, [np.mean(_) for _ in values], width=width, align='center', label=k, color='blue', alpha=0.3)
            else:
                plt.bar(x+(i-4)*width, [np.mean(_) for _ in values], width=width, align='center', label=k, color='lightgray')
            """

        plt.legend()
        plt.title(test_lang)
        plt.xticks(x, ["{}-{}".format(*k) for k in keys])
        #plt.show()
        pdf.savefig()



#
