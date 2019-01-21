# results grabber

import os
from subprocess import call
import pandas as pd

dir = "./results"

results = []

for path in os.listdir(dir):
    fullpath = os.path.join(dir, path)
    if "1234" in path:
        train_lang = path.split("-")[0].split()
        test_lang = path.split("-")[1]
        with open(fullpath, "r") as f:
            score = float(list(f.readlines())[-1].strip("\n"))
            results.append([train_lang, test_lang, score])


d = pd.DataFrame(data=results, columns=["train_lang", "test_lang", "score"])
d.to_csv("results.csv")
