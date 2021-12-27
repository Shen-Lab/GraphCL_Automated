import numpy as np
import sys


file_name = str(sys.argv[1])

with open(file_name, 'r') as f:
    data = f.read().split('\n')[:-1]

res_dict = {}
for d in data:
    pref, res = d.split()
    pref, res = pref[:-2], float(res)

    if not pref in res_dict.keys():
        res_dict[pref] = [res]
    else:
        res_dict[pref].append(res)

pref_best, res_best = '', 0
for k, v in res_dict.items():
    if not '0.001' in k:
        continue
    if len(v) < 5:
        continue
    elif np.mean(v) < np.mean(res_best):
        continue

    pref_best, res_best = k, v

print(pref_best, np.mean(res_best), np.std(res_best))

