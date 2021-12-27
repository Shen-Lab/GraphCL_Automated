import sys
import numpy as np


fn = str(sys.argv[1])
with open(fn, 'r') as f:
    data = f.read().split('\n')[:-1]

val_res = [float(d.split()[0])*100 for d in data]
test_res = [float(d.split()[1])*100 for d in data]


print('val', np.mean(val_res), np.std(val_res))
print('test', np.mean(test_res), np.std(test_res))
