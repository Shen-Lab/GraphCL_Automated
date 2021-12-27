import sys
import numpy as np


fileName = str(sys.argv[1])


with open(fileName, 'r') as f:
    data = f.read().split('\n')[:-1]
    

acc_val_dict, acc_test_dict = {}, {}
for d in data:
    d = d.split()
    dsName, acc_val, acc_test = d[0], float(d[4]), float(d[5])
    if not dsName in acc_val_dict.keys():
        acc_val_dict[dsName] = [acc_val]
        acc_test_dict[dsName] = [acc_test]
    else:
        acc_val_dict[dsName].append(acc_val)
        acc_test_dict[dsName].append(acc_test)


for dsName in acc_val_dict.keys():
    print(dsName, np.mean(acc_val_dict[dsName]), np.std(acc_val_dict[dsName]), np.mean(acc_test_dict[dsName]), np.std(acc_test_dict[dsName]))

