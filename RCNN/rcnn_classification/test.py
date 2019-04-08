import numpy as np
'''import rcnn_classification.evaluateCluster as eva
import scipy.io as sio
fea_LE_bin = sio.loadmat('../data/fea1234567.mat')
all_label = np.load('../data/labels_all_1234567.npy')
fea_LE_bin = fea_LE_bin['Y']
all_label = all_label.astype(int)
all_label = all_label.reshape(all_label.shape[0])

eva.evaluateCluster(fea_LE_bin, all_label)'''

'''all = np.load('../data/all_1234567.npy')
all[all==16904] = 0
print(np.max(all))'''
import scipy.io as sio
CR_E = sio.loadmat('../data/CR_E.mat')
CR_E = CR_E['YF']
# CR_E_1 = np.zeros((CR_E.shape[1], CR_E.shape[0]))
CR_E = CR_E.swapaxes(1, 0)
print()