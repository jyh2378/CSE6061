import glob
import random
import datetime
import numpy as np
import pickle
import Pmf_ML
import Gaussian_ML
import Gaussian_MAP
from sklearn.linear_model import LogisticRegression

file_list = glob.glob('data/npy_data/RGB_channel/*_RGB.npy')

print('training start:', datetime.datetime.now().strftime('%H:%M:%S'))
random.shuffle(file_list)
train = np.load(file_list[0], allow_pickle=False, fix_imports=False)
for file in file_list[1:2845]:
    read = np.load(file, allow_pickle=False, fix_imports=False)
    train = np.concatenate((train, read), axis=0)

print('model fitting and save:', datetime.datetime.now().strftime('%H:%M:%S'))

# Logistic Regression
model = LogisticRegression()
model.fit(train[:, :3], train[:, 3])
pickle.dump(model, open('10.Logistic.sav', 'wb'))

# Pmf
model = Pmf_ML.Empirical_Model()
model.fit(train[:, :3], train[:, 3])
pickle.dump(model, open('1.Pmf.sav', 'wb'))

# Gaussian
model = Gaussian_ML.Gaussian_Model()
model.fit(train[:, :3], train[:, 3])
pickle.dump(model, open('2.Gaussian_ML.sav', 'wb'))
  
# Gaussian parameter inferenced by Normal Inverse Wishart
model = Gaussian_MAP.Gaussian_NIW_Model()
model.fit(train[:, :3], train[:, 3])
pickle.dump(model, open('3.Gaussian_MAP.sav', 'wb'))

# Pmf with prior 0.4
model = Pmf_ML.Empirical_Model()
model.fit(train[:, :3], train[:, 3], prior=0.4)
pickle.dump(model, open('4.Pmf_prior.sav', 'wb'))

# Gaussian with prior 0.4
model = Gaussian_ML.Gaussian_Model()
model.fit(train[:, :3], train[:, 3], prior=0.4)
pickle.dump(model, open('5.Gaussian_prior.sav', 'wb'))

# train data with out {0, 255}
train = train[(train[:, 0] != 0) & (train[:, 0] != 255), :]
train = train[(train[:, 1] != 0) & (train[:, 1] != 255), :]
train = train[(train[:, 2] != 0) & (train[:, 2] != 255), :]

# Pmf with out {0, 255}
model = Pmf_ML.Empirical_Model()
model.fit(train[:, :3], train[:, 3])
pickle.dump(model, open('6.Pmf_without_0_255.sav', 'wb'))

# Gaussian with out {0, 255}
model = Gaussian_ML.Gaussian_Model()
model.fit(train[:, :3], train[:, 3])
pickle.dump(model, open('7.Gaussian_Pmf_without_0_255.sav', 'wb'))
