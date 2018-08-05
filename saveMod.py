import numpy as np
import os
import pandas as pd
import dill
from datetime import timedelta
from dateutil.parser import parser
import os
os.chdir(os.path.expanduser('~/Documents/Abhijit_epidemicModel/notebooks/'))
#from csv_pkl_sql import save_it, csv_it, pkl_it
os.chdir('..')

import matplotlib.pyplot as plt
import seaborn as sns

with open('./pkl/11_features_engineered.pkl', 'rb') as fh:
    features = dill.load(fh)


from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import auc, accuracy_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score

for col in features.columns:
    if col not in ['date', 'location']:
        features[col] = features[col].astype(np.float)
feat_cols = [x for x in features.columns if x not in ['date','location']]
features[feat_cols] = Normalizer().fit_transform(features[feat_cols])

framework_a_max   = pd.read_pickle('./pkl/10_class_balancing_framework_a_max.pkl')



fwd_a_max = pd.merge(framework_a_max, 
                       features, 
                       on=['date','location'], how='left').dropna()

X = fwd_a_max[feat_cols].values
y = fwd_a_max['zika_bool'].values
y=y.astype('int32')
with open('./pkl/X', 'wb') as fh:
    dill.dump(X, fh)
with open('./pkl/y', 'wb') as fh:
    dill.dump(y,fh)


