import numpy as np
import pandas as pd
import math
from IPython.display import display
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from sklearn.externals import joblib
import scipy as sp
from scipy.optimize import *

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('train_ZoGVYWq.csv')
test = pd.read_csv('test_66516Ee.csv')

#integer encode
label_encoder = LabelEncoder()
#For training data
sourcing_encoded_train = label_encoder.fit_transform(train.sourcing_channel)
res_encoded_train = label_encoder.fit_transform(train.residence_area_type)
#For test data
sourcing_encoded_test = label_encoder.fit_transform(test.sourcing_channel)
res_encoded_test = label_encoder.fit_transform(test.residence_area_type)

#binary encode
onehot_encoder = OneHotEncoder(sparse=False)
#reshaping
sourcing_encoded_train = sourcing_encoded_train.reshape(len(sourcing_encoded_train), 1)
res_encoded_train = res_encoded_train.reshape(len(res_encoded_train), 1)
sourcing_encoded_test = sourcing_encoded_test.reshape(len(sourcing_encoded_test), 1)
res_encoded_test = res_encoded_test.reshape(len(res_encoded_test), 1)
#one-hot encoding
sourcing_onehot_train = onehot_encoder.fit_transform(sourcing_encoded_train)
res_onehot_train = onehot_encoder.fit_transform(res_encoded_train)
sourcing_onehot_test = onehot_encoder.fit_transform(sourcing_encoded_test)
res_onehot_test = onehot_encoder.fit_transform(res_encoded_test)

#Convert to pandas dataframe
sourcing_df_train = pd.DataFrame(sourcing_onehot_train, columns = ['sourcing' + str(i) for i in range(sourcing_onehot_train.shape[1])])
res_df_train = pd.DataFrame(res_onehot_train, columns = ['res' + str(i) for i in range(res_onehot_train.shape[1])])
sourcing_df_test = pd.DataFrame(sourcing_onehot_test, columns = ['sourcing' + str(i) for i in range(sourcing_onehot_test.shape[1])])
res_df_test = pd.DataFrame(res_onehot_test, columns = ['res' + str(i) for i in range(res_onehot_test.shape[1])])

#Concatenating one-hot encodings and dropping categorical columns
train_concat = pd.concat([train, sourcing_df_train, res_df_train], axis=1)
test_concat = pd.concat([test, sourcing_df_test, res_df_test], axis=1)
train_concat = train_concat.drop(columns = ['sourcing_channel', 'residence_area_type'])
test_concat = test_concat.drop(columns = ['sourcing_channel', 'residence_area_type'])

#Imputing mean
mean_aus = np.mean(train_concat.application_underwriting_score)
train_concat.application_underwriting_score = train_concat.application_underwriting_score.fillna(mean_aus)
test_concat.application_underwriting_score = test_concat.application_underwriting_score.fillna(mean_aus)

#Imputing median
imp_med = Imputer(strategy='median')
train_columns = train_concat.columns
test_columns = test_concat.columns
train_prepoc = imp_med.fit_transform(train_concat)
test_prepoc = imp_med.fit_transform(test_concat)

train_prepoc = pd.DataFrame(train_prepoc, columns=train_columns)
test_prepoc = pd.DataFrame(test_prepoc, columns=test_columns)

train_prepoc = train_prepoc.set_index('id')
test_prepoc = test_prepoc.set_index('id')

X = train_prepoc.loc[:, filter(lambda x: x not in ['renewal'], train_prepoc.columns)]
y = train_prepoc.loc[:,['renewal']]
X_test = test_prepoc

#Converting y to 1D array
y = y.squeeze()
#Reducing learning rate and increasing esimators
GB_model_tuned_final = GradientBoostingClassifier(learning_rate=0.001,n_estimators=2000,max_depth=9,
                                            min_samples_split=600,min_samples_leaf=220,
                                            max_features='sqrt',random_state=10, subsample=0.8)
GB_model_tuned_final.fit(X, y)
#joblib.dump(GB_model_tuned_final, "pima_6_6.joblib.dat")

#using final model to predict on test set
test_predictions = GB_model_tuned_final.predict(X_test)
test_predprob = GB_model_tuned_final.predict_proba(X_test)[:,1]
X_test = pd.concat([X_test, pd.DataFrame(test_predprob, columns=['prob_bench']).set_index(X_test.index.values)], axis=1)

def net_total_rev_test(e, i):
    df = X_test.iloc[i:i+4,:]
    p = df.prob_bench
    pr = df.premium
    r = np.array([])
    def net_rev(p, pr, e):
        return(p*pr*0.04*math.exp(-e/5) - 400/(10-e))
    for i in range(len(df)):
        r = np.append(r,net_rev(p.iloc[i], pr.iloc[i], e[i]))
    return(r)
incent_test = np.array([])
for i in range(0,X_test.shape[0], 4):
    guess = 5
    x0_test = np.array([guess]*4)
    sol_test = root(net_total_rev_test, x0_test, args=i, method = 'lm')
#print('effort=5 :', sol_test.x, sol_test.success, sol_test.nfev, np.sum(net_total_rev_test(sol_test.x)))
    incent_test_out = -400*np.log(1-sol_test.x/10)
    incent_test_out[incent_test_out<0] = 0
    incent_test = np.append(incent_test, incent_test_out)

incent_df = pd.DataFrame(incent_test, index=X_test.index, columns=['incentive'])
X_test = pd.concat([X_test, incent_df], axis=1)

output = pd.DataFrame({'id':X_test.index.astype('int32'), 'renewal':np.array(X_test.prob_bench), 'incentives':np.array(X_test.incentive)})
output.reindex(columns=['id','renewal','incentives']).set_index('id').to_csv('GBM_final.csv', sep=",")
