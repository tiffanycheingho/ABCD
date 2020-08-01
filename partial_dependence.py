'''
    Stub to generate partial dependence plots for the best ML models
'''
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.linear_model import ElasticNet
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
from IPython.display import display
from pylab import rcParams
rcParams['figure.figsize'] = 40, 20
rcParams['font.size'] = 24
rcParams['axes.grid'] = True

path =  '../ABCD 2.0/ABCDstudyNDA_RELEASE2/'
read_file = lambda file_name: pd.read_csv(file_name, delimiter='\t', skiprows=[1])

dependent = 'cbcl_scr_dsm5_depress_t'
dep_cov = 'asr_scr_anxdep_t'
file = 'abcd_cbcls01.txt'
features = ['subjectkey',dependent]
depression = pd.read_csv(path + file, delimiter='\t', skiprows=[1], low_memory=False)
depression = depression[depression['eventname'] == 'baseline_year_1_arm_1']
depression = depression[features].dropna()
cov = read_file(path + 'abcd_asrs01.txt')[['subjectkey', dep_cov]].dropna()
struct = pd.read_pickle('../data/structural').reset_index()
func = pd.read_pickle('../data/functional').reset_index()
func.drop('gender', axis=1, inplace=True)
covariates = pd.read_pickle('../data/covariates').reset_index()
cov = cov.set_index('subjectkey').reset_index()
covariates = covariates.merge(cov, how='inner', on='subjectkey', validate='1:1')
index = pd.read_pickle('../data/no_twins_triplets')
data = struct.merge(depression, how='inner', on='subjectkey', validate='1:1')
data = data.merge(func, how='inner', on='subjectkey', validate='1:1')
data = data.merge(covariates, how='inner', on='subjectkey', validate='1:1')
data = data.merge(index, how='inner', on='subjectkey', validate='1:1').set_index('subjectkey')
X = data.drop(dependent, axis=1)
for col in list(X.dtypes[X.dtypes == 'category'].keys()):
    X[col] = X[col].factorize()[0]
y = data[dependent]

clf = ElasticNet(l1_ratio=1.0, alpha=0.1233, normalize=False, max_iter=10**7).fit(X,y)
# ydata, xdata = partial_dependence(clf, features=['gender'], X=X, percentiles=(0, 1), grid_resolution=100)
plot_partial_dependence(clf, X, ['rsfmri_c_ngd_cgc_ngd_sa', 'asr_scr_anxdep_t', 'sleepdisturb1_p'], grid_resolution=1000)
plt.savefig('elastic_net_partial_dependence.png')
plt.show()

clf = HistGradientBoostingRegressor(l2_regularization=0, learning_rate=0.05, max_depth=30, max_leaf_nodes=50, max_iter=10**3).fit(X,y)
# ydata, xdata = partial_dependence(clf, features=['gender'], X=X, percentiles=(0, 1), grid_resolution=100)
plot_partial_dependence(clf, X, features=['rsfmri_c_ngd_ad_ngd_sa', 'rsfmri_cor_ngd_au_scs_aarh', 'rsfmri_cor_ngd_df_scs_aarh', 'asr_scr_anxdep_t', 'sleepdisturb1_p'], grid_resolution=1000, n_jobs=-1)
plt.savefig('light_gbt_partial_dependence.png')
plt.show()
