'''
    Python Libraries Import
'''
import shap
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from sklearn.linear_model import ElasticNet
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, explained_variance_score
from sklearn.inspection import permutation_importance
from shap import TreeExplainer, LinearExplainer



'''
    Variable & Scope Initilization of the Analyses
'''
filenames = ['baseline_smri', 'baseline_fmri', 'baseline_covr',
             'baseline_smri_covr', 'baseline_fmri_covr', 'baseline_smri_fmri_covr','1yr_smri', '1yr_fmri', '1yr_covr',
             '1yr_smri_covr', '1yr_fmri_covr', '1yr_smri_fmri_covr']

model_names  = ['en', 'hg']
model_list   = {'en': ElasticNet(max_iter=1e7),
                'hg': HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=50, scoring='neg_mean_absolute_error')}

model_grids  = {'en': {'l1_ratio': [.1, .3, .5, .7, .9, .95, .99, 1],
                      'alpha': np.logspace(-3, 3, 7)},
                'hg': {'max_depth': [10, 20, 30],
                       'l2_regularization': [0, 0.25, 0.5, 0.75, 1],
                       'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01]}
               }

cv = KFold(n_splits=10)
scoring = {'mnae': 'neg_mean_absolute_error', 'mdae': 'neg_median_absolute_error', 'rsqe': 'r2', 'evar': 'explained_variance'}
per_strategy_best_models = {}



'''
    Code to carry out regression-based analysis, compute variable importance using SHAP, and save results to a specified path
'''
def execute(X, y, model_name):
    model = model_list[model_name]
    grid  = model_grids[model_name]
    search = GridSearchCV(model, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv, refit=True, verbose=0)
    search.fit(X,y)
    best_estimator = search.best_estimator_
    return best_estimator

def find_shapely(model, X_train, y_train, X_test, model_name, filename):
    if(model_name=='en'):
        model = model.fit(X_train, y_train)
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=10, show=False)
        plt.title('Elastic Net - ' + filename)
        plt.savefig('../results/'+model_name+'/'+filename+"/shapely_fexp.png", dpi=600)
        plt.close()

        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
        plt.title('Elastic Net - ' + filename)
        plt.xlabel('mean(|SHAP value|)')
        plt.savefig('../results/'+model_name+'/'+filename+"/shapely_fimp.png", dpi=600)
        plt.close()

    elif(model_name=='hg'):
        model = model.fit(X_train, y_train)
        explainer = TreeExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=10, show=False)
        plt.title('Histogram Gradient Boosted Trees - ' + filename)
        plt.savefig('../results/'+model_name+'/'+filename+"/shapely_fexp.png", dpi=600)
        plt.close()

        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
        plt.title('Histogram Gradient Boosted Trees - ' + filename)
        plt.xlabel('mean(|SHAP value|)')
        plt.savefig('../results/'+model_name+'/'+filename+"/shapely_fimp.png", dpi=600)
        plt.close()

for model_name in model_names:
    performance = pd.DataFrame(columns=filenames)
    perm_imp = {}
    best_model_ = {}

    for filename in tqdm(filenames):
        df = pd.read_pickle('../data/'+filename)
        X = df.drop('cbcl_scr_dsm5_depress_t', axis=1)
        y = df['cbcl_scr_dsm5_depress_t']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
        best_model = execute(X_train, y_train, model_name)
        best_model_[filename] = best_model
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        performance.loc['held-out mean absolute error', filename]   = '%.3f'%mean_absolute_error(y_test, y_pred)
        performance.loc['held-out median absolute error', filename] = '%.3f'%median_absolute_error(y_test, y_pred)
        performance.loc['held-out r2 score', filename]              = '%.3E'%r2_score(y_test, y_pred)
        performance.loc['held-out explained variance', filename]    = '%.3E'%explained_variance_score(y_test, y_pred)

        results = cross_validate(best_model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        performance.loc['test mean absolute error', filename] =  '%.3f ± %.3f'%(-results['test_mnae'].mean(), results['test_mnae'].std())
        performance.loc['test median absolute error', filename] =  '%.3f ± %.3f'%(-results['test_mdae'].mean(), results['test_mdae'].std())
        performance.loc['test r2 score', filename] =  '%.3E ± %.3E'%(results['test_rsqe'].mean(), results['test_rsqe'].std())
        performance.loc['test explained variance', filename] =  '%.3E ± %.3E'%(results['test_evar'].mean(), results['test_evar'].std())

        performance.loc['train mean absolute error', filename] =  '%.3f ± %.3f'%(-results['train_mnae'].mean(), results['train_mnae'].std())
        performance.loc['train median absolute error', filename] =  '%.3f ± %.3f'%(-results['train_mdae'].mean(), results['train_mdae'].std())
        performance.loc['train r2 score', filename] =  '%.3E ± %.3E'%(results['train_rsqe'].mean(), results['train_rsqe'].std())
        performance.loc['train explained variance', filename] =  '%.3E ± %.3E'%(results['train_evar'].mean(), results['train_evar'].std())

        importances_mean = pd.DataFrame(columns=X.columns)
        importances_std = pd.DataFrame(columns=X.columns)

        find_shapely(best_model, X_train, y_train, X_test, model_name, filename)

    performance.sort_index(ascending=True).to_excel('../results/'+model_name+'/performance.xlsx')
    per_strategy_best_models[model_name] = best_model_

    print('--- %s ---'%(model_name))

pd.DataFrame(per_strategy_best_models).to_pickle('../results/per_strategy_best_models')
pd.DataFrame(per_strategy_best_models).to_excel('../results/per_strategy_best_models.xlsx')

df = pd.read_pickle('../results/per_strategy_best_models')
for filename in tqdm(filenames):
    item = df.loc[filename, :]
    dat = pd.read_pickle('../data/'+filename)
    X = dat.drop('cbcl_scr_dsm5_depress_t', axis=1)
    y = dat['cbcl_scr_dsm5_depress_t']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
    find_shapely(item['en'], X_train, y_train, X_test, 'en', filename)
    find_shapely(item['hg'], X_train, y_train, X_test, 'hg', filename)
