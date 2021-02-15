'''
    Python Libraries Import
'''
import numpy as np
import pandas as pd
from IPython.display import display as printdf

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import robust_scale
from scipy.stats import zscore
from tqdm import tqdm



'''
    Stub to read ABCD data from a specified path
'''
path = "../ABCD 3.0/"
to_path = '../data/'
read_file = lambda file_name: pd.read_csv(file_name, delimiter='\t', skiprows=[1], index_col='subjectkey', low_memory=False)
baseline = lambda df: df[df['eventname']=='baseline_year_1_arm_1']
followup = lambda df: df[df['eventname']=='1_year_follow_up_y_arm_1']



'''
    Code to extract and pre-process Structural MRI ABCD 3.0 Data
'''
smrip101 = baseline(read_file(path+'abcd_smrip101.txt'))
a = pd.Series(smrip101.columns)
cols = a[a.str.contains('smri_vol_cdk_')].values
smrip101 = smrip101[cols]
print('smrip101 shape:', smrip101.shape)

smrip201 = baseline(read_file(path+'abcd_smrip201.txt'))
a = pd.Series(smrip201.columns)
cols = a[a.str.contains('smri_vol_scs_')].values
smrip201 = smrip201[cols]
print('smrip201 shape:', smrip201.shape)

fsqc = baseline(read_file(path + 'freesqc01.txt'))[['fsqc_qc']]
print('fsqc shape:', fsqc.shape)

d1 = smrip101.merge(smrip201, on='subjectkey', how='outer', validate='1:1')
print('structural union shape:', d1.shape)
d2 = d1.merge(fsqc, on='subjectkey', how='inner', validate='1:1')
d2['fsqc_qc'] = d2['fsqc_qc'].fillna(0)
structural = d2.drop(['smri_vol_scs_lesionlh', 'smri_vol_scs_lesionrh', 'smri_vol_scs_wmhintlh', 'smri_vol_scs_wmhintrh'], axis=1)
print('final shape:', structural.shape)

structural = structural[structural['fsqc_qc'] == 1].drop(['fsqc_qc'], axis=1)
print('structural post freesurfer qc shape:', structural.shape)

structural = structural.astype('float')
imputer = SimpleImputer(strategy='mean')
structural[:] = imputer.fit_transform(structural)
structural[:] = robust_scale(structural)
print('preprocessed structural shape:', structural.shape)



'''
    Code to extract and pre-process Functional MRI ABCD 3.0 Data
'''
mrirstv02 = baseline(read_file(path+'abcd_mrirstv02.txt'))[['rsfmri_var_meanmotion', 'rsfmri_var_ntpoints']]
print('mrirstv02 shape:', mrirstv02.shape)

betnet02 = baseline(read_file(path+'abcd_betnet02.txt')).iloc[:,21:-2]
features = []
for col in betnet02.columns:
    elements = col.split('_')
    if 'n' in elements:
        continue
    else:
        features.append(col)
betnet02 = betnet02[features]
print('betnet02 shape:', betnet02.shape)

subset = ['aglh', 'agrh', 'hplh', 'hprh', 'aalh', 'aarh', 'ptlh', 'ptrh', 'cdelh', 'cderh']
mrirscor02 = baseline(read_file(path + 'mrirscor02.txt'))
features = []
for col in list(mrirscor02.columns):
    elements = col.split('_')
    if('none' in elements):
        continue
    for region in subset:
        if(region in elements):
            features.append(col)
mrirscor02 = mrirscor02[features]

d1 = mrirstv02.merge(betnet02, on='subjectkey', how='outer', validate='1:1')
d2 = d1.merge(mrirscor02, on='subjectkey', how='outer', validate='1:1')
print('functional union shape:', d2.shape)
d3 = d2.merge(fsqc, on='subjectkey', how='inner', validate='1:1')
d3['fsqc_qc'] = d3['fsqc_qc'].fillna(0)
functional = d3
print('final shape:', functional.shape)

functional = functional[functional['fsqc_qc'] == 1].drop(['fsqc_qc'], axis=1)
print('functional post freesurfer qc shape:', functional.shape)

exclude_subjects = set()
fmriqc01 = baseline(read_file(path+'fmriqc01.txt'))[['fmri_postqc_b0warp', 'fmri_postqc_imgqual', 'fmri_postqc_cutoff']]
SK = set(fmriqc01.index.values)
imputer = SimpleImputer(strategy='constant')
fmriqc01[:] = imputer.fit_transform(fmriqc01)
sk = set(fmriqc01[(fmriqc01['fmri_postqc_b0warp']<=1.5) & (fmriqc01['fmri_postqc_imgqual']<=1.5) & (fmriqc01['fmri_postqc_cutoff']<=1.5)].index.values)
excluded_subjects = SK - sk
exclude_subjects = exclude_subjects.union(excluded_subjects)

mrirstv02 = baseline(read_file(path+'abcd_mrirstv02.txt'))[['rsfmri_var_meanmotion', 'rsfmri_var_ntpoints']]
SK = set(mrirstv02.index.values)
sk = set(mrirstv02[mrirstv02['rsfmri_var_ntpoints']>375].index.values)
excluded_subjects = SK - sk
exclude_subjects = exclude_subjects.union(excluded_subjects)

indexes_to_keep = list(set(functional.index.values) - exclude_subjects)
functional = functional.loc[indexes_to_keep]
print('functional post other qc filtering shape:', functional.shape)

functional = functional.astype('float')
imputer = SimpleImputer(strategy='mean')
functional[:] = imputer.fit_transform(functional)
functional[:] = robust_scale(functional)
print('preprocessed functional shape:', functional.shape)



'''
    Code to extract and pre-process Non-Brain ABCD 3.0 Data
'''
ant01 = baseline(read_file(path+'abcd_ant01.txt'))[['anthroweightcalc','anthroheightcalc']]
m = ant01['anthroweightcalc']
h2 = ant01['anthroheightcalc']**2
ant01['bmi'] = (m/h2)*703
ant01 = ant01[['bmi']]
print('ant01 shape:', ant01.shape)

ppdms01 = baseline(read_file(path+'abcd_ppdms01.txt'))
ppdms01 = ppdms01.replace(999.0, 0)
ppdms01 = ppdms01.replace(np.nan, 0)
ppdms01['pubertal_score'] = ppdms01.apply(lambda x : x['pds_1_p'] + x['pds_2_p'] + x['pds_3_p'] + x['pds_m4_p'] + x['pds_m5_p']  if (x['pubertal_sex_p']==1.0) else x['pds_1_p'] + x['pds_2_p'] + x['pds_3_p'] + x['pds_f4_p'] + x['pds_f5b_p'], axis=1, result_type='reduce')
ppdms01 = ppdms01[['pubertal_score']].astype('int')
ppdms01 = ppdms01.replace(0, np.nan)
print('ppdms01 shape:', ppdms01.shape)

medsy01 = baseline(read_file(path+'medsy01.txt'))
a = pd.Series(medsy01.columns)
cols = a[a.str.contains('_24')].values
medsy01 = medsy01[cols]
medsy01 = medsy01.replace(999.0, 0)
medsy01 = medsy01.fillna(0)
f = lambda x: 1 if x>=1 else 0
cols = medsy01.columns
a = pd.Series(medsy01.columns)
rx_cols = a[a.str.contains('rx')].values
otc_cols = a[a.str.contains('otc')].values
medsy01['rx_24']=medsy01[rx_cols].sum(axis=1).apply(f)
medsy01['otc_24']=medsy01[otc_cols].sum(axis=1).apply(f)
medsy01['caff_24'] = medsy01['caff_24'].astype(int)
medsy01 = medsy01[['rx_24', 'otc_24', 'caff_24']]
print('medsy01 shape:', medsy01.shape)

pdem02 = baseline(read_file(path+'pdem02.txt'))
cols = ['demo_brthdat_v2','demo_ed_v2',
        'demo_race_a_p___10','demo_race_a_p___11','demo_race_a_p___12',
        'demo_race_a_p___13','demo_race_a_p___14','demo_race_a_p___15',
        'demo_race_a_p___16','demo_race_a_p___17','demo_race_a_p___18',
        'demo_race_a_p___19','demo_race_a_p___20','demo_race_a_p___21',
        'demo_race_a_p___22','demo_race_a_p___23',
        'demo_prnt_marital_v2','demo_prnt_ed_v2','demo_prnt_income_v2',
        'demo_prnt_prtnr_v2','demo_prtnr_ed_v2','demo_comb_income_v2']
pdem02['race_white'] = pdem02['demo_race_a_p___10']
pdem02['race_mixed'] = pdem02[['demo_race_a_p___11','demo_race_a_p___12','demo_race_a_p___13',
                               'demo_race_a_p___14','demo_race_a_p___15','demo_race_a_p___16',
                               'demo_race_a_p___17','demo_race_a_p___18','demo_race_a_p___19',
                               'demo_race_a_p___20','demo_race_a_p___21','demo_race_a_p___22',
                               'demo_race_a_p___23']].sum(axis=1)
pdem02['race_mixed'] = pdem02['race_mixed'].apply(f)
pdem02['demo_prnt_ed_v2'] = pdem02['demo_prnt_ed_v2'].replace(999, 0)
pdem02['demo_prnt_ed_v2'] = pdem02['demo_prnt_ed_v2'].replace(777, 0)
pdem02['demo_prnt_ed_v2'] = pdem02['demo_prnt_ed_v2'].replace(np.nan, 0)
pdem02['demo_prtnr_ed_v2'] = pdem02['demo_prtnr_ed_v2'].replace(999, 0)
pdem02['demo_prtnr_ed_v2'] = pdem02['demo_prtnr_ed_v2'].replace(777, 0)
pdem02['demo_prtnr_ed_v2'] = pdem02['demo_prtnr_ed_v2'].replace(np.nan, 0)
pdem02['parent_edu_max'] = pdem02[['demo_prnt_ed_v2','demo_prtnr_ed_v2']].max(axis=1)
pdem02['parent_edu_max'] = pdem02['parent_edu_max'].replace(0, np.nan)
pdem02 = pdem02[['demo_brthdat_v2','demo_ed_v2','race_white','race_mixed',
                 'demo_prnt_marital_v2','parent_edu_max','demo_prnt_prtnr_v2',
                 'demo_comb_income_v2']]
pdem02 = pdem02.replace(999.0, np.nan)
pdem02 = pdem02.replace(777.0, np.nan)
print('pdem02 shape:', pdem02.shape)

sds01 = baseline(read_file(path+'abcd_sds01.txt'))[['sleepdisturb1_p']]
sds01 = sds01.reset_index().drop_duplicates().set_index('subjectkey')
print('sds01 shape:', sds01.shape)

stq01 = baseline(read_file(path+'stq01.txt'))[['screentime2_p_hours']]
print('stq01 shape:', stq01.shape)

fes02 = baseline(read_file(path+'fes02.txt'))[['fam_enviro1_p','fam_enviro2r_p', 'fam_enviro3_p',
                                                'fam_enviro4r_p','fam_enviro5_p', 'fam_enviro6_p',
                                                'fam_enviro7r_p','fam_enviro8_p', 'fam_enviro9r_p']]
fes02['fam_enviro_sum'] = fes02.sum(axis=1)
fes02 = fes02[['fam_enviro_sum']].astype('int')
fes02 = fes02.reset_index().drop_duplicates().set_index('subjectkey')
print('fes02 shape:', fes02.shape)

lt01 = baseline(read_file(path + 'abcd_lt01.txt'))[['sex', 'site_id_l']]
f = lambda x: 1 if x=='F' else x
m = lambda x: 0 if x=='M' else x
s = lambda x: int(x.split('site')[1])
lt01['sex'] = lt01['sex'].apply(m).apply(f)
lt01['site_id_l'] = lt01['site_id_l'].apply(s)
print('lt01 shape:', lt01.shape)

asrs01 = baseline(read_file(path+'abcd_asrs01.txt'))[['asr_scr_anxdep_t']]
print('asrs01 shape:', asrs01.shape)

d1 = ant01.merge(ppdms01, on='subjectkey', how='outer', validate='1:1')
d2 = d1.merge(medsy01, on='subjectkey', how='outer', validate='1:1')
d3 = d2.merge(pdem02, on='subjectkey', how='outer', validate='1:1')
d4 = d3.merge(sds01, on='subjectkey', how='outer', validate='1:1')
d5 = d4.merge(stq01, on='subjectkey', how='outer', validate='1:1')
d6 = d5.merge(lt01, on='subjectkey', how='outer', validate='1:1')
d7 = d6.merge(asrs01, on='subjectkey', how='outer', validate='1:1')
covariates = d7.merge(fes02, on='subjectkey', how='outer', validate='1:1')
print('final shape:', covariates.shape)

covariates = covariates.astype('float')
imputer = SimpleImputer(strategy='median')
covariates[:] = imputer.fit_transform(covariates)
covariates[:] = robust_scale(covariates)
print('preprocessed covariates shape:', covariates.shape)



'''
    Code to remove twins & triplets from the analyses
'''
acspsw03 = baseline(read_file(path+'acspsw03.txt'))[['rel_family_id', 'rel_group_id']]
use_keys = []
unique_family = acspsw03['rel_family_id'].unique()
for uf in tqdm(unique_family):
    use_keys = use_keys + list(acspsw03[acspsw03['rel_family_id']==uf].reset_index().groupby(['rel_group_id']).min()['subjectkey'])
print("# subjects (with no twins or triplets):",len(use_keys))

baseline_smri = structural.copy(deep=True)
baseline_fmri = functional.copy(deep=True)
baseline_covr = covariates.copy(deep=True)
baseline_smri_covr = structural.merge(covariates, on='subjectkey', how='inner', validate='1:1')
baseline_fmri_covr = functional.merge(covariates, on='subjectkey', how='inner', validate='1:1')
baseline_smri_fmri_covr = covariates.merge(structural, on='subjectkey', how='inner', validate='1:1')
baseline_smri_fmri_covr = baseline_smri_fmri_covr.merge(functional, on='subjectkey', how='inner', validate='1:1')

print('--- shapes with twins and triplets ---\n')
print('baseline_smri shape           :', baseline_smri.shape)
print('baseline_fmri shape           :', baseline_fmri.shape)
print('baseline_covr shape           :', baseline_covr.shape)
print('baseline_smri_cov shape       :', baseline_smri_covr.shape)
print('baseline_fmri_covr shape      :', baseline_fmri_covr.shape)
print('baseline_smri_fmri_covr shape :', baseline_smri_fmri_covr.shape)

baseline_smri = baseline_smri.loc[baseline_smri.index.intersection(use_keys), :]; baseline_smri.index.name = 'subjectkey'
baseline_fmri = baseline_fmri.loc[baseline_fmri.index.intersection(use_keys), :]; baseline_fmri.index.name = 'subjectkey'
baseline_covr = baseline_covr.loc[baseline_covr.index.intersection(use_keys), :]; baseline_covr.index.name = 'subjectkey'
baseline_smri_covr = baseline_smri_covr.loc[baseline_smri_covr.index.intersection(use_keys), :]; baseline_smri_covr.index.name = 'subjectkey'
baseline_fmri_covr = baseline_fmri_covr.loc[baseline_fmri_covr.index.intersection(use_keys), :]; baseline_fmri_covr.index.name = 'subjectkey'
baseline_smri_fmri_covr = baseline_smri_fmri_covr.loc[baseline_smri_fmri_covr.index.intersection(use_keys), :]; baseline_smri_fmri_covr.index.name = 'subjectkey'

print('--- shapes without twins and triplets ---\n')
print('baseline_smri shape           :', baseline_smri.shape)
print('baseline_fmri shape           :', baseline_fmri.shape)
print('baseline_covr shape           :', baseline_covr.shape)
print('baseline_smri_cov shape       :', baseline_smri_covr.shape)
print('baseline_fmri_covr shape      :', baseline_fmri_covr.shape)
print('baseline_smri_fmri_covr shape :', baseline_smri_fmri_covr.shape)



'''
    Preparing & exporting combined "Baseline" data for ML-based analysis to a specified path
'''
dependent = baseline(read_file(path+'abcd_cbcls01.txt'))[['cbcl_scr_dsm5_depress_t']].dropna()

baseline_smri = baseline_smri.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_fmri = baseline_fmri.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_covr = baseline_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_smri_covr = baseline_smri_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_fmri_covr = baseline_fmri_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_smri_fmri_covr = baseline_smri_fmri_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')

print('--- shapes with dependent variable ---\n')
print('baseline_smri shape           :', baseline_smri.shape)
print('baseline_fmri shape           :', baseline_fmri.shape)
print('baseline_covr shape           :', baseline_covr.shape)
print('baseline_smri_cov shape       :', baseline_smri_covr.shape)
print('baseline_fmri_covr shape      :', baseline_fmri_covr.shape)
print('baseline_smri_fmri_covr shape :', baseline_smri_fmri_covr.shape)

baseline_smri.to_pickle(to_path+'baseline_smri')
baseline_fmri.to_pickle(to_path+'baseline_fmri')
baseline_covr.to_pickle(to_path+'baseline_covr')
baseline_smri_covr.to_pickle(to_path+'baseline_smri_covr')
baseline_fmri_covr.to_pickle(to_path+'baseline_fmri_covr')
baseline_smri_fmri_covr.to_pickle(to_path+'baseline_smri_fmri_covr')



'''
    Preparing & exporting combined "Follow-up" data for ML-based analysis to a specified path
'''
dependent = followup(read_file(path+'abcd_cbcls01.txt'))[['cbcl_scr_dsm5_depress_t']].dropna()

baseline_smri = baseline_smri.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_fmri = baseline_fmri.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_covr = baseline_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_smri_covr = baseline_smri_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_fmri_covr = baseline_fmri_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')
baseline_smri_fmri_covr = baseline_smri_fmri_covr.merge(dependent, on='subjectkey', how='inner', validate='1:1')

print('--- shapes with dependent variable ---\n')
print('baseline_smri shape           :', baseline_smri.shape)
print('baseline_fmri shape           :', baseline_fmri.shape)
print('baseline_covr shape           :', baseline_covr.shape)
print('baseline_smri_cov shape       :', baseline_smri_covr.shape)
print('baseline_fmri_covr shape      :', baseline_fmri_covr.shape)
print('baseline_smri_fmri_covr shape :', baseline_smri_fmri_covr.shape)

baseline_smri.to_pickle(to_path+'1yr_smri')
baseline_fmri.to_pickle(to_path+'1yr_fmri')
baseline_covr.to_pickle(to_path+'1yr_covr')
baseline_smri_covr.to_pickle(to_path+'1yr_smri_covr')
baseline_fmri_covr.to_pickle(to_path+'1yr_fmri_covr')
baseline_smri_fmri_covr.to_pickle(to_path+'1yr_smri_fmri_covr')
