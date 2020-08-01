'''
    Python Libraries Import
'''
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm



'''
    Stub to read ABCD data from a specified path
'''
path =  'ABCD 2.0/ABCDstudyNDA_RELEASE2/'
read_file = lambda file_name: pd.read_csv(file_name, delimiter='\t', skiprows=[1])



'''
    Code to remove twins and triplets from the study
'''
family = read_file(path + 'acspsw03.txt')[['subjectkey', 'rel_family_id', 'rel_group_id']]
use_keys = []
unique_family = family['rel_family_id'].unique()
for uf in unique_family:
    fam = family[family['rel_family_id']==uf]
    fam = fam.groupby(['rel_group_id']).min()
    use_keys = use_keys + list(fam['subjectkey'])
keys = pd.DataFrame(use_keys, columns=['subjectkey'])
keys = pd.DataFrame(use_keys, columns=['subjectkey'])
data = keys.merge(family, how='inner', on='subjectkey', validate='1:1').drop('rel_group_id', axis=1)
data['rel_family_id'] = data['rel_family_id'].astype('category')
fam = data



'''
    Code to extract structural MRI data
'''
smrip101 = read_file(path + 'abcd_smrip101.txt'); smrip101 = smrip101[['subjectkey', 'gender'] + list(smrip101.columns)[331:438]]
smrip201 = read_file(path + 'abcd_smrip201.txt'); smrip201 = smrip201[['subjectkey'] + list(smrip201.columns)[329:375]]
fsqc     = read_file(path + 'freesqc01.txt'); fsqc = fsqc[['subjectkey', 'fsqc_qc']]
data = smrip101.merge(smrip201, how='inner', on='subjectkey', validate='1:1')
data = data.merge(fsqc, how='inner', on='subjectkey', validate='1:1')
data = data.merge(fam, how='inner', on='subjectkey', validate='1:1')
data.drop(['smri_vol_scs_wholeb','smri_vol_scs_latventricles','smri_vol_scs_allventricles'], axis=1, inplace=True)

data = data[data['fsqc_qc'] == 1.0]
data.drop(['fsqc_qc'], axis=1, inplace=True)

gender = data[['subjectkey', 'gender']]
data.drop('gender', axis=1, inplace=True)
data = data.set_index('subjectkey')
data.dropna(axis=1, how='all', inplace=True)
data.dropna(axis=0, how='any', inplace=True)

# data = data.apply(zscore)
data = data.reset_index()
data = data.merge(gender, how='inner', on='subjectkey', validate='1:1')
data = data.set_index('subjectkey')
data.dropna(axis=1, how='all', inplace=True)
data.dropna(axis=0, how='any', inplace=True)
data['gender'] = data['gender'].astype('category')
data.to_pickle('data/structural')


'''
    Code to extract functional MRI data
'''
use_subjects = read_file(path + 'abcd_mri01.txt')
use_subjects = use_subjects[['subjectkey', 'mri_info_manufacturer', 'mri_info_deviceserialnumber']]
use_subjects = use_subjects[use_subjects['mri_info_manufacturer'] != 'Philips Medical Systems']
use_subjects['mri_info_manufacturer'] = use_subjects['mri_info_manufacturer'].astype('category')
use_subjects['mri_info_deviceserialnumber'] = use_subjects['mri_info_deviceserialnumber'].astype('category')
subset = ['aglh', 'agrh', 'hplh', 'hprh', 'aalh', 'aarh', 'ptlh', 'ptrh', 'cdelh', 'cderh']

abcd_betnet02 = read_file(path + 'abcd_betnet02.txt')
abcd_betnet02 = abcd_betnet02[['subjectkey', 'gender', 'rsfmri_c_ngd_meantrans', 'rsfmri_c_ngd_meanrot'] + list(abcd_betnet02.columns)[22:-2]]

mrirscor02 = read_file(path + 'mrirscor02.txt')
features = ['subjectkey']
for col in list(mrirscor02.columns):
    elements = col.split('_')
    for region in subset:
        if(region in elements):
            features.append(col)
mrirscor02 = mrirscor02[features]

data = use_subjects.merge(abcd_betnet02, how='inner', on='subjectkey', validate='1:1')
data = data.merge(mrirscor02, how='inner', on='subjectkey', validate='1:1')
data = data.merge(fsqc, how='inner', on='subjectkey', validate='1:1')
data = data.merge(fam, how='inner', on='subjectkey', validate='1:1')

data = data[data['fsqc_qc'] == 1.0]
data.drop(['fsqc_qc'], axis=1, inplace=True)

catg = data[['subjectkey', 'gender', 'mri_info_manufacturer', 'mri_info_deviceserialnumber']]
data.drop(['gender', 'mri_info_manufacturer', 'mri_info_deviceserialnumber'], axis=1, inplace=True)
data = data.set_index('subjectkey')
data.dropna(axis=1, how='all', inplace=True)
data.dropna(axis=0, how='any', inplace=True)

# data = data.apply(zscore)
data = data.reset_index()
data = data.merge(catg, how='inner', on='subjectkey', validate='1:1')
data = data.set_index('subjectkey')
data['gender'] = data['gender'].astype('category')
data.to_pickle('data/functional')



'''
    Code to extract non-brain data
'''
file = 'abcd_ant01.txt' # BMI Data
features = ['subjectkey','anthroweightcalc','anthroheightcalc']
df = pd.read_csv(path + file, delimiter='\t', skiprows=[1])
df = df[df['eventname'] == 'baseline_year_1_arm_1']
df = df[features].dropna()
df['bmi'] = (df['anthroweightcalc']/(df['anthroheightcalc'])**2)*703
bmi = df[['subjectkey', 'bmi']].set_index('subjectkey').dropna()

file = 'abcd_ppdms01.txt' # Pubertal Data
features = ['subjectkey','eventname','pubertal_sex_p','pds_1_p', 'pds_2_p', 'pds_3_p', 'pds_m4_p', 'pds_m5_p','pds_f4_p', 'pds_f5b_p']
df = pd.read_csv(path + file, delimiter='\t', skiprows=[1])[features]
df = df[df['eventname'] == 'baseline_year_1_arm_1']
df = df.replace(777.0, np.nan)
df = df.replace(999.0, np.nan)
df = df.replace(np.nan, 0.0)
df['puberty_score'] = df.apply(lambda x : x['pds_1_p'] + x['pds_2_p'] + x['pds_3_p'] + x['pds_m4_p'] + x['pds_m5_p']  if (x['pubertal_sex_p']==1.0) else x['pds_1_p'] + x['pds_2_p'] + x['pds_3_p'] + x['pds_f4_p'] + x['pds_f5b_p'], axis=1, result_type='reduce')
pubertal = df[['subjectkey', 'puberty_score']]
pubertal['puberty_score'] = pubertal['puberty_score'].astype('int')

file = 'medsy01.txt' # Prescription and OTC Meds
rx_features = []
otc_features = []
features = ['subjectkey', 'caff_24']
for i in range(1,16):
    rx_features.append('rx_med%d_24'%(i))
    otc_features.append('otc_med%d_24'%(i))
    features.append('rx_med%d_24'%(i))
    features.append('otc_med%d_24'%(i))
df = pd.read_csv(path + file, delimiter='\t', skiprows=[1], low_memory=False)
df = df[df['eventname'] == 'baseline_year_1_arm_1']
df = df[features]
df['caff_24'] = df[df['caff_24'] != 999]['caff_24']
df = df.fillna(0.0)
df['meds_rx_sum'] = df[rx_features].sum(axis=1)
df['meds_otc_sum'] = df[otc_features].sum(axis=1)
meds = df[['subjectkey', 'meds_rx_sum', 'meds_otc_sum', 'caff_24']].set_index('subjectkey')
meds = meds.astype('int').reset_index()

file = 'pdem02.txt' # Demographics Data
features = ['subjectkey', 'demo_brthdat_v2', 'demo_ed_v2', 'demo_race_a_p___10','demo_race_a_p___11','demo_race_a_p___12','demo_race_a_p___13','demo_race_a_p___14','demo_race_a_p___15','demo_race_a_p___16',
             'demo_race_a_p___17','demo_race_a_p___18','demo_race_a_p___19','demo_race_a_p___20','demo_race_a_p___21','demo_race_a_p___22','demo_race_a_p___23', 'demo_prnt_marital_v2', 'demo_prnt_ed_v2', 'demo_prnt_income_v2', 'demo_prnt_prtnr_v2', 'demo_prtnr_ed_v2', 'demo_prtnr_income_v2']
df = pd.read_csv(path + file, delimiter='\t', skiprows=[1])[features].set_index('subjectkey')
df = df.replace(777.0, np.nan)
df = df.replace(999.0, np.nan)
df = df.astype('int')
df['race_white'] = df['demo_race_a_p___10']
df['race_mixed'] = df['demo_race_a_p___11'] + df['demo_race_a_p___12'] + df['demo_race_a_p___13'] + df['demo_race_a_p___14'] + df['demo_race_a_p___15'] + df['demo_race_a_p___16'] + df['demo_race_a_p___17'] + df['demo_race_a_p___18'] + df['demo_race_a_p___19'] + df['demo_race_a_p___20'] + df['demo_race_a_p___21'] + df['demo_race_a_p___22'] + df['demo_race_a_p___23']
df['parent_education'] = df[['demo_prnt_ed_v2', 'demo_prtnr_ed_v2']].max(axis=1)
df['parent_income'] = df[['demo_prnt_income_v2', 'demo_prtnr_income_v2']].max(axis=1)
demo = df[['demo_brthdat_v2', 'demo_ed_v2', 'race_white', 'race_mixed', 'demo_prnt_marital_v2', 'parent_education', 'parent_income']].reset_index()

file = 'abcd_sds01.txt' # Sleep Data
sleep = read_file(path + file)
sleep = sleep[sleep['eventname']=='baseline_year_1_arm_1']
sleep = sleep[['subjectkey', 'sleepdisturb1_p']].dropna()
sleep['sleepdisturb1_p'] = sleep['sleepdisturb1_p'].astype('int')

data = bmi.merge(pubertal, how='inner', on='subjectkey', validate='1:1')
data = data.merge(meds, how='inner', on='subjectkey', validate='1:1')
data = data.merge(demo, how='inner', on='subjectkey', validate='1:1')
data = data.merge(sleep, how='inner', on='subjectkey', validate='1:1')
data['demo_brthdat_v2'] = data['demo_brthdat_v2'].astype('category')
data['demo_ed_v2'] = data['demo_ed_v2'].astype('category')
data['race_white'] = data['race_white'].astype('category')
data['race_mixed'] = data['race_mixed'].astype('category')
data['demo_prnt_marital_v2'] = data['demo_prnt_marital_v2'].astype('category')
data['parent_education'] = data['parent_education'].astype('category')
data = data.set_index('subjectkey')
data.to_pickle('data/covariates')
