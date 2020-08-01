'''
    Stub to generate correlation plots
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# abcd_ml_subplot file contains followup ABCD data
df1 = pd.read_csv('abcd_ml_subplot.csv').set_index('subjectkey')[['plotresid', 'asr_scr_anxdep_t',
                                                                  'sleepdisturb1_p', 'meds_rx_sum', 'meds_otc_sum',
                                                                  'rsfmri_cor_ngd_copa_scs_agrh']]

df1 = df1.rename({'asr_scr_anxdep_t':'Anxious/Depressed Adult Self-Report Syndrome Scale', 'sleepdisturb1_p':'Child Sleep Duration',
                  'meds_rx_sum':'No. of prescription medications taken in the last 24 hours',
                  'meds_otc_sum':'No. of over-the-counter medications taken in the last 24 hours',
                  'rsfmri_cor_ngd_copa_scs_agrh':'Avg. correlation between cingulo-parietal network and right amygdala'}, axis=1)

sns.set(style="white")
corr = df1.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.color_palette("Blues")
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig('plotresid.png', dpi=500)
plt.show()


# all_current_data file contains current ABCD data
df2 = pd.read_excel('all_current_data.xlsx').set_index('subjectkey')
df2 = df2[['cbcl_scr_dsm5_depress_t', 'asr_scr_anxdep_t', 'sleepdisturb1_p',
           'meds_rx_sum', 'meds_otc_sum', 'rsfmri_cor_ngd_copa_scs_agrh']]

df2 = df2.rename({'asr_scr_anxdep_t':'Anxious/Depressed Adult Self-Report Syndrome Scale', 'sleepdisturb1_p':'Child Sleep Duration',
                  'meds_rx_sum':'No. of prescription medications taken in the last 24 hours',
                  'meds_otc_sum':'No. of over-the-counter medications taken in the last 24 hours',
                  'rsfmri_cor_ngd_copa_scs_agrh':'Avg. correlation between cingulo-parietal network and right amygdala'}, axis=1)

sns.set(style="white")
corr = df2.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.color_palette("Blues")
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig('cbcl.png', dpi=500)
plt.show()
