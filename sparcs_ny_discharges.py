pip install tableone

pip install researchpy

# Import packages
import seaborn
from tableone import TableOne, load_dataset
import researchpy as rp
import patsy
from statsmodels.formula.api import ols
import pandas
import scipy
from scipy import stats
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.express as px
import urllib.request
import os

sparcs = pandas.read_csv('https://health.data.ny.gov/resource/gnzp-ekau.csv')
sparcs

sparcs.shape    # 1000 rows and 34 columns
sparcs.columns
sparcs.dtypes

sparcs.mean()
sparcs.var()
sparcs.describe()

#catagorical values: comparing groups or catagories
# seeing the relationship between severity of illness and race
race_sever = ols("apr_severity_of_illness_code ~ race + 1", sparcs).fit()
print(race_sever.summary())

# seeing the relationship between length of stay and age group
LOS_age = ols("length_of_stay ~ age_group + 1", sparcs).fit()
print(LOS_age.summary())

# a correlation coefficient with scipy.stats.linregress():
scipy.stats.linregress(sparcs['length_of_stay'],
                       sparcs['apr_severity_of_illness_code'])

#### TableOne ####

### DATASET 1 ###
sparcs_df1 = sparcs.copy()
sparcs_df1.dtypes
list(sparcs_df1)

sparcs_df1_columns = ['age_group', 'gender', 'race', 'ethnicity',
                      'ccs_diagnosis_code', 'apr_severity_of_illness_code']
sparcs_df1_categories = ['age_group', 'gender', 'race',
                         'ethnicity', 'apr_severity_of_illness_code']
sparcs_df1_groupby = ['apr_risk_of_mortality']

sparcs_df1_table1 = TableOne(sparcs_df1, columns=sparcs_df1_columns,
                             categorical=sparcs_df1_categories, groupby=sparcs_df1_groupby, pval=False)
print(sparcs_df1_table1.tabulate(tablefmt="fancy_grid"))
sparcs_df1_table1.to_csv('Data/modified_dataset.csv')

##  ResearchOne  ##

# examples of getting descriptives for the entire dataset
rp.codebook(sparcs)

rp.summary_cont(
    sparcs[['length_of_stay', 'apr_severity_of_illness_code', 'total_charges']])
rp.summary_cat(sparcs[['age_group', 'gender', 'race',
                       'ethnicity', 'apr_severity_of_illness_code']])

##  Visualizing Data ##

# using a histogram to see the frequency counts of length of stay
hist, bin_edges = np.histogram(sparcs['length_of_stay'], bins=50)
hist
bin_edges

fig, ax = plt.subplots()
ax.hist(sparcs['length_of_stay'], bin_edges, cumulative=False)
ax.set_xlabel('LOS')
ax.set_ylabel('Frequency')
plt.show()
