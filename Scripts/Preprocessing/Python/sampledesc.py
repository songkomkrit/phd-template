import os
import pandas as pd
import warnings

from module.utility import create_dir, import_dict
from module.eda import crosstab

texlive_binpath = '/usr/local/texlive/2024/bin/x86_64-linux'
os.environ['PATH'] += os.pathsep + texlive_binpath

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

# Given Information
dataset_name = "seltrain20num3each20"

# Predefined Directories
meta_dir = "../../../Data/Original/metadata"
csv_dir = "../../../Samples/selproc20/train"

output_dir = f"../../../Outputs/Main/EDA/{dataset_name}"
log_dir = f"../../../Logs/samples"
log_filepath = f"{log_dir}/sampledesc.log"

backup_dir = "../../../Backups"

create_dir(log_dir)

# Data Preparation
indep_dict_full = import_dict(metadatapath=f"{meta_dir}/meta-indep.json")
filepath_csv = f"{csv_dir}/{dataset_name}.csv"
df = pd.read_csv(filepath_csv)

indep_dict = dict()
for attr in df.columns[0:-1]:
    indep_dict[attr] = indep_dict_full[attr]

df['code'] = df['class'].apply(
    lambda v: 'NNN' if v == 0
    else 'NNY' if v == 1
    else 'NY_' if v == 2
    else 'YNN' if v == 3
    else 'Y1Y', 
).astype('category')

# Cross Tabulation Analysis
print("-----------------------------------------")
crosstab(df=df, indep_dict=indep_dict, cont_bins=5, plot=True, output_dir=output_dir, log_filepath=log_filepath, backup_dir=backup_dir)
