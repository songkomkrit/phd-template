import os
import pandas as pd
import warnings

from module.utility import create_dir, import_dict
from module.eda import *
from module.dataset import *
from cls.ThesisExtension import *

texlive_binpath = '/usr/local/texlive/2024/bin/x86_64-linux'
os.environ['PATH'] += os.pathsep + texlive_binpath

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

# Given Information
dataset_name = "pppub20"

# Predefined Directories
meta_dir = "../../../Data/Original/metadata"
feather_dir = "../../../Data/Original/feather"
csv_dir = "../../../Data/Original/csv"

output_dir = f"../../../Outputs/Main/EDA/{dataset_name}"
log_dir = f"../../../Logs/preprocessing"
log_filepath = f"{log_dir}/describe.log"

backup_dir = "../../../Backups"

create_dir(log_dir)

# Data Preparation
indep_dict = import_dict(metadatapath=f"{meta_dir}/meta-indep.json")
dep_attrs = ['GRP', 'DIR', 'PUB']
print()
describe_var(indep_dict)
print()
df = import_dataset(dataset_name=dataset_name, feather_dir=feather_dir)
print()
dep_features = ['class_orig', 'code_orig', 'code', 'class']
acpt_types = {'category', 'int16', 'int32', 'int8', 'uint16', 'uint32', 'uint8'}
preprocess = True

if all(feat in df.columns for feat in dep_features):
    col_types = set()
    for col in df.columns:
        col_types.add(str(df[col].dtype))
        if col_types == acpt_types:
            preprocess = False

if preprocess:
    df.thesis.code(indep_dict, dep_attrs)
    df.thesis.recode()

filepath_feather = f"{feather_dir}/{dataset_name}.feather"
filepath_csv = f"{csv_dir}/{dataset_name}.csv"

if not os.path.isfile(filepath_feather):
    export_dataset(df, file_dir='data/feather', dataset_name=dataset_name, format='feather')

if not os.path.isfile(filepath_csv):
    dfther = pd.read_feather(filepath_feather)
    export_dataset(dfther, file_dir='data/csv', dataset_name=dataset_name, format='csv')

# Univariate Data Analysis
df.thesis.show_type(option='full')
print()
df[['GRP','DIR','PUB','class_orig','code_orig','code','class']].drop_duplicates().sort_values('class').reset_index(drop=True)
print(f"Code: Employment-based plan (GRP) | Direct-purchase plan (DIR) | Public health insurance (PUB)")
print(df.groupby('code').size())
print('\n'*2)

# Cross Tabulation Analysis
print("-----------------------------------------")
crosstab(df=df, indep_dict=indep_dict, cont_bins=10, plot=True, output_dir=output_dir, log_filepath=log_filepath, backup_dir=backup_dir)
