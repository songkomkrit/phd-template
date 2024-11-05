import os
import pandas as pd
import pyarrow

from module.utility import create_dir, import_dict, export_json, export_txt
from module.metaencode import *
from cls.Data import *

# Given Information
dataset_inname = "pppub20"
dataset_encname = f"{dataset_inname}enc"
dataset_procname = "proc20"

# Predefined Directories
meta_indir = "../../../Data/Original/metadata"
meta_extra_indir = f"{meta_indir}/extra"
feather_indir = "../../../Data/Original/feather"
csv_indir = "../../../Data/Original/csv"

meta_encdir = "../../../Data/Encoded/metadata"
meta_extra_encdir = f"{meta_encdir}/extra"
feather_encdir = "../../../Data/Encoded/feather"
csv_encdir = "../../../Data/Encoded/csv"
info_encdir = "../../../Data/Encoded/info"

csv_procdir = "../../../Data/Processed/csv"

create_dir(meta_extra_indir)
create_dir(feather_indir)
create_dir(csv_indir)
create_dir(meta_extra_encdir)
create_dir(feather_encdir)
create_dir(csv_encdir)
create_dir(info_encdir)
create_dir(csv_procdir)

# Metadata
indep_dict = import_dict(metadatapath=f"{meta_indir}/meta-indep.json")
export_json(extract_dict_cat(indep_dict), f"{meta_extra_indir}/meta-indep-cat.json")
export_json(extract_dict_cont(indep_dict), f"{meta_extra_indir}/meta-indep-cont.json")

# Imported Dataset
if os.path.isfile(f"{feather_indir}/{dataset_inname}.feather"):
    df = pd.read_feather(f"{feather_indir}/{dataset_inname}.feather")
    if not os.path.isfile(f"{csv_indir}/{dataset_inname}.csv"):
        df.to_csv(f"{csv_indir}/{dataset_inname}.csv", index=False)
else:
    df = pd.read_csv(f"{csv_indir}/{dataset_inname}.csv")

# Encoded Dataset and Dictionary
data_obj = Data(df.copy(), indep_dict.copy())
cat_var_change = data_obj.encodecat()
cont_var_nonpos = data_obj.encodecont()
df_enc = data_obj.dataset
indep_dict_enc = data_obj.metadata

# Processed Dataset
dep_attrs = ['GRP', 'DIR', 'PUB']
class_attrs = ['class_orig','code_orig','code','class']
df_proc_enc = df_enc.drop(columns=['COV']+dep_attrs+class_attrs)
df_proc_enc = sort_cols(df_proc_enc, indep_dict_enc).join(df_enc['class'])
df_proc_info = indep_info(df_proc_enc.loc[:, df_proc_enc.columns != 'class'], indep_dict_enc)
df_count_info = count_info(df_proc_info)

# Exported Results
df_enc.to_feather(f"{feather_encdir}/{dataset_encname}.feather")
df_enc.to_csv(f"{csv_encdir}/{dataset_encname}.csv", index=False)
export_json(extract_dict_cat(indep_dict_enc), f"{meta_encdir}/meta-indep-cat-{dataset_encname}.json")

df_proc_enc.to_csv(f"{csv_procdir}/{dataset_procname}.csv", header=True, index=False)

df_proc_info.index = df_proc_info.index + 1
df_proc_info.to_csv(f"{info_encdir}/{dataset_encname}-info.csv", index_label="id")
df_count_info.to_csv(f"{info_encdir}/{dataset_encname}-countinfo.csv", header=True, index=False)

export_txt(cat_var_change, f"{meta_extra_encdir}/catchange-{dataset_encname}.txt")
export_txt(cont_var_nonpos, f"{meta_extra_encdir}/contnonpos-{dataset_encname}.txt")
