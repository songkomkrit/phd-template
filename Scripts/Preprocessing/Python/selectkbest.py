import pandas as pd
from functools import partial
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from module.utility import create_dir

sel_num_ls = [3, 4, 8]
train_eachclass_num = 20

data_filepath = "../../../Data/Processed/csv/proc20.csv"
info_filepath = "../../../Data/Encoded/info/pppub20enc-info.csv"

data_selname = "selproc20"
train_name = "seltrain20"
test_name = "seltest20"

# Predefined Directories
sample_dir = "../../../Samples/random"
sel_dir = f"{sample_dir}/{data_selname}"

data_dir = f"{sel_dir}/data"
info_dir = f"{sel_dir}/info"
feat_dir = f"{sel_dir}/features"
score_dir = f"{sel_dir}/scores"
train_dir = f"{sel_dir}/train"
test_dir = f"{sel_dir}/test"

create_dir(data_dir)
create_dir(info_dir)
create_dir(feat_dir)
create_dir(score_dir)
create_dir(train_dir)
create_dir(test_dir)

# Univariate Feature Selection
def feat_select(df_indata, df_info, sel_num):
    discrete_feat_idx = df_info.index[df_info['type']=='Categorical']
    score_func = partial(mutual_info_classif, discrete_features=discrete_feat_idx)
    feat_selector = SelectKBest(score_func, k=sel_num)
    feat_selector.fit(df_indata.drop('class', axis=1), df_indata['class'])

    df_scores = pd.DataFrame()
    df_scores["Attribute"] = df_indata.drop('class', axis=1).columns
    df_scores['Type'] = df_info['type']
    df_scores["Support"] = feat_selector.get_support()
    df_scores["F Score"] = feat_selector.scores_
    df_scores["P Value"] = feat_selector.pvalues_
    
    df_selfeat = df_scores[df_scores['Support']].drop('Support', axis=1).reset_index(drop=True)
    df_seldata = df_indata[df_selfeat['Attribute']].join(df_indata['class'])

    minmax = df_seldata.loc[:, df_seldata.columns != 'class'].agg(['min','max']).values.tolist()
    df_selfeat['Min'] = minmax[0]
    df_selfeat['Max'] = minmax[1]
    del minmax
    
    return df_seldata, df_selfeat, df_scores

# Implementation
df_indata = pd.read_csv(data_filepath)
df_info = pd.read_csv(info_filepath)

print(f"\n{df_indata.head()}\n")
print(f"{df_info.head()}\n")

for sel_num in sel_num_ls:
    
    # Univariate feature selection
    df_seldata, df_selfeat, df_scores = feat_select(df_indata=df_indata, df_info=df_info, sel_num=sel_num)

    # Display results (selected features)
    print(f"Select {sel_num} features:\n")
    print(f"{df_selfeat}\n")

    # Train-test split
    df_seltrain = df_seldata.groupby('class', group_keys=False).apply(
        lambda x: x.sample(train_eachclass_num)
    )
    df_seltest = df_seldata.drop(df_seltrain.index)
    
    # Exported results
    df_seldata.to_csv(f"{data_dir}/{data_selname}num{sel_num}.csv", header=True, index=False)
    
    df_selfeat.to_csv(f"{feat_dir}/fnum{sel_num}.csv", header=True, index=False)
    df_scores.to_csv(f"{score_dir}/snum{sel_num}.csv", header=True, index=False)
    
    df_selfeat.index = df_selfeat.index + 1
    df_selinfo = df_selfeat.drop(['F Score', 'P Value'], axis=1)
    df_selinfo.columns = ['variable', 'type', 'min', 'max']
    df_selinfo.to_csv(f"{info_dir}/{data_selname}num{sel_num}info.csv", index_label='id')

    df_seltrain.to_csv(f"{train_dir}/{train_name}num{sel_num}each{train_eachclass_num}.csv", header=True, index=False)
    df_seltest.to_csv(f"{test_dir}/{test_name}num{sel_num}exc{train_eachclass_num}.csv", header=True, index=False)
