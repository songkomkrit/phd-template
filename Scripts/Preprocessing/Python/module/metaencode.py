import pandas as pd

def extract_dict_cat(indep_dict):
    return {attr: info for (attr, info) in indep_dict.items() if indep_dict[attr]['type'] == 'Categorical'}

def extract_dict_cont(indep_dict):
    return {attr: info for (attr, info) in indep_dict.items() if indep_dict[attr]['type'] == 'Continuous'}

def sort_cols(df_indep, indep_dict):
    sorted_cols = sorted(
        df_indep.head(), 
        key=lambda attr: indep_dict[attr]['type'],
        reverse=True
    )
    return df_indep[sorted_cols]

def indep_info(df_indep, indep_dict):
    df_info = pd.DataFrame({'variable': df_indep.head().columns})
    df_info['type'] = df_info['variable'].apply(lambda attr: indep_dict[attr]['type'])
    minmax = df_indep.agg(['min','max']).values.tolist()
    df_info['min'] = minmax[0]
    df_info['max'] = minmax[1]
    del minmax
    return df_info

def count_info(df_info):
    df_count = df_info.groupby('type').count().reset_index()[['type','variable']]
    df_count.rename(columns = {'variable': 'count'}, inplace=True)
    df_count.sort_values('type', ascending=False, inplace=True, ignore_index=True)
    return df_count
