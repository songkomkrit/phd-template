import os
import urllib.request
import pandas as pd
import pyarrow

from module.utility import create_dir, backup_duplicate

# Import
def import_dataset(dataset_name, feather_dir, sas_dir='', sas_url=''):
    filepath_feather = f"{feather_dir}/{dataset_name}.feather"
    
    if os.path.isfile(filepath_feather):
        print(f"{filepath_feather} is found")
        print(f"{filepath_feather} was previously preprocessed")
        df0 = pd.read_feather(filepath_feather)
    else:
        print(f"{filepath_feather} is not found")
        if sas_dir == '':
            raise Exception("SAS data directory is empty")
        filepath_sas = f"sas_dir/{dataset_name}.sas7bdat"
        if os.path.isfile(filepath_sas):
            print(f"{filepath_sas} is found")
        else:
            print(f"{filepath_sas} is not found")
            create_dir('original/data-orig')
            print(f"{filepath_sas} will be downloaded")
            print("Download starts")
            try:
                urllib.request.urlretrieve(sas_url, filepath_sas)
                print("Download finishes")
            except:
                raise Exception("Download fails")
            print(f"{filepath_sas} is successfully downloaded")
        df0 = pd.read_sas(filepath_sas)
    
    print(f"\nNumber of original data: {len(df0)}")
    df0 = df0[df0['COV']!=0]
    print(f"An infant born after calendar year (COV = 0) is excluded")
    print(f"Number of training data: {len(df0)}")
    return df0

# Export
def export_dataset(df, file_dir, dataset_name, format, info=True, backup_dir=''):
    create_dir(file_dir)
    if format == 'feather' or format == 'csv':
        filepath = f"{file_dir}/{dataset_name}.{format}"
        if backup_dir != '':
            backup_duplicate(
                file_dir=file_dir, filename=dataset_name,
                format=format,
                backup_dir=backup_dir, info=info
            )
        if format == 'feather':
            df.to_feather(filepath)
        else:
            df.to_csv(filepath, index=False)
        if info:
            print(f"The dataframe is successfully exported to {filepath}")
    else:
        print(f"Input format {format} is unrecognized")
