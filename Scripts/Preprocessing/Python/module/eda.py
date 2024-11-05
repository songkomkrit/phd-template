import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

from module.utility import create_dir, backup_duplicate
from module.dataset import export_dataset

# Variables
def describe_var(var_dict, role='independent'):
    num_cat = 0
    num_cont = 0   
    for key in var_dict:
        if var_dict[key]['type'] == 'Categorical':
            num_cat += 1
        else:
            num_cont += 1           
    print(f"There are {num_cat + num_cont} {role} variables of interest: {num_cat} categorical and {num_cont} continuous")

# Cross Tabulation Analysis
def crosstab(df, indep_dict, cont_bins, plot, output_dir, log_filepath, backup_dir=''):
    dir_main = f"{output_dir}/tab-cbins-{cont_bins}"
    
    for key, val in indep_dict.items():
        fname_main = f"{key}-cbins-{cont_bins}"
        
        if val['type'] == "Categorical":
            crosstb = pd.crosstab(index=df[key].map(lambda x: val['values'][str(x)]), columns=df['code'])
        else:
            dat = df[[key, 'code']].copy()
            dat['bins'] = pd.cut(dat[key], bins=cont_bins)
            crosstb = pd.crosstab(index=dat['bins'],columns=dat['code'])
            del dat
            
        print(key)
        print(f"Label: {val['label']}")
        print(f"Universe: {val['universe']}")
        print(f"Type: {val['type']}")
        print(f"Topic: {val['topic']}")
        print(f"Subtopic: {val['subtopic']}")
        print("\n")
        
        print(f"Code: Employment-based plan (GRP) | Direct-purchase plan (DIR) | Public health insurance (PUB)")
        print(crosstb)
        '''
        dir_crosstb = f"{dir_main}/cross-{cont_bins}"
        create_dir(dir_crosstb)
        export_dataset(
            crosstb,
            file_dir=f"{dir_crosstb}/feather", dataset_name=f"{fname_main}-cross",
            format='feather', info=False,
            backup_dir=backup_dir
        )
        export_dataset(
            crosstb,
            file_dir=f"{dir_crosstb}/csv", dataset_name=f"{fname_main}-cross",
            format='csv', info=False,
            backup_dir=backup_dir
        )
        '''
        print("\n")
        
        if plot:
            barplot = crosstb.plot.bar()
            barplot.legend(title='(GRP,DIR,PUB)',
                          bbox_to_anchor=(1,1.02),
                          loc='upper left')
            plt.title(val['label'])
            plt.xlabel(key)
            plt.ylabel('Frequency')
            ls_format = ['svg', 'pgf', 'pdf']
            for format in ls_format:
                dir_fig = f"{dir_main}/figures/{format}"
                figname = f"{key}-cbins-{cont_bins}"
                figpath = f"{dir_fig}/{figname}.{format}"
                create_dir(dir_fig)
                backup_duplicate(
                    file_dir=dir_fig, filename=figname,
                    format=format,
                    backup_dir=backup_dir, info=False
                )
                f = open(log_filepath, 'a')
                temp = sys.stdout
                sys.stdout = f
                count, tries = 0, 4
                success = False
                while count < tries:
                    try:
                        plt.savefig(figpath, bbox_inches='tight')
                        success = True
                        break
                    except:
                        pass
                    count += 1
                if not success:
                    curtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    print(f"{curtime} | {key}: {figpath} cannot be saved")
                sys.stdout = temp
                f.close()
            #plt.show()
            
        dftb = crosstb.reset_index().rename_axis(None, axis=1)
        dftb[dftb.columns[1:]] = dftb[dftb.columns[1:]].astype('uint32')
        export_dataset(
            dftb,
            file_dir=f"{dir_main}/feather", dataset_name=fname_main,
            format='feather', info=False,
            backup_dir=backup_dir
        )
        export_dataset(
            dftb,
            file_dir=f"{dir_main}/csv", dataset_name=fname_main,
            format='csv', info=False,
            backup_dir=backup_dir
        )
        print("\n-----------------------------------------")
