import pandas as pd

from module.utility import create_dir
from cls.Info import *

# Given Information
pcut_ls = [2, 3]
info_ls = []
info_ls.append({
    'indir': "../../../Data/Encoded/info",
    'infile': "pppub20enc-info.csv",
    'outdir': "../../../Samples/proc20/cuts"
})
extra_infile_ls = [
    "selproc20num3info.csv",
    "selproc20num4info.csv",
    "selproc20num8info.csv"
]
for file in extra_infile_ls:
    info_ls.append({
        'indir': "../../../Samples/selproc20/info",
        'infile': file,
        'outdir': "../../../Samples/selproc20/cuts"
    })
print(f"\n{info_ls}\n")

# Implementation
for dc in info_ls:
    for pcut in pcut_ls:
        
        # Import
        inpath = f"{dc['indir']}/{dc['infile']}"
        df = pd.read_csv(inpath)
        
        # Set cuts
        pcont, pcatmax = pcut, pcut
        df.info.setcut(pcont, pcatmax)

        # Set output path
        infilename = dc['infile'].replace('.csv', '').replace('info', '').replace('-', '')
        cutfilename = f"{infilename}co{pcont}ca{pcatmax}cutinfo"
        outpath = f"{dc['outdir']}/{cutfilename}.csv"
        
        # Display results
        print(f"Input: {inpath}")
        print(f"NUmber of features: {len(df)}")
        print(f"Number of continuous cuts: {pcont}")
        print(f"Number of maximum categorical cuts: {pcatmax}")
        print(f"Output: {outpath}\n")
        print(f"{df.head()}\n")
        
        # Export
        create_dir(dc['outdir'])
        df.to_csv(outpath, header=True, index=False)
