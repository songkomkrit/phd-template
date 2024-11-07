import os
import shutil
import csv
import pandas as pd

from module.xutil import create_dir
from module.predregs import predregs

# Predefined directories
ts = "75305" # timestamp prefix
data = "seltrain20num3each20" # data name (no file extension)

indir = "../../../Projects/box/output"
inerrfile = f"{ts}-{data}-export-error-mfullaltseltol-2-t-1440.csv"
inselfile = f"{ts}-{data}-export-select-var-str-pcont-3-mfullaltseltol-2-t-1440.csv"
inregfile = f"{ts}-{data}-export-predict-region-pcont-3-mfullaltseltol-2-t-1440.csv"

outdir = f"../../../Outputs/Main/Box/{data}"
outerrfile = f"{ts}-error.csv"
outselfile = f"{ts}-selvarfin.csv"
outregfile = f"{ts}-predregfin.csv"

create_dir(outdir)

# Import DataFrame iterators
iterr = pd.read_csv(f"{indir}/{inerrfile}").itertuples() # classification errors
itsel = pd.read_csv(f"{indir}/{inselfile}").itertuples() # selected string variables
itreg = pd.read_csv(f"{indir}/{inregfile}").itertuples() # full decision regions

# Recalculate decision regions and predictions
# tsels: dictionary of selected variables
# tpreds: dictionary of new predicted classes in all new regions
tsels, tpreds = predregs(iterr, itsel, itreg, n_classes=5)

# Examples
iter = 15
print(f"\nSelected features (iteration {iter})\n{tsels[iter]}\n")
print(f"Decision regions (iteration {iter})\n{tpreds[iter]}\n")

# Export classification accuracies/errors
shutil.copy2(f"{indir}/{inerrfile}", outdir) # preserve file metadata
os.rename(f"{outdir}/{inerrfile}", f"{outdir}/{outerrfile}") # rename new file

# Export selected variables
with open(f"{outdir}/{outselfile}", 'w', newline='') as file:
    writer = csv.DictWriter(
        file,
        fieldnames = ['iter', 'jfin', 'j', 'var', 'type']
    )
    writer.writeheader()
for citer, info in tsels.items():
    dfs = pd.DataFrame({
        'iter': citer,
        'jfin': range(1, len(info['js'])+1),
        'j': info['js'],
        'variable': info['variables'],
        'type': info['types']
    })
    dfs.to_csv(f"{outdir}/{outselfile}", mode='a', header=False, index=False)

# Export predicted classes in all decision regions
with open(f"{outdir}/{outregfile}", 'w', newline='') as file:
    writer = csv.DictWriter(
        file,
        fieldnames = ['iter', 'regfin', 'occfin', 'predfin']
    )
    writer.writeheader()   
    for citer, preds in tpreds.items():
        for reg, info in preds.items():
            writer.writerow({
                'iter': citer,
                'regfin': reg,
                'occfin': 1 if info['occupy'] else 0,
                'predfin': str(info['classes'])
            })
