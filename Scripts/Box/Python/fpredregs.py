import csv
import pandas as pd

from module.xutil import create_dir, copy
from module.predregs import predregs

# Informational prefixes/postfixes
ts = "75305" # last digits of timestamp
data = "seltrain20num3each20" # data name (no file extension)
inprefix = f"{ts}-{data}-export-" # input filename prefix
inpostfix = "-mfullaltseltol-2-t-1440" # input filename postfix

# Required inputs
indir = "../../../Projects/box/output" # main directory
inerrfile = f"{inprefix}error{inpostfix}.csv" # classification errors
inselfile = f"{inprefix}select-var-str-pcont-3{inpostfix}.csv" # selected string variables
inregfile = f"{inprefix}predict-region-pcont-3{inpostfix}.csv" # full decision regions

# Optional inputs
incutcontfile = f"{inprefix}cutcont-full-pcont-3{inpostfix}.csv" # continuous cuts
incutcatfile = f"{inprefix}cutcat-full-pcont-3{inpostfix}.csv" # categorical cuts

# Required outputs
outdir = f"../../../Outputs/Main/Box/{data}" # main directory
outerrfile = f"{ts}-error.csv" # classification errors
outselfile = f"{ts}-selvarfin.csv" # selected string variables
outregfile = f"{ts}-predregfin.csv" # full decision regions

# Optional outputs
outcutcontfile = f"{ts}-cutcont.csv" # continuous cuts
outcutcatfile = f"{ts}-cutcat.csv" # categorical cuts

# Create main output directory (if not exist)
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

# Export non-edited information
copy(f"{indir}/{inerrfile}", f"{outdir}/{outerrfile}") # classification errors
copy(f"{indir}/{incutcontfile}", f"{outdir}/{outcutcontfile}") # continuous cuts
copy(f"{indir}/{incutcatfile}", f"{outdir}/{outcutcatfile}") # categorical cuts

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
