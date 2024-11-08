import csv
import pandas as pd

from module.xutil import create_dir, copy, import_dict
from module.predregs import predregs


# Parameters
isreport = True # whether report is written

# Informational prefixes/postfixes
ts = "75305" # last digits of timestamp
data = "seltrain20num3each20" # data name (no file extension)
inprefix = f"{ts}-{data}-export-" # input filename prefix
inpostfix = "-mfullaltseltol-2-t-1440" # input filename postfix

# Required inputs
indir = "../../../Projects/box/output" # main input directory
inerrfile = f"{inprefix}error{inpostfix}.csv" # classification errors and performance metrics
inselfile = f"{inprefix}select-var-str-pcont-3{inpostfix}.csv" # selected string variables
inregfile = f"{inprefix}predict-region-pcont-3{inpostfix}.csv" # full decision regions
incutcontfile = f"{inprefix}cutcont-full-pcont-3{inpostfix}.csv" # continuous cuts
incutcatfile = f"{inprefix}cutcat-full-pcont-3{inpostfix}.csv" # categorical cuts

# Optional inputs
if isreport: # report must be written
    metadir = "../../../Data/Encoded/metadata" # metadata directory
    catmetafile = "meta-indep-cat-pppub20enc.json" # categorical metadata (after encoding) file

# Required outputs
outdir = f"../../../Outputs/Main/Box/{data}" # main output directory
outselfile = f"{ts}-selvarfin.csv" # selected string variables, cuts and groups
outregfile = f"{ts}-predregfin.csv" # full decision regions

# Optional outputs
outerrfile = f"{ts}-error.csv" # classification errors
outcutcontfile = f"{ts}-cutcont.csv" # continuous cuts
outcutcatfile = f"{ts}-cutcat.csv" # categorical cuts
if isreport: # report must be written
    outrepfile = f"{ts}-report.csv" # report

# Create main output directory (if not exist)
create_dir(outdir)

# Import DataFrame iterators
iterr = pd.read_csv(f"{indir}/{inerrfile}").itertuples() # classification errors and performance metrics
itsel = pd.read_csv(f"{indir}/{inselfile}").itertuples() # selected string variables
itreg = pd.read_csv(f"{indir}/{inregfile}").itertuples() # full decision regions
itcont = pd.read_csv(f"{indir}/{incutcontfile}").itertuples() # full continuous cuts
itcat = pd.read_csv(f"{indir}/{incutcatfile}").itertuples() # full categorical cuts

''' 
    Recalculate decision regions and predictions, and also provide cuts
        tsels: dictionary of selected variables
        tpreds: dictionary of new predicted classes in all new regions
        tcuts: dictionary of cuts along all selected features
'''
tsels, tpreds, tcuts = predregs(iterr, itsel, itreg, itcont, itcat, n_classes=5)


# Examples
iters = [1, 2, 15]
print()
for citer in iters:
    print(f"Selected features (iteration {citer})\n{tsels[citer]}\n")
    print(f"Decision regions (iteration {citer})\n{tpreds[citer]}\n")
    print(f"Cuts (iteration {citer})\n{tcuts[citer]}\n")


# Export non-edited information
copy(f"{indir}/{inerrfile}", f"{outdir}/{outerrfile}") # classification errors
copy(f"{indir}/{incutcontfile}", f"{outdir}/{outcutcontfile}") # continuous cuts
copy(f"{indir}/{incutcatfile}", f"{outdir}/{outcutcatfile}") # categorical cuts


# Export selected variables, cuts and groups
with open(f"{outdir}/{outselfile}", 'w', newline='') as file:
    writer = csv.DictWriter(
        file,
        fieldnames = ['iter', 'jfin', 'j', 'var', 'type', 'p', 'cuts', 'groups']
    )
    writer.writeheader()
for citer, info in tsels.items():
    cuts = [tcuts[citer][j]['cuts'] for j in info['js']]
    groups = [tcuts[citer][j]['groups'] for j in info['js']]
    dfs = pd.DataFrame({
        'iter': citer,
        'jfin': range(1, len(info['js'])+1),
        'j': info['js'],
        'variable': info['variables'],
        'type': info['types'],
        'p': info['ps'],
        'cuts': cuts,
        'groups': groups
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


# Export final report

if isreport: # report must be written

    # Metadata (after encoding)
    catmetadc = import_dict(jsonpath=f"{metadir}/{catmetafile}")
    
    # Classification errors and performance metrics
    efields = ['iter', 'error', 'accuracy', 'ms', 'acctmin', 'status', 'relgap']
    dfe = pd.read_csv(f"{indir}/{inerrfile}", usecols=efields) # error/metric dataframe
    
    # Groups
    grls = list() # list of all member groups across all features and iterations
    for citer, scuts in tcuts.items():
        for j, info in scuts.items(): # cuts along all selected feature
            if info['type'] == 'cont': # continuous feature (no grouping: group = -1)
                dc = {
                    'iter': citer,
                    'j': j, 'variable': info['variable'], 'type': 'Continuous',
                    'group': -1, 'member': info['cuts'], 'desc': 'NA'
                }
                grls.append(dc)
            else: # categorical feature (grouping allowed)
                for gr, members in info['groups'].items():
                    for elem in members: # all members in a specific group
                        desc = catmetadc[info['variable']]['values'][str(elem)]
                        dc = {
                            'iter': citer,
                            'j': j, 'variable': info['variable'], 'type': 'Categorical',
                            'group': gr, 'member': elem, 'desc': desc
                        }
                        grls.append(dc)
    dfg = pd.DataFrame(grls) # group dataframe
    
    # Merge two dataframes: error/metric and group
    dfrp = pd.merge(dfe, dfg) # report dataframe

    # Export
    dfrp.to_csv(f"{outdir}/{outrepfile}", header=True, index=False)


# Display example of final report
print(f"{dfrp.head()}\n")
