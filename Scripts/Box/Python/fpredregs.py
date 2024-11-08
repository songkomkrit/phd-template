import csv
import re
import pandas as pd

from module.xutil import create_dir, copy, import_dict
from module.predregs import predregs


# Parameters
isreport = True # whether reports are written

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
if isreport: # reports must be written
    metadir = "../../../Data/Encoded/metadata" # metadata directory
    catmetafile = "meta-indep-cat-pppub20enc.json" # categorical metadata (after encoding) file
    # Relabel case-insensitive NIU values for all selected categorical features
    niudc = {
        'SS_YN': "NIU (aged below 15)",
        'PEMLR': "NIU (PEMLR)"
    }

# Required outputs
outdir = f"../../../Outputs/Main/Box/{data}" # main output directory
outselfile = f"{ts}-selvarfin.csv" # selected string variables, cuts and groups
outregfile = f"{ts}-predregfin.csv" # full decision regions

# Optional outputs
outerrfile = f"{ts}-error.csv" # classification errors
outcutcontfile = f"{ts}-cutcont.csv" # continuous cuts
outcutcatfile = f"{ts}-cutcat.csv" # categorical cuts
if isreport: # reports must be written
    outrepwdfile = f"{ts}-report-dup.csv" # report with duplicate entries
    outrepndfile = f"{ts}-report-nondup.csv" # nonduplicate with nonduplicate entries

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


# Export final reports (both duplicate and nonduplicate) (if specified)

if isreport: # reports must be written

    # New labels of selected categorical features (catvdc)
    catmetadc = import_dict(jsonpath=f"{metadir}/{catmetafile}") # metadata for categorical features
    catvars = set() # all selected categorical features (initialized)
    pattern = r'(^|[^\w])(niu)([^\w]|$)' # regex to search for niu
    pattern = re.compile(pattern, re.IGNORECASE)
    for info in tsels.values():
        for ind, attr in enumerate(info['variables']):
            if info['types'][ind] == 'cat':
                catvars.add(attr)
    catvdc = {attr: catmetadc[attr]['values'] for attr in catvars} # labels of selected categorical features
    for attr, valdc in catvdc.items():
        for val, desc in valdc.items():
            matches = re.search(pattern, desc.replace(',', ' '))
            if bool(matches): # case-insensitive value label containing niu
                try:
                    catvdc[attr][val] = niudc[attr] # relabel
                except KeyError: # new NIU label of current feature is missing
                    pass
    
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
                        desc = catvdc[info['variable']][str(elem)]
                        dc = {
                            'iter': citer,
                            'j': j, 'variable': info['variable'], 'type': 'Categorical',
                            'group': gr, 'member': elem, 'desc': desc
                        }
                        grls.append(dc)
    dfg = pd.DataFrame(grls) # group dataframe
    
    # Report dataframe with duplicate entries (dfrp)
    dfrp = pd.merge(dfe, dfg) # merge two dataframes: error/metric and group

    # Report dataframe with nonduplicate entries (dfn)
    dfn = dfrp.copy(deep=True)
    intcols = ['iter', 'status', 'j', 'group'] # integer columns
    ndcols = [
        ['iter', 'error', 'accuracy', 'ms', 'acctmin', 'status', 'relgap'],
        ['j', 'variable', 'type'],
        ['group']
    ]
    for i in range(len(ndcols),0,-1): # iterate over multilevel column lists with nonduplicate entries
        ccols = [f for cols in ndcols[0:i] for f in cols]
        dfn.loc[dfn[ccols].duplicated(), ccols] = pd.NA
    for col in intcols:
        dfn[col] = pd.array(dfn[col], dtype='Int16')
    
    # Export final reports
    dfrp.to_csv(f"{outdir}/{outrepwdfile}", header=True, index=False) # with duplicate entries
    dfn.to_csv(f"{outdir}/{outrepndfile}", sep=',', na_rep='', header=True, index=False) # with non-duplicate entries


# Display example of final report (with duplicate entries)
print(f"{dfrp.head()}\n")

# Display example of final report (with nonduplicate entries)
print(f"{dfn.head()}\n")
