import csv
import re
import pandas as pd

from module.operation.xutil import *
from module.operation.typecast import settostr, itvtostr
from module.model.findsels import findsels
from module.model.findcuts import findcuts
from module.model.findtregs import findtregs
from module.model.findcregs import findcregs
from module.model.findcorr import findcorr


# Parameters
pcuto = [3,3,2] # original cut numbers across all given features
isexample = True # whether example is shown
issreport = True # whether reports of feature selection are written
isrreport = True # whether reports of detailed decision regions are written

# Informational prefixes/postfixes
ts = "75305" # last digits of timestamp
data = "seltrain20num3each20" # data name (no file extension)
inprefix = f"{ts}-{data}-export-" # input filename prefix
inpostfix = "-mfullaltseltol-2-t-1440" # input filename postfix

# Required inputs
datdir = "../../../Projects/box/input" # directory of training instances (cplex inputs)
indir = "../../../Projects/box/output" # main input directory (cplex results)
datfile = f"{data}.csv" # training dataset with target variable
datpredfile = f"{inprefix}predict-instance-pcont-3{inpostfix}.csv" # individual result of cplex prediction
inerrfile = f"{inprefix}error{inpostfix}.csv" # classification errors and performance metrics
inselfile = f"{inprefix}select-var-str-pcont-3{inpostfix}.csv" # selected string variables
incutcontfile = f"{inprefix}cutcont-full-pcont-3{inpostfix}.csv" # continuous cuts
incutcatfile = f"{inprefix}cutcat-full-pcont-3{inpostfix}.csv" # categorical cuts

# Optional inputs
if issreport: # reports of feature selection must be written
    metadir = "../../../Data/Encoded/metadata" # metadata directory
    catmetafile = "meta-indep-cat-pppub20enc.json" # categorical metadata (after encoding) file
    # Relabel case-insensitive NIU values for all selected categorical features
    niudc = {'SS_YN': "NIU (aged below 15)", 'PEMLR': "NIU"}
if isrreport: # reports of detailed decision regions must be written
    clabels = {0: 'NNN', 1: 'NNY', 2: 'NY_', 3: 'YNN', 4: 'Y1Y'}

# Required outputs
outdir = f"../../../Outputs/Main/Box/{data}" # main output directory
outeperffile = f"{ts}-eperf.csv" # classification performances (accuracy/error/time)
outselfile = f"{ts}-selvarfin.csv" # selected string variables, cuts and groups
outregfile = f"{ts}-predregfin.csv" # full decision regions

# Optional outputs
outcutcontfile = f"{ts}-cutcont.csv" # continuous cuts
outcutcatfile = f"{ts}-cutcat.csv" # categorical cuts
if issreport: # reports of feature selection must be written
    outsrepwdfile = f"{ts}-report-sel-dup.csv" # with duplicate entries
    outsrepndfile = f"{ts}-report-sel-nondup.csv" # with nonduplicate entries
if isrreport: # reports of detailed decision regions
    outrrepwdfile = f"{ts}-report-reg-dup.csv" # with duplicate entries
    outrrepndfile = f"{ts}-report-reg-nondup.csv" # with nonduplicate entries

# Create main output directory (if not exist)
create_dir(outdir)

# Import datasets
dfe = pd.read_csv(f"{indir}/{inerrfile}") # cplex classification errors and performance metrics
dfs = pd.read_csv(f"{indir}/{inselfile}") # selected string variables
dfcont = pd.read_csv(f"{indir}/{incutcontfile}") # full continuous cuts
dfcat = pd.read_csv(f"{indir}/{incutcatfile}") # full categorical cuts
df = pd.read_csv(f"{datdir}/{datfile}") # training dataset including target variable
dfp = pd.read_csv(f"{indir}/{datpredfile}") # individual result of cplex prediction

# Initialize DataFrame iterators
itsel = dfs.itertuples() # selected string variables
itcont = dfcont.itertuples() # full continuous cuts
itcat = dfcat.itertuples() # full categorical cuts
itpred = dfp.itertuples() # individual result of cplex prediction

# Main execution
tsels = findsels(itsel, pcuto) # selected variables
tcuts = findcuts(tsels, itcont, itcat) # cuts along all selected features
ttregs = findtregs(tsels, tcuts, df) # new true regions and predicted classes
tcregs = findcregs(tsels, itpred, pcuto) # new cplex regions and predicted classes
tcorr, ccorr = findcorr(ttregs, tcregs) # true/cplex correctness

# Calculate performance results
dfen = pd.DataFrame({
    'iter': tcorr.keys(), # iteration that selects feature
    'taccuracy': [info['correct']*100/len(df) for info in tcorr.values()], # true accuracies
    'caccuracy': [info['correct']*100/len(df) for info in ccorr.values()] # recalculated cplex accuracies
})
dfen['terror'] = 100 - dfen['taccuracy'] # true errors
dfen['cerror'] = 100 - dfen['caccuracy'] # recalculated cplex errors
dfen = pd.merge(dfen, dfe, how='outer')
dfen.rename(columns = {
    'error': 'rerror', # reported cplex errors
    'accuracy': 'raccuracy' # reported cplex accuracies
}, inplace=True)
cols = dfen.columns.tolist()
cols = cols[0:1] + cols[5:5+len(pcuto)] + cols[1:3] + cols[-6:-5] + cols[3:5] + cols[-7:-6] + cols[-5:]
dfen = dfen[cols] # rearranged columns
dfen['ms'] = dfen['ms']/60000 # convert milliseconds to minutes
dfen = dfen.rename(columns={'ms':'minute'})

# Display performance results
print(f"\n{dfen}\n")

# Examples
if isexample:
    iters = [1, 2, 15]
    for citer in iters:
        try:
            print(f"Selected features (iteration {citer})\n{tsels[citer]}\n")
            print(f"Cuts (iteration {citer})\n{tcuts[citer]}\n")
            print(f"True decision regions (iteration {citer})\n{ttregs[citer]}\n")
            print(f"Cplex decision regions (iteration {citer})\n{tcregs[citer]}\n")
            print(f"True correctness (iteration {citer})\n{tcorr[citer]}\n")
            print(f"Cplex correctness (iteration {citer})\n{ccorr[citer]}\n")
        except KeyError:
            print(f"Iteration {citer} selects no features\n")

# Export non-edited information
copy(f"{indir}/{incutcontfile}", f"{outdir}/{outcutcontfile}") # continuous cuts
copy(f"{indir}/{incutcatfile}", f"{outdir}/{outcutcatfile}") # categorical cuts

# Export performance results (accuracy/error/time)
dfen.to_csv(f"{outdir}/{outeperffile}", float_format="%.2f", header=True, index=False)

# Export selected variables, cuts and groups
with open(f"{outdir}/{outselfile}", 'w', newline='') as file:
    writer = csv.DictWriter(
        file,
        fieldnames = [
            'iter', 'jfin', 'j', 'var', 'type',
            'p', 'cuts', 'groups'
        ]
    )
    writer.writeheader()
for citer, info in tsels.items():
    cuts = [[round(cut, 2) for cut in tcuts[citer][j]['cuts']] for j in info['js']]
    groups = list()
    for ind, j in enumerate(info['js']):
        if info['types'][ind] == 'cont': # continuous feature
            jgrs = dict()
            for gr, member in tcuts[citer][j]['groups'].items():
                jgrs[gr] = itvtostr(member)
            groups.append(jgrs)
        else: # categorical feature
            groups.append(tcuts[citer][j]['groups'])
    dfs = pd.DataFrame({
        'iter': citer,
        'jfin': range(1, len(info['js'])+1), # 1, 2, ...
        'j': info['js'], # j in cplex model
        'variable': info['variables'],
        'type': info['types'],
        'p': info['ps'],
        'cuts': cuts,
        'groups': groups
    })
    dfs.to_csv(f"{outdir}/{outselfile}", mode='a', header=False, index=False)

# Export predicted classes and number of instances in all decision regions
with open(f"{outdir}/{outregfile}", 'w', newline='') as file:
    writer = csv.DictWriter(
        file,
        fieldnames = ['iter', 'reg', 'ninst', 'tpred', 'cpred',
                      'tcorr', 'ccorr', 'ncinst']
    )
    writer.writeheader()   
    for citer, tregs in ttregs.items():
        for b, treg in tregs.items():
            writer.writerow({
                'iter': citer,
                'reg': b,
                'ninst': treg['ninst'], # number of instances
                'tpred': settostr(treg['classes']), # true predicted class
                'cpred': ','.join([settostr(st) for st in tcregs[citer][b]['lclasses']]), # cplex predicted class
                'tcorr': tcorr[citer]['detail'][b], # true correctness
                'ccorr': ccorr[citer]['detail'][b], # cplex correctness
                'ncinst': treg['ncinst'] # targets and number of member instances
            })


# Export final reports of feature selection (with duplicate/nonduplicate entries) (if specified)

if issreport: # reports of feature selection must be written

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
    
    # True classification accuracies and performance metrics
    efields = ['iter', 'taccuracy', 'minute', 'acctmin', 'status']
    
    # Groups
    grls = list() # list of all member groups across all features and iterations
    for citer, scuts in tcuts.items():
        for j, info in scuts.items(): # cuts along all selected feature
            if info['type'] == 'cont': # continuous feature (groups not displayed for convenience)
                dc = {
                    'iter': citer,
                    'j': j, 'variable': info['variable'], 'type': 'Continuous',
                    'group': pd.NA, 'member': info['cuts'], 'desc': pd.NA
                }
                grls.append(dc)
            else: # categorical feature (groups displayed)
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
    
    # Report dataframe of feature selection with duplicate entries (dfrp)
    dfsrp = pd.merge(dfen[efields], dfg) # merge two dataframes: error/metric and group

    # Report dataframe of feature selection with nonduplicate entries (dfn)
    dfsrpn = nondup(
        dfsrp,
        ndcols=[
            ['iter', 'taccuracy', 'minute', 'acctmin', 'status'],
            ['j', 'variable', 'type'],
            ['group']
        ],
        intcols=['iter', 'status', 'j', 'group'] # integer columns
    )
    
    # Export final reports of feature selection
    dfsrp.to_csv( # with duplicate entries
        f"{outdir}/{outsrepwdfile}",
        float_format="%.2f",
        header=True, index=False
    )
    dfsrpn.to_csv( # with nonduplicate entries
        f"{outdir}/{outsrepndfile}",
        sep=',', na_rep='',
        float_format="%.2f",
        header=True, index=False
    )

print(f"{dfsrp.head()}\n") # feature selection (with duplicate entries)
print(f"{dfsrpn.head()}\n") # feature selection (with nonduplicate entries)


# Export final reports of detailed decision regions (with duplicate/nonduplicate entries) (if specified)

if isrreport: # reports of detailed decision regions must be written

    # Export final reports of detailed regions (with duplicate entries)
    with open(f"{outdir}/{outrrepwdfile}", 'w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames = [
                'iter', 'reg',
                'ordvars', 'strvars',
                'ordreg', 'crossreg',
                'tpreds', 'strtpreds',
                'ninst'
            ])
        writer.writeheader()   
        for citer, tregs in ttregs.items():
            strvars = ', '.join(tsels[citer]['variables'])
            ps = tsels[citer]['ps']
            qs = [0]*len(ps) # base representation of numerical decision region
            js = tsels[citer]['js']
            for b, treg in tregs.items():
                grls = list() # list of group members
                for ind in range(len(ps)):
                    member = tcuts[citer][js[ind]]['groups'][qs[ind]]
                    if isinstance(member, pd._libs.interval.Interval): # Pandas interval
                        grls.append(itvtostr(member))
                    elif isinstance(member, set): # set
                        grls.append(settostr(member))
                    else:
                        raise TypeError("Cut intervals can be either Pandas intervals or sets")
                writer.writerow({
                    'iter': citer,
                    'reg': b,
                    'ordvars': f"({','.join([str(j) for j in js])})", # ordered pair of selected features
                    'strvars': strvars, # string of selected features
                    'ordreg': f"({','.join([str(q) for q in qs])})", # ordered pair of numerical region
                    'crossreg': ' x '.join(grls), # cross product of features in string format
                    'tpreds': ','.join([str(v) for v in treg['classes']]), # true predicted classes
                    'strtpreds': ', '.join([clabels[v] for v in treg['classes']]), # true predicted classes
                    'ninst': treg['ninst'] # number of training instances in region
                })
                for ind in range(len(ps)): # increment base representation of region for next for loop
                    qs[ind] += 1 # increment by 1
                    if qs[ind] > ps[ind]: qs[ind] = 0 # new leading one
                    else: break # same leading one
    
    # Export final reports of detailed regions (with nonduplicate entries)
    dfrrp = pd.read_csv(f"{outdir}/{outrrepwdfile}")
    dfrrpn = nondup(dfrrp, ndcols=[['iter', 'ordvars', 'strvars']], intcols=['iter'])
    dfrrpn.to_csv( # with nonduplicate entries
        f"{outdir}/{outrrepndfile}",
        sep=',', na_rep='',
        header=True, index=False
    )

print(f"{dfrrp.head()}\n") # detailed decision regions (with duplicate entries)
print(f"{dfrrpn.head()}\n")  # detailed decision regions (with nonduplicate entries)
