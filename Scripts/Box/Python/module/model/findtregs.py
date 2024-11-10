import numpy as np
import pandas as pd

from module.operation.xutil import max_dictval, itvpos


# Calculate new true decision regions and predictions (truly correct)
def findtregs(tsels, tcuts, df, pdtype=np.int16):
    '''
        Usage: calculate new true decision regions and predictions (per file)
        Required arguments:
            tsels: dictionary of selected variables and given number of cuts
            tcuts: dictionary of cuts and groups along all selected features
            df: training dataset including target variable (DataFrame, not iterator)
        Optional arguments:
            pdtype: NumPy data type of cut number (default: np.int16)           
        Outputs:
            ttregs: dictionary of new true decision regions and their predicted classes
    '''

    ttregs = dict() # new true regions with predicted classes (truly correct)
    classes = df['class'].unique() # all possible classes
    
    for citer in tsels.keys():
        regs = pd.Series([0]*len(df))
        js = tsels[citer]['js']
        pcutn = np.array(tsels[citer]['ps'], dtype=pdtype) # new cut numbers
        pncum = np.cumprod(np.append([1], pcutn[0:-1]+1), dtype=pdtype) # cumulative number of new box regions
        BN = np.prod(pcutn+1) # number of new regions

        # Convert base representation of decision region to base 10
        for ind, j in enumerate(js):
            info = tcuts[citer][j]
            attr = info['variable']
            cuts = info['cuts']
            if info['type'] == 'cont': # continuous feature
                regs = regs + pncum[ind]*df[attr].apply(lambda x: itvpos(x, cuts))
            else: # categorical feature
                regs = regs + pncum[ind]*pd.Series([cuts[x] for x in df[attr]])

        # Find predicted classes in decision regions
        ttregs[citer] = {
            b: {
                'classes': set(), # true predicted class set
                'correct': 0, # number of instances correctly predicted
                'ninst': 0, # number of training instances (total)
                'ncinst': {n: 0 for n in range(len(classes))} # number of training instances in targets
            } for b in range(BN)
        }
        for i in range(len(df)):
            ttregs[citer][regs[i]]['ninst'] += 1 # instance in region
            ttregs[citer][regs[i]]['ncinst'][df['class'][i]] += 1 # instance of specific target in region
        for b in range(BN):
            kmax, vmax = max_dictval(ttregs[citer][b]['ncinst']) # true majority voting
            ttregs[citer][b]['classes'] = kmax # all classes that have maximum number of instances
            ttregs[citer][b]['correct'] = vmax # maximum number of instances
            
    return ttregs
