# Per-File Operations
import numpy as np

from module.typecast import strtoset
from module.calregs import calregs


# Calculate new corresponding decision regions and predictions (per file)
def predregs(iterr, itsel, itreg, n_classes, pdtype=np.int16, idtype=np.int8):
    '''
        Usage: calculate new corresponding decision regions and predictions (per file)
        Required arguments:
            iterr: classification errors (DataFrame iterator)
            itsel: selected string variables (DataFrame iterator)
            itreg: full decision regions (DataFrame iterator)
            n_classes: number of total classes (labeled by 0, 1, ..., n_classes-1)
        Optional arguments:
            pdtype: data type of cut number (default: np.int16)
            idtype: data type of index (default: np.int8)
        Outputs:
            tsels: dictionary of selected variables
            tpreds: dictionary of predicted classes in all decision regions
    '''
    
    csrow = next(itsel) # selected string variables (across all iterations)
    crrow = next(itreg) # full decision regions (across all iterations)
    tsels = dict()
    tpreds = dict()
    
    for erow in iterr:
        # erow: Index, iter, pcut's, error, ..., relgap
        pcuto = np.array(erow[2:-7], dtype=pdtype) # old cut numbers
        sidl = [] # list of selected indexes 
        svars = [] # list of selected features
        stypes = [] # list of types of selected features
    
        try:
            while csrow.iter == erow.iter:
                if csrow.aselect == 1: # for selected variable
                    sidl.append(csrow.jnew)
                    svars.append(csrow.variable)
                    stypes.append(csrow.type)
                csrow = next(itsel) # update DataFrame iterator of selected string variables
        except StopIteration:
            pass

        # Selected variables
        tsels[erow.iter] = {'variables': svars,
                            'types': stypes,
                            'js': sidl}

        # New corresponding decision regions
        sidx = np.array(sidl, dtype=idtype) - 1 # Index starts at 0
        pcutn = pcuto[sidx] # new cut numbers
        BN = np.prod(pcutn+1) # number of new regions     
        bns = calregs(pcuto, sidx) # new correspoding regions

        # Predicted classes in all new decision regions
        preds = {k: {
            'occupy': False,
            'classes': set()} for k in range(BN)
        }
        
        try:
            while crrow.iter == erow.iter:
                if crrow.occupy == 1: # for old occupied region
                    cpred = preds[bns[crrow.region]] # predicted classes in specific region
                    cpred['occupy'] = True
                    cpred['classes'] = cpred['classes'].union(strtoset(crrow.predict)) # allow for more classes
                crrow = next(itreg) # update DataFrame iterator of full decision regions
        except StopIteration:
            pass
        
        for val in preds.values():
            if not val['occupy']: # for new, still unoccupied, region
                val['classes'] = set(range(n_classes))
        tpreds[erow.iter] = preds # store predicted classes in specific region

    return tsels, tpreds
