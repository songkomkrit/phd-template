# Per-File Operations
import numpy as np

from module.typecast import strtoset
from module.calregs import calregs


# Calculate new corresponding decision regions and predictions, and also provide cuts and groups (per file)
def predregs(iterr, itsel, itreg, itcont, itcat, n_classes, pdtype=np.int16, idtype=np.int8):
    '''
        Usage: calculate new corresponding decision regions and predictions, and also provide cuts and groups (per file)
        Required arguments:
            iterr: classification errors (DataFrame iterator)
            itsel: selected string variables (DataFrame iterator)
            itreg: full decision regions (DataFrame iterator)
            itcont: full continuous cuts (DataFrame iterator)
            itcat: full categiorical cuts (DataFrame iterator)
            n_classes: number of total classes (labeled by 0, 1, ..., n_classes-1)
        Optional arguments:
            pdtype: data type of cut number (default: np.int16)
            idtype: data type of index (default: np.int8)
        Outputs:
            tsels: dictionary of selected variables and given number of cuts
            tpreds: dictionary of new predicted classes in all new decision regions
            tcuts: dictionary of cuts along all selected features and groups along categorical features
    '''

    # Input iterators
    csrow = next(itsel) # selected string variables (across all iterations)
    crrow = next(itreg) # full decision regions (across all iterations)
    ccontrow = next(itcont) # full continuous cuts (across all iterations)
    ccatrow = next(itcat) # full categorical cuts (across all iterations)

    # Initialized dictionary outputs
    tsels = dict() # selected variables and given number of cuts
    tpreds = dict() # new predicted classes in all new regions
    tcuts = dict() # cuts and groups along all selected features

    # Iterate over all iterations
    for erow in iterr:
        # erow: Index, iter, pcut's, error, ..., relgap
        pcuto = np.array(erow[2:-7], dtype=pdtype) # old cut numbers
        scuts = dict() # dictionary of cuts along selected features
        
        # Selected variables and given number of cuts
        sidl = list() # list of selected indexes 
        svars = list() # list of selected features
        stypes = list() # list of types of selected features
        try:
            while csrow.iter == erow.iter:
                if csrow.aselect == 1: # for selected variable
                    sidl.append(csrow.jnew) # selected index
                    svars.append(csrow.variable) # selected feature
                    stypes.append(csrow.type) # type of selected feature
                    # Initialize cuts along specific selected feature
                    scuts[csrow.jnew] = {'variable': csrow.variable,
                                         'type': csrow.type,
                                         'cuts': list(),
                                         'groups': dict()}
                # Update DataFrame iterator of selected string variables
                csrow = next(itsel)
        except StopIteration:
            pass
        sidx = np.array(sidl, dtype=idtype) - 1 # index starts at 0
        pcutn = pcuto[sidx] # new cut numbers
        tsels[erow.iter] = {'variables': svars, 'types': stypes,
                            'js': sidl, 'ps': pcutn.tolist()}

        # New corresponding decision regions
        BN = np.prod(pcutn+1) # number of new regions     
        bns = calregs(pcuto, sidx) # new correspoding regions

        # Predicted classes in all new decision regions
        preds = {k: {'occupy': False,
                     'classes': set()} for k in range(BN)}
        try:
            while crrow.iter == erow.iter:
                if crrow.occupy == 1: # for old occupied region
                    cpred = preds[bns[crrow.region]] # predicted classes in specific region
                    cpred['occupy'] = True
                    cpred['classes'] = cpred['classes'].union(strtoset(crrow.predict)) # allow for more classes
                # Update DataFrame iterator of full decision regions
                crrow = next(itreg)
        except StopIteration:
            pass
        for val in preds.values():
            if not val['occupy']: # for new, still unoccupied, region
                val['classes'] = set(range(n_classes))
        tpreds[erow.iter] = preds # store predicted classes in all regions

        # Cuts
        while ccontrow.iter < erow.iter: # previous iteration may select no continuous feature
            ccontrow = next(itcont)
        while ccatrow.iter < erow.iter: # previous iteration may select no categorical feature
            ccatrow = next(itcat)
        for jcur in (ssidl:=sorted(sidl)): # numerically sorted features selected
            cuts = scuts[jcur]['cuts'] # list of cuts along specific selected feature
            try:
                while ccontrow.iter == erow.iter:
                    if ccontrow.j < jcur: # seek before current feature
                        ccontrow = next(itcont)
                    elif ccontrow.j == jcur: # seek up to current feature
                        cuts.append(ccontrow.bc) # continuous feature seen
                        ccontrow = next(itcont)
                    elif jcur == ssidl[-1]: # last selected feature
                        ccontrow = next(itcont)
                    else: # seek no more than current (except last) feature
                        break           
            except StopIteration:
                pass
            try:
                while ccatrow.iter == erow.iter:
                    if ccatrow.j < jcur: # seek before current feature
                        ccatrow = next(itcat)
                    elif ccatrow.j == jcur: # seek up to current feature
                        cuts.append(ccatrow.v) # categorical feature seen
                        ccatrow = next(itcat) 
                    elif jcur == ssidl[-1]: # last selected feature
                        ccatrow = next(itcat)
                    else: # seek no more than current (except last) feature
                        break
            except StopIteration:
                pass 
        
        # Groups (only for categorical features)
        for j, info in scuts.items():
            if info['type'] == 'cat': # categorical feature
                info['groups'] = {gr: set() for gr in range(pcuto[j-1]+1)}
                for val, gr in enumerate(info['cuts']):
                    info['groups'][gr].add(val) # categorical value in cut/group
        
        # Store cuts and groups along all selected features
        tcuts[erow.iter] = scuts
    
    # Outputs
    return tsels, tpreds, tcuts
