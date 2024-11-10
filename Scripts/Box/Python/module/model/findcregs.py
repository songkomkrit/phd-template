import numpy as np

from module.operation.typecast import strtoset
from module.operation.calregs import calregs


# Calculate new cplex decision regions and predictions (partially correct)
def findcregs(tsels, itpred, pcuto, idtype=np.int16, pdtype=np.int16):
    '''
        Usage: calculate new cplex decision regions and predictions (per file)
        Required arguments:
            tsels: dictionary of selected variables and given number of cuts
            itpred: individual result of cplex prediction (DataFrame iterator)
            pcuto: old cut numbers
        Optional arguments:
            pdtype: NumPy data type of cut number (default: np.int16)
            idtype: NumPy data type of index (default: np.int16)
        Outputs:
            tcregs: dictionary of new cplex decision regions and their predicted classes
    '''
    
    cprow = next(itpred) # iterator of instance predictions across all iterations
    tcregs = dict() # new cplex regions with predicted classes (partially correct)
    classes = set() # set all possible classes (collected from training dataset)
    
    citer = -1 # current iteration
    
    while True: # reported by cplex as occupied region
        try:
            if cprow.iter != citer: # new iteration
                citer = cprow.iter
                if citer in tsels.keys(): # current iteration actually selects at least one feature
                    keep = True # keep doing in this while loop
                    pcutn = np.array(tsels[citer]['ps'], dtype=pdtype)
                    sidx = np.array(tsels[citer]['js'], dtype=idtype) - 1 # index starts at 0
                    BN = np.prod(pcutn+1) # number of new regions     
                    bns = calregs(pcuto, sidx) # new correspoding regions
                    tcregs[citer] = {
                        b: {
                            'lclasses': list(), # list of cplex predicted class set
                            'nlcinst': list() # list of instance number in corresponding cplex class set
                        } for b in range(BN)
                    }
                else: # current iteration selects no feature
                    keep = False # update iterator and go to the next while loop
            if keep and cprow.iter == citer: # every record in iteration that selects feature
                creg = tcregs[citer][bns[cprow.region]] # new cplex region
                pset = strtoset(cprow.predict) # current set of classes predicted by cplex
                classes = classes.union(pset) # add to set of all possible classes
                try: # current set of predicted classes already exists
                    creg['nlcinst'][creg['lclasses'].index(pset)] += 1
                except ValueError: # new set of predicted classes
                    creg['lclasses'].append(pset)
                    creg['nlcinst'].append(1)
            cprow = next(itpred) # update DataFrame iterator  
        except StopIteration:
            break
    
    for cregs in tcregs.values(): # reported by cplex as unoccupied region
        for creg in cregs.values():
            if not creg['lclasses']:
                creg['lclasses'] = [classes] # predict only one of the entire set
                nlcinst = [0] # no instance reported by cplex in the rest of new regions
    
    return tcregs
