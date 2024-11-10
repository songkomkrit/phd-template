# Find both true and recalculated cplex correctness
def findcorr(ttregs, tcregs):
    '''
        Usage: find both true and recalculated cplex correctness (per file)
        Required arguments:
            ttregs: dictionary of new true decision regions and their predicted classes
            tcregs: dictionary of new cplex decision regions and their predicted classes      
        Outputs:
            tcorr: true number of correctly classified instances per region
            ccorr: recalculated cplex number of correctly classified instances per region
    '''

    tcorr = dict() # true correctness
    ccorr = dict() # cplex correctness    
    for citer, tregs in ttregs.items(): # true classification
        tcorr[citer] = {
            'correct': 0,
            'detail': {b: tregs[b]['correct'] for b in tregs.keys()}
        }
        tcorr[citer]['correct'] = sum(tcorr[citer]['detail'].values())       
    for citer, cregs in tcregs.items(): # cplex classification
        ccorr[citer] = {
            'correct': 0,
            'detail': {b: 0 for b in cregs.keys()}
        }
        for b in cregs.keys():
            for soc in tcregs[citer][b]['lclasses']:
                ccorr[citer]['detail'][b] = max([ttregs[citer][b]['ncinst'][c] for c in soc])
        ccorr[citer]['correct'] = sum(ccorr[citer]['detail'].values())
    
    return tcorr, ccorr
