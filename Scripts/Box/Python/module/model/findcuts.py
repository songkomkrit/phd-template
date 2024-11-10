import numpy as np
import pandas as pd

# Find cuts and groups
def findcuts(tsels, itcont, itcat, intvclosed='neither', intvsubtype='float32'):
    '''
        Usage: find cuts and groups (per file)
        Required arguments:
            tsels: dictionary of selected variables and given number of cuts
            itcont: full continuous cuts (DataFrame iterator)
            itcat: full categiorical cuts (DataFrame iterator)
        Optional arguments:
            intvclosed: types of Pandas interval sides (values: 'left', 'right', 'both', 'neither')
            intvsubtype: types of Pandas interval bounds (subtype of pandas.IntervalDtype)            
        Outputs:
            tcuts: dictionary of cuts and groups along all selected features
    '''

    ccontrow = next(itcont) # iterator of full continuous cuts across all iterations
    ccatrow = next(itcat) # iterator of full categorical cuts across all iterations
    tcuts = dict() # cuts and groups along all selected features

    for citer, sel in tsels.items(): # cuts across all selected features
        tcuts[citer] = dict()
        for ind, j in enumerate(sel['js']):
            tcuts[citer][j] = {
                'variable': tsels[citer]['variables'][ind],
                'type': tsels[citer]['types'][ind],
                'cuts': list(),
                'groups': dict()
            }

        # Cuts
        while ccontrow.iter < citer: # previous iteration may select no continuous feature
            ccontrow = next(itcont)
        while ccatrow.iter < citer: # previous iteration may select no categorical feature
            ccatrow = next(itcat)
        for jcur in sorted(sel['js']): # numerically sorted features selected
            cuts = tcuts[citer][jcur]['cuts'] # list of cuts along specific selected feature
            try: # iterate over full continuous cuts
                while ccontrow.iter == citer:
                    if ccontrow.j > jcur: # seek no more than current feature
                        break
                    else:
                        if ccontrow.j == jcur: # at current selected feature
                            cuts.append(ccontrow.bc) # continuous feature seen
                        ccontrow = next(itcont) # update DataFrame iterator
            except StopIteration:
                pass
            try: # iterate over full categorical cuts
                while ccatrow.iter == citer:
                    if ccatrow.j > jcur: # seek no more than current feature
                        break
                    else:
                        if ccatrow.j == jcur: # at current selected feature
                            cuts.append(ccatrow.v) # categorical feature seen
                        ccatrow = next(itcat) # update DataFrame iterator
            except StopIteration:
                pass 
        
        # Groups
        pcutdc = dict(zip(tsels[citer]['js'], tsels[citer]['ps'])) # cut numbers along selected features
        for j, info in tcuts[citer].items():
            pnum = pcutdc[j] # number of cuts current selected feature
            cuts = info['cuts']
            if info['type'] == 'cont': # continuous feature
                excuts = [-np.inf] + cuts + [np.inf]
                intvs = pd.arrays.IntervalArray.from_breaks(
                    breaks=excuts,
                    copy=False, # default: False
                    closed=intvclosed, # types of Pandas interval sides
                    dtype=pd.IntervalDtype(subtype=intvsubtype) # types of Pandas interval bounds
                )
                info['groups'] = {gr: intvs[gr] for gr in range(pnum+1)}
            else: # categorical feature
                info['groups'] = {gr: set() for gr in range(pnum+1)}
                for val, gr in enumerate(cuts):
                    info['groups'][gr].add(val) # categorical value in cut/group

    return tcuts 
