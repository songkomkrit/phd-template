import numpy as np


# Calculate new corresponding region label (helper)
def hcalbn(bo, bnprev, idxn, pcuto, pocum, pncumx):
    '''
        Usage: calculate new corresponding region label (helper)
        Required arguments:
            bo: region label for old features (nonzero)
            bnprev: previous region label for new features
            idxn: new feature indexes
            pcuto: old cut numbers
            pocum: cumulative number of box regions across old features
            pncumx: cumulative number of extrended box regions across new features
        Outputs: corresponding region label
    '''
    
    # bo must be between 1 and np.prod(pcuto+1)-1
    bn = bnprev
    for jmax in range(len(pcuto)-1,-1,-1):
        # bo (incremented by 1) in base representation has the last nonzero at digit jmax
        if bo%pocum[jmax] == 0:
            for j in range(jmax):
                bn -= pcuto[j]*pncumx[idxn[j]]
            bn += pncumx[idxn[jmax]]
            break
    
    return bn


# Calculate corresponding decision regions (helper)
def hcalregs(BO, idxn, pcuto, pocum, pncumx):
    '''
        Usage: calculate corresponding decision regions (helper)
        Required arguments:
            BO: total number of old box regions
            idxn: new feature indexes
            pcuto: old cut numbers
            pocum: cumulative number of box regions across old features
            pncumx: cumulative number of extrended box regions across new features
        Outputs: corresponding region label
    '''
    
    bns = [0] # list of corresponding box regions (region 0)
    for bo in range(1, BO):
        bnprev = bns[-1] 
        bn = hcalbn(bo, bnprev, idxn, pcuto, pocum, pncumx)
        bns.append(bn)
    
    return bns


# Calculate new corresponding decision regions (main)
def calregs(pcuto, sidx, pdtype=np.int16, idtype=np.int16, rdtype=np.int16):
    '''
        Usage: calculate new corresponding decision regions (main)
        Required arguments:
            pcuto: old cut numbers
            sidx: selected feature indexes (in order)
        Optional arguments:
            pdtype: NumPy data type of cut number (default: np.int16)
            idtype: NumPy data type of index (default: np.int16)
            rdtype: NumPy data type of region number (default: np.int16)
        Outputs: new correspoding regions
    '''
    
    # Typecasting
    pcuto = np.array(pcuto, dtype=pdtype)
    sidx = np.array(sidx, dtype=idtype)
    
    # Basic calculation
    dimo = pcuto.size # old dimension
    dimn = sidx.size # new dimension
    pcutn = pcuto[sidx] # new cut numbers
    BO = np.prod(pcuto+1).astype(rdtype) # number of old regions
    BN = np.prod(pcutn+1).astype(rdtype) # number of new regions
    
    # New feature indexes
    idxn = np.full(dimo, -1, dtype=idtype)
    idxn[sidx] = np.arange(dimn, dtype=idtype)
    idxn[idxn < 0] = np.arange(dimn, dimo, dtype=idtype)
    
    # Cumulative number of box regions
    pocum = np.cumprod(np.append([1], pcuto[0:-1]+1), dtype=rdtype) # old
    pncum = np.cumprod(np.append([1], pcutn[0:-1]+1), dtype=rdtype) # new
    pncumx = np.concatenate((pncum, np.zeros(dimo-dimn, dtype=rdtype))) # new and extended
    
    # New correspoding regions (helper function called)
    bns = np.array(hcalregs(BO, idxn, pcuto, pocum, pncumx), dtype=rdtype)

    # Output
    return bns


# Illustration
'''
print('pcuto: {0}\nsidx: {1}\nbns: {2}\n'.format(pcuto:=[3, 4], sidx:=[0], calregs(pcuto, sidx)))
print('pcuto: {0}\nsidx: {1}\nbns: {2}\n'.format(pcuto:=[3, 4], sidx:=[1], calregs(pcuto, sidx)))
print('pcuto: {0}\nsidx: {1}\nbns: {2}\n'.format(pcuto:=[3, 4], sidx:=[0, 1], calregs(pcuto, sidx)))
print('pcuto: {0}\nsidx: {1}\nbns: {2}\n'.format(pcuto:=[3, 4], sidx:=[1, 0], calregs(pcuto, sidx)))
'''
