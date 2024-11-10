import numpy as np
import pandas as pd

from module.operation.xutil import itvtopts


# Convert set/number in string format to Python set
def strtoset(setstr):
    '''
        Usage: convert set/number in string format to Python set
        Required arguments:
            setstr: set/number in string format
        Outputs: corresponding set
    '''
    
    strset = set(setstr.strip().strip('{ }'))
    try: strset.remove(' ') # for set of more than two elements
    except: pass
    numset = set(map(int, strset))
    
    return numset


# Convert set to string
def settostr(st, sep=',', left='{', right='}'):
    '''
        Usage: convert set to string
        Required arguments:
            st: set
        Optional arguments:
            sep: separator (default: ',')
            left: left symbol (default: '{')
            right: right symbol (default: '}')
        Outputs: string representing given set
    '''
    
    stre = sep.join([str(e) for e in st])
    
    return f"{left}{stre}{right}"


# Convert Pandas interval to string
def itvtostr(itv, decimals=2, extend=True):
    '''
        Usage: convert Pandas interval to string
        Required arguments:
            itv: Pandas interval
        Optional arguments:
            decimals: number of decimal places to round to (default: 2)
            extend: whether extend (true) or shrink (default) interval (default: True)
        Outputs: string interval
    '''

    lpt, rpt = itvtopts(itv, decimals, extend)
    l = f"{lpt:.{decimals}f}"
    r = f"{rpt:.{decimals}f}"

    if itv.closed == 'neither': return f"({l}, {r})"
    elif itv.closed == 'left': return f"[{l}, {r})"
    elif itv.closed == 'right': return f"({l}, {r}]"
    else: return f"[{l}, {r}]"


# Describe Pandas interval in text format
def itvtodesc(itv, decimals=2, extend=True):
    '''
        Usage: describe Pandas interval in text format
        Required arguments:
            itv: Pandas interval
        Optional arguments:
            decimals: number of decimal places to round to (default: 2)
            extend: whether extend (true) or shrink (default) interval (default: True)
        Outputs: description of interval in text format
    '''

    lpt, rpt = itvtopts(itv, decimals, extend)
    l = f"{lpt:.{decimals}f}"
    r = f"{rpt:.{decimals}f}"
    
    esum = itv.left + itv.right
    if np.isnan(esum): # -np.inf, np.inf
        return "any number"
    elif not np.isinf(esum): # num, num
        return f"between {l} and {r}"
    elif esum < 0: # -np.inf, num
        return f"below {r}"
    else: # num, np.inf
        return f"above {l}"
