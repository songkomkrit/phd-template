import numpy as np
import pandas as pd

# Typecast object to integer NumPy array
def tointnp(obj, intdtype=np.int16):
    '''
        Usage: typecast object to integer NumPy array
        Required arguments:
            obj: iterable object
            intdtype: NumPy integer data type
        Outputs: integer NumPy array
    '''
    
    try:
        if obj.dtype == intdtype: return obj
    except:
        try: return np.array(obj, dtype=intdtype)
        except: return np.array(map(int, obj), dtype=intdtype)


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
def itvtostr(itv, decimals=2):
    '''
        Usage: convert Pandas interval to string
        Required arguments:
            itv: Pandas interval
        Optional arguments:
            decimals: number of decimal places (default: 2)
        Outputs: string interval
    '''
    
    if isinstance(itv, pd._libs.interval.Interval):
        l = f"{itv.left:.{decimals}f}"
        r = f"{itv.right:.{decimals}f}"
        if itv.closed == 'neither': return f"({l}, {r})"
        elif itv.closed == 'left': return f"[{l}, {r})"
        elif itv.closed == 'right': return f"({l}, {r}]"
        else: return f"[{l}, {r}]"
    else:
        raise TypeError("Only Pandas intervals are allowed")
