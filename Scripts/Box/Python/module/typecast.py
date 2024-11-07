import numpy as np


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
        try:
            return np.array(obj, dtype=intdtype)
        except:
            return np.array(map(int, obj), dtype=intdtype)


# Convert set/number in string format to Python set
def strtoset(setstr):
    '''
        Usage: convert set/number in string format to Python set
        Required arguments:
            setstr: set/number in string format
        Outputs: corresponding set
    '''
    
    strset = set(setstr.strip().strip('{ }'))
    try:
        strset.remove(' ') # for set of more than two elements
    except:
        pass
    numset = set(map(int, strset))
    
    return numset
