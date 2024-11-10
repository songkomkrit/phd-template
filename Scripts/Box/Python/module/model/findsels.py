# Find feature selection
def findsels(itsel, pcuto):
    '''
        Usage: find feature selection (per file)
        Required arguments:
            itsel: selected string variables (DataFrame iterator)
            pcuto: old cut numbers         
        Outputs:
            tsels: dictionary of selected variables and given number of cuts
    '''
    
    csrow = next(itsel) # iterator of selected string variables across all iterations
    tsels = dict() # selected variables and given number of cuts

    citer = -1 # current iteration
    while True:
        try:
            if csrow.aselect == 1: # for selected variable
                if csrow.iter != citer:
                    citer = csrow.iter
                    tsels[citer] = {
                        'variables': list(), # selected feature
                        'types': list(), # type of selected feature
                        'js': list(), # selected index
                        'ps': list() # given cut number
                    }
                tsels[citer]['variables'].append(csrow.variable)
                tsels[citer]['types'].append(csrow.type)
                tsels[citer]['js'].append(csrow.jnew)
                tsels[citer]['ps'].append(pcuto[csrow.jnew-1])
            csrow = next(itsel) # update DataFrame iterator
        except StopIteration:
            break

    return tsels
