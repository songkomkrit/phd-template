import pandas as pd

# Delete the accessor to avoid warning 
try:
    del pd.DataFrame.info
except AttributeError:
    pass

@pd.api.extensions.register_dataframe_accessor("info")
class Info:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj, ['id', 'variable', 'type', 'min', 'max'])
        self.dataset = pandas_obj
    
    @staticmethod
    def _validate(obj, cols):
        if any(x not in obj.columns for x in cols):
            raise AttributeError("Some attributes are missing")

    def setcut(self, pcont, pcatmax):
        self.dataset['cut'] = 0
        self.dataset.loc[self.dataset['type'] == 'Continuous', 'cut'] = pcont
        self.dataset.loc[self.dataset['type'] == 'Categorical', 'cut'] = self.dataset['max'].map(lambda v: min(v, pcatmax))
