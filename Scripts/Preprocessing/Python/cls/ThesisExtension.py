import re
import pandas as pd

@pd.api.extensions.register_dataframe_accessor("thesis")
class ThesisExtension:
    def __init__(self, pandas_obj):
        #self._validate(pandas_obj, list(indep_dict.keys()) + ['COV'] + dep_attrs)
        self.dataset = pandas_obj
    
    '''
    @staticmethod
    def _validate(obj, cols):
        if any(x not in obj.columns for x in cols):
            raise AttributeError("Some attributes are missing")
    '''

    def select(self, cols):
        self.dataset.drop(self.dataset.columns.difference(cols), axis=1, inplace=True)

    def show_type(self, option='short'):
        if option.lower() == 'full':
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.dataset.dtypes)
        else:
            print(self.dataset.dtypes)
        
    @staticmethod
    def retype(ser):
        if all(ser.apply(lambda x: isinstance(x, int))):
            flag_int = True
        elif all(ser.apply(lambda x: x.is_integer())):
            flag_int = True
        else:
            flag_int = False
        
        if flag_int:
            if all(ser.apply(lambda x: x>=0)):
                if max(ser) <= 255:
                    return ser.astype('uint8')
                elif max(ser) <= 65535:
                    return ser.astype('uint16')
                else:
                    return ser.astype('uint32')
            else:
                if min(ser) >= -128 and max(ser) <= 127:
                    return ser.astype('int8')
                elif min(ser) >= -32768 and max(ser) <= 32767:
                    return ser.astype('int16')
                else:
                    return ser.astype('int32')
        else:
            return ser.astype('float32')        

    def code(self, indep_dict, dep_attrs):  
        self.select(list(indep_dict.keys()) + ['COV'] + dep_attrs)
        for v in indep_dict.keys():
            if indep_dict[v]['type'] == 'Categorical':
                self.dataset[v] = self.dataset[v].astype('int8').astype('category')
            else:
                self.dataset[v] = self.retype(self.dataset[v])
        self.dataset['COV'] = self.dataset['COV'].astype('int8').astype('category')
        self.dataset[dep_attrs] = self.dataset[dep_attrs].astype('int8')
        self.dataset['class_orig'] = 0
        self.dataset['code_orig'] = ""
        for v in dep_attrs:
        	self.dataset[v] = self.dataset[v].replace([2.0, 1.0], [False, True])
        	self.dataset['class_orig'] = 2*self.dataset['class_orig'] + self.dataset[v]
        	self.dataset['code_orig'] = self.dataset['code_orig'] + self.dataset[v].replace([True, False], ['Y', 'N'])
        self.dataset[dep_attrs] = self.dataset[dep_attrs].astype('category')
        self.dataset['class_orig'] = self.dataset['class_orig'].astype('int8').astype('category')
        self.dataset['code_orig'] = self.dataset['code_orig'].astype('category')

    def recode(self):
        self.dataset['code'] =  self.dataset['code_orig'].apply(
            lambda v: 'NY_' if re.match('(NY)', v) 
            else 'Y1Y' if re.match(r'^Y(?:\w*Y)', v)    # raw string to prevent invalid escape sequence '\w'
            else v
        ).astype('category')
        self.dataset['class'] = self.dataset[['class_orig', 'code']].apply(
            lambda v: 2 if v['code'] == 'NY_'
            else 3 if v['code'] == 'YNN'
            else 4 if v['code'] == 'Y1Y'
            else v['class_orig'], 
            axis=1
        ).astype('int8').astype('category')
