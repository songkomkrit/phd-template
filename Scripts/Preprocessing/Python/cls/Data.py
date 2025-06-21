import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

@pd.api.extensions.register_dataframe_accessor("data")
class Data:
    def __init__(self, pandas_obj, indep_dict):
        self.dataset = pandas_obj
        self.metadata = indep_dict
    
    def encodecat(self):
        cat_change = ""
        for attr in self.metadata.keys():
            if self.metadata[attr]['type'] == 'Categorical':
                le = LabelEncoder()
                le.fit(self.dataset[attr])
                self.dataset[attr] = list(le.transform(self.dataset[attr]).astype('int8'))
                newkeys = list()
                unseen = 0
                for strval in self.metadata[attr]['values'].keys():
                    try:
                        newkeys.append(int(le.transform([int(strval)])))
                    except ValueError: # for previously unseen labels
                        unseen -= 1
                        newkeys.append(unseen)
                if list(self.metadata[attr]['values'].keys()) != newkeys:
                    cat_change += attr+"\n"
                newdict = {key: val for key, val in zip(newkeys, self.metadata[attr]['values'].values())}
                self.metadata[attr]['values'] = newdict
        return cat_change[0:-1]
    
    def encodecont(self):
        pattern = r'(^|[^\w])(niu|universe)([^\w]|$)'   # raw string to prevent invalid escape sequence '\w'
        pattern = re.compile(pattern, re.IGNORECASE)
        cont_nonpos = ""
        for attr in self.metadata.keys():
            if self.metadata[attr]['type'] == 'Continuous':
                flag = False
                for strval in self.metadata[attr]['values'].keys():
                    if not flag:
                        try:
                            if int(strval) <= 0:
                                text = self.metadata[attr]['values'][strval]
                                matches = re.search(pattern, text.replace(',', ' '))
                                if bool(matches):
                                    flag = True
                                    cont_nonpos += attr+"\n"
                                    self.dataset[attr] = self.dataset[attr].apply(lambda v: 0 if v < 0 else v)
                                    break
                        except:
                            pass
                    if flag:
                        try:
                            if int(strval) <= 0:
                                self.metadata[attr]['values'].pop(strval, None)
                        except:
                            pass
                if flag:
                    self.metadata[attr]['values']['0'] = 'NIU'
        return cont_nonpos[0:-1]
