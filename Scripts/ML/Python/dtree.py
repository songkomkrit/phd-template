import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

def create_dir(dir):
    try:
       os.makedirs(dir)
    except FileExistsError:
       pass

# Given Information
data_ls = []
data_ls.append(
    {
        'data': "../../../Samples/selproc20/train/seltrain20num3each20.csv",
        'info': "../../../Samples/selproc20/info/selproc20num3info.csv",
        'depths': [3, 4],
        'outdir': "../../../Outputs/Main/Tree",
        'prefixout': "seltrain20num3each20"
    }
)
print(f"{data_ls}\n")

# Decision Tree
def dtree(df_data, df_info, depth, data_path='', info_path=''):

    # One-hot encoding
    feat_cat = list(df_info[df_info['type'] == 'Categorical']['variable'])
    for v in feat_cat:
        df_data[v] = df_data[v].astype('category')
    one_hot_data = pd.get_dummies(df_data[feat_cat], drop_first=True)
    X = df_data.iloc[:,0:-(len(feat_cat)+1)].join(one_hot_data)
    y = df_data['class']

    # Build decision tree
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X, y)

    # Performance metrics (summary)
    score = clf.score(X, y)
    y_pred = clf.predict(X)
    err_ind = (y_pred != y.to_numpy().flatten()).astype(int)
    error = np.count_nonzero(err_ind)
    accuracy = (1-error/len(y_pred))*100
    summary = {'error': error, 'accuracy': accuracy, 'score': score}

    # Predicted values
    df_pred = pd.DataFrame({'y_true': df_data['class'], 'y_pred': y_pred, 'e': err_ind})

    # Display results
    if data_path != '':
        print(f"Data: {data_path}")
    if info_path != '':
        print(f"Info: {info_path}")
    print(f"Depth: {depth}\n")
    print(f"Categorical features: {feat_cat}")
    print(f"X: {X.columns.values}\n")
    print(f"Summary: Error = {error} | Accuracy = {accuracy} | Score = {score}\n")

    # Return statement
    return clf, summary, df_pred

# Implementation
for dc in data_ls:

    # Import
    df_data = pd.read_csv(dc['data'])
    df_info = pd.read_csv(dc['info'])

    # Exported figure formats
    fig_formats = ['svg', 'pgf', 'pdf']
    
    # Create directories
    create_dir(f"{dc['outdir']}/summary")
    create_dir(f"{dc['outdir']}/prediction")
    for format in fig_formats:
        create_dir(f"{dc['outdir']}/figures/{format}")

    for depth in dc['depths']:

        # Decision tree
        clf, summary, df_pred = dtree(df_data, df_info, depth, data_path=dc['data'], info_path=dc['info'])

        # Summary
        outsumfile = f"{dc['outdir']}/summary/{dc['prefixout']}-sum.csv"
        sumheader = ['depth', 'error', 'accuracy', 'score']
        summary['depth'] = depth
        with open(outsumfile, 'a') as file:
            writer = csv.DictWriter(file, fieldnames=sumheader)
            writer.writeheader()
            writer.writerow(summary)

        # Prediction
        outpredfile = f"{dc['outdir']}/prediction/{dc['prefixout']}-pred-depth-{depth}.csv"
        df_pred.index = df_pred.index + 1
        df_pred.to_csv(outpredfile, index_label='id')

        # Tree
        tree.plot_tree(clf)
        #tree.plot_tree(clf, label='none', impurity=False)
        for format in fig_formats:
            outfigfile = f"{dc['outdir']}/figures/{format}/{dc['prefixout']}-fig-depth-{depth}.{format}"
            plt.savefig(outfigfile, bbox_inches='tight')
        plt.show()

        # Newline
        print()
