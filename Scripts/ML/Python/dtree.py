import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

def create_dir(dir):
    try:
       os.makedirs(dir)
    except FileExistsError:
       pass

# Given Information
data_ls = []
data_ls.append({
    'data': "../../../Samples/cplex/seltrain20num3each20.csv",
    'info': "../../../Samples/cplex/selproc20num3co3ca3cutinfo.csv",
    'configs': [
        {'max_depth': 3, 'max_leaves': 16},
        {'max_depth': 4, 'max_leaves': 16},
        {'max_depth': 5, 'max_leaves': 16}
    ],
    'outdir': "../../../Outputs/Main/Tree"
})
print(f"{data_ls}\n")

# Decision Tree
def dtree(df_data, df_info, max_depth, max_leaves, data_path='', info_path=''):

    # One-hot encoding
    feat_cat = list(df_info[df_info['type'] == 'Categorical']['variable'])
    for v in feat_cat:
        df_data[v] = df_data[v].astype('category')
    one_hot_data = pd.get_dummies(df_data[feat_cat], drop_first=True)
    X = df_data.iloc[:,0:-(len(feat_cat)+1)].join(one_hot_data)
    y = df_data['class']

    # Build decision tree
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaves,
        random_state=0
    )
    clf.fit(X, y)

    # Performance
    score = clf.score(X, y)
    y_pred = clf.predict(X)
    err_ind = (y_pred != y.to_numpy().flatten()).astype(int)
    error = np.count_nonzero(err_ind)
    accuracy = (1-error/len(y_pred))*100

    # Tree structure
    depth = clf.tree_.max_depth
    nodes = clf.tree_.node_count
    leaves = clf.tree_.n_leaves
    splits = nodes - leaves

    # Decision tree summary
    summary = {
        'error': error, 'accuracy': accuracy, 'score': score,
        'depth': depth,
        'nodes': nodes, 'leaves': leaves, 'splits': splits
    }

    # Decision rules
    rules = export_text(clf, feature_names=list(X.columns))

    # Predicted values
    df_pred = pd.DataFrame({
        'y_true': df_data['class'],
        'y_pred': y_pred, 
        'e': err_ind
    })

    # Display results
    if data_path != '':
        print(f"Data: {data_path}")
    if info_path != '':
        print(f"Info: {info_path}")
    print(f"Maximum depth: {max_depth}")
    print(f"Maximum number of leaves: {max_leaves}\n")
    print(f"Categorical features: {feat_cat}")
    print(f"X: {X.columns.values}\n")
    print(f"Summary:")
    print(f"\tDepth = {depth} | Leaves = {leaves}")
    print(f"\tError = {error} | Accuracy = {accuracy} | Score = {score}")
    print(f"\tNodes = {nodes} | Splits = {splits}\n")
    print(f"Decision rules:\n{rules}\n")

    # Return statement
    return clf, summary, rules, df_pred

# Implementation
for dc in data_ls:

    # Export information
    datname = os.path.splitext(os.path.basename(dc['data']))[0] # without file extension
    outdatdir = f"{dc['outdir']}/{datname}"
    outprefix = datname
    outsumfile = f"{outdatdir}/{outprefix}-summary.csv"
    outruledir = f"{outdatdir}/rules"
    outpreddir = f"{outdatdir}/prediction"
    outfigdir = f"{outdatdir}/figures"
    
    # Import
    df_data = pd.read_csv(dc['data'])
    df_info = pd.read_csv(dc['info'])

    # Exported figure formats
    fig_formats = ['svg', 'pgf', 'pdf']
    
    # Create directories
    create_dir(f"{outdatdir}/rules")
    create_dir(f"{outdatdir}/prediction")
    for format in fig_formats:
        create_dir(f"{outdatdir}/figures/{format}")

    # Export summary file in CSV format
    with open(outsumfile, 'w') as sumfile:

        sumheader = [
            'mdepth', 'mleaves', 'depth', 'leaves',
            'error', 'accuracy', 'score',
            'nodes', 'splits'
        ]
        writer = csv.DictWriter(sumfile, fieldnames=sumheader)
        writer.writeheader()
        
        for config in dc['configs']:

            # Tree configuration
            mdepth = config['max_depth'] # depth
            mleaves = config['max_leaves'] # number of leaves
            
            # Postfix of exported files with specific depth and number of leaves
            outpostfix = f"mdepth-{mdepth}-mleaves-{mleaves}"
            
            # Decision tree
            clf, summary, rules, df_pred = dtree(
                df_data, df_info, mdepth, mleaves,
                data_path=dc['data'], info_path=dc['info']
            )
            
            # Export summary result to CSV file
            summary['mdepth'] = mdepth
            summary['mleaves'] = mleaves
            writer.writerow(summary)

            # Decision rules
            with open(f"{outruledir}/{outprefix}-rule-{outpostfix}.txt", 'w') as rulefile:
                rulefile.write(rules)
            
            # Prediction
            outpredfile = f"{outpreddir}/{outprefix}-pred-{outpostfix}.csv"
            df_pred.index = df_pred.index + 1
            df_pred.to_csv(outpredfile, index_label='id')
    
            # Tree plots
            plot_tree(clf)
            #plot_tree(clf, label='none', impurity=False)
            for format in fig_formats:
                outfigfile = f"{outfigdir}/{format}/{outprefix}-fig-{outpostfix}.{format}"
                plt.savefig(outfigfile, bbox_inches='tight')
            #plt.show()

            # Newline
            print()
