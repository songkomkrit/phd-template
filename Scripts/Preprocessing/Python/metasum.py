import csv
import pandas as pd

from module.utility import create_dir, import_dict, export_json

# Input directory
meta_dir = "../../../Data/Original/metadata"
# Output directory
metasum_dir = "../../../Outputs/Main/Metasum"

# Execution
indep_dict = import_dict(metadatapath=f"{meta_dir}/meta-indep.json")
mkeys = ['tops', 'subs', 'vars']

org_dict = {mkeys[0]: []}
curtop = ''
itop = iter(org_dict[mkeys[0]])

create_dir(metasum_dir)

with open(f"{metasum_dir}/metavar.csv", 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames = ['variable', 'type', 'topic', 'subtopic', 'label', 'universe'])
    writer.writeheader()

    for key, val in indep_dict.items():
        
        # Write to dictionary
        if val['topic'] != curtop:
            curtop = val['topic']
            org_dict[mkeys[0]].append({'topic': curtop, mkeys[1]: []})
            cursub = ''
            curitop = next(itop)
            isub = iter(curitop['subs'])   
        if val['subtopic'] != cursub:
            cursub = val['subtopic']
            curitop[mkeys[1]].append({'subtopic': cursub, mkeys[2]: []})
            curisub = next(isub)
        curisub[mkeys[2]].append(key)

        # Write to CSV file
        writer.writerow({
            'variable': key, 'type': val['type'],
            'topic': curtop, 'subtopic': cursub,
            'label': val['label'], 'universe': val['universe']
        })

# Write to JSON file
export_json(org_dict, f"{metasum_dir}/metasum.json")

# Normalize JSON data
df = pd.json_normalize(data=org_dict['tops'], record_path=['subs'], meta='topic')
df = df[['topic', 'subtopic', 'vars']] # Original: subtopic, vars, topic
df.to_csv(f"{metasum_dir}/metasum.csv", header=True, index=False)

# Print dictionary example
print(f"\n{org_dict['tops'][0]}\n")

# Print DataFrame example
print(f"{df.head()}\n")
