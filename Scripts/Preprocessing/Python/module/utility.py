import os
import time
import json

# Directory
def create_dir(dir):
    try:
       os.makedirs(dir)
    except FileExistsError:
       pass

# Backup
def backup_duplicate(file_dir, filename, format, backup_dir, info):
    filepath = f"{file_dir}/{filename}.{format}"
    date = time.strftime("%Y%m%d", time.localtime(time.time()))
    if os.path.isfile(filepath):
        backup_subdir = f"{backup_dir}/{date}/{file_dir.replace('../', '')}"
        create_dir(backup_subdir)
        filepath_backup = f"{backup_subdir}/{filename}-backup.{format}"
        os.replace(filepath, filepath_backup)
        if info:
            print(f"{filepath} previously exists")
            print(f"Back up to {filepath_backup}")
    elif info:
        print(f"{filepath} does not previously exists")

# Import/export dict/JSON
def import_dict(metadatapath):
    with open(metadatapath) as myfile:
    	indep_contents = myfile.read()
    return json.loads(indep_contents)

def export_json(dictfile, jsonfile):
    with open(jsonfile, 'w', encoding='utf-8') as f:
        json.dump(dictfile, f, ensure_ascii=False, indent=4)

def export_txt(string, txtfile):
    f = open(txtfile, 'w')
    f.write(string)
    f.close()
