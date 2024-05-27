import os

source_dir = 'Class8/Test'
dest_dir = f'{source_dir}/nonDefect'
defect_dir = f'{source_dir}/Defect'

os.makedirs(dest_dir, exist_ok=True)
os.makedirs(defect_dir, exist_ok=True)

label_file = f'{source_dir}/Label/Labels.txt'
with open(label_file, 'r') as f:
    labels = f.readlines()[1:]

for label in labels:
    parts = label.strip().split('\t')
    file_num = parts[0]
    is_anomalous = int(parts[1])
    image_file = parts[2]
    label_file = parts[4] if is_anomalous else None
    if not is_anomalous:
        os.rename(f'{source_dir}/{image_file}', f'{dest_dir}/{image_file}')
    else:
        os.rename(f'{source_dir}/{image_file}', f'{defect_dir}/{image_file}')