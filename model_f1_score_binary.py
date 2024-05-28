import torch
from model import AnomalyAE
from torchvision.transforms import Compose, Grayscale, ToTensor
from PIL import Image

def read_labels(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skip the header
    filelist = []
    labels = []
    for line in lines:
        parts = line.strip().split('\t')
        file_num = parts[0]
        is_anomalous = int(parts[1])
        image_file = parts[2]
        label_file = parts[4] if is_anomalous else None
        labels.append(is_anomalous)
        filelist.append(image_file)
    return filelist, labels

def f1_score(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(y_pred)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        else:
            tn += 1
    print(f'tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}')
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyAE().to(device)
model.load_state_dict(torch.load(r'tensorboard_logs_28052024_06-39\models\best_model_25_loss=-0.0004.pt')) # class8
model.eval()

test_folder = 'Class8/Train'
filelist, y_true = read_labels(f'{test_folder}/Label/Labels.txt')
transform = Compose([Grayscale(), ToTensor()])
y_pred = []

threshold = 0.2
postive_threshold = 3

# for debugging
verbose = True
i = 0
for file, label in zip(filelist, y_true):
    img = Image.open(f'{test_folder}/{file}').convert('L')
    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    y = model(img)
    residual = torch.abs(img[0][0]-y[0][0])
    sum = (residual.detach().cpu().numpy()>threshold).ravel().sum()
    if verbose:
        print(f'{file}: {sum}, {label} {"fn" if label == 1 and sum < postive_threshold else "fp" if label == 0 and sum > postive_threshold else ""}')
    y_pred.append(1 if sum >= postive_threshold else 0)
    i += 1
    if i > 100:
        pass
        # break

f1 = f1_score(y_true, y_pred)
print(f'F1 score: {f1}')