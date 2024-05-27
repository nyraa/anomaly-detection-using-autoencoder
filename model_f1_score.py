from model import AnomalyAE
from PIL import Image
import torch
from torchvision.transforms import Compose, Grayscale, ToTensor
from sklearn.metrics import f1_score
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_labels(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skip the header
    labels = []
    for line in lines:
        parts = line.strip().split('\t')
        file_num = parts[0]
        is_anomalous = int(parts[1])
        image_file = parts[2]
        label_file = parts[4] if is_anomalous else None
        labels.append((file_num, is_anomalous, image_file, label_file))
    return labels

def f1_score_torch(y_true, y_pred):
    y_true = torch.tensor(y_true).to(device)
    y_pred = torch.tensor(y_pred).to(device)
    # Assuming y_true and y_pred are tensors
    assert y_true.shape == y_pred.shape, "Input tensors must have the same shape"
    
    # Compute true positives, false positives, and false negatives
    tp = (y_true * y_pred).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)
    
    # Compute precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Compute F1 score (macro average)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Return the average F1 score across classes
    return f1.mean().item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyAE().to(device)
model.load_state_dict(torch.load('./tensorboard_logs_04092023_04-25/models/best_model_25_loss=-0.0000.pt')) # class8
model.eval()
model = model.to('cuda')

test_dir = 'Class8/Test'
label_dir = 'Class8/Test/Label'
label_file = 'Class8/Test/Label/Labels.txt'

log_dir = 'image/class8_residual'

labels = read_labels(label_file)

threshold = 0.007

y_true = []
y_pred = []

with torch.no_grad():
    for file_num, is_anomalous, image_file, label_file in labels:
        print(f'Processing {file_num}...')

        img = Image.open(f'{test_dir}/{image_file}').convert('L')
        transform = Compose([Grayscale(), ToTensor()])
        img = transform(img)
        img = img.to('cuda')
        img = img.unsqueeze(0)
        if True:
            y = model(img)
            residual = torch.abs(img[0][0]-y[0][0])
            residual_binary = (residual > threshold).type(torch.uint8).cpu().numpy()
            cv2.imwrite(f'{log_dir}/{file_num}_residual.png', residual_binary * 255)
        else:
            residual_binary = cv2.imread(f'{log_dir}/{file_num}_residual.png', cv2.IMREAD_GRAYSCALE)
            residual_binary = (residual_binary > 0).astype(np.uint8)
        

        if is_anomalous:
            label_path = f'{label_dir}/{label_file}'
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label_binary = (label > 0).astype(np.uint8)
        else:
            label_binary = np.zeros_like(residual_binary)
        
        y_true.extend(label_binary.flatten())
        y_pred.extend(residual_binary.flatten())
        if file_num == '0100':
            break
        
print('Calculating F1 Score...')
f1 = f1_score_torch(y_true, y_pred)
print(f'F1 Score: {f1:.4f}')