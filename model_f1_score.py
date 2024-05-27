from model import AnomalyAE
from PIL import Image
import torch
from torchvision.transforms import Compose, Grayscale, ToTensor
from sklearn.metrics import f1_score
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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
model.load_state_dict(torch.load(r'tensorboard_logs_28052024_01-29\models\best_model_7_loss=-0.0000.pt')) # class8
model.eval()
model = model.to('cuda')

test_dir = 'Class8/Test'
label_dir = 'Class8/Test/Label'
label_file = 'Class8/Test/Label/Labels.txt'

log_dir = 'image/class8_residual'
cmp_dir = 'image/class8_cmp'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(cmp_dir, exist_ok=True)

labels = read_labels(label_file)

threshold = 0.02

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
            residual_np = residual.cpu().numpy()
            residual_np = (residual_np > threshold).astype(np.uint8) * 255
            # blur
            residual_np = cv2.GaussianBlur(residual_np, (21, 21), 0)
            #  _, residual_binary = cv2.threshold(residual_np, 20, 255, cv2.THRESH_BINARY)
            residual_binary = (residual_np > 20).astype(np.uint8)

            
            cv2.imwrite(f'{log_dir}/{file_num}_residual.png', residual_binary)
        else:
            residual_binary = cv2.imread(f'{log_dir}/{file_num}_residual.png', cv2.IMREAD_GRAYSCALE)
            residual_binary = (residual_binary > 0).astype(np.uint8)
        

        if is_anomalous:
            label_path = f'{label_dir}/{label_file}'
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label_binary = (label > 0).astype(np.uint8)
        else:
            label_binary = np.zeros_like(residual_binary)
        
        if False:
            plt.figure(figsize=(15,10))
            plt.subplot(121)
            plt.imshow(residual_binary)
            plt.title('Residual Thresholded')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(label_binary)
            plt.title('Label')
            plt.axis('off')
            plt.savefig(f'{cmp_dir}/{file_num}_comparison.png', bbox_inches='tight')
            # plt.show()
            plt.close()
        y_true.extend(label_binary.flatten())
        y_pred.extend(residual_binary.flatten())
        if file_num == '0050':
            break
        
print('Calculating F1 Score...')
f1 = f1_score_torch(y_true, y_pred)
print(f'F1 Score: {f1:.4f}')