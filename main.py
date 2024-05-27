from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Grayscale
import argparse
import torch.nn.functional as F
from torch.optim import Adam
import torch
from trainer import train
from model import AnomalyAE
from datetime import datetime
import os
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used

class NonDefectImageDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []

        labels = os.path.join(root_dir, 'Label/labels.txt')
        with open(labels, 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split('\t')
            file_num = parts[0]
            is_anomalous = int(parts[1])
            if is_anomalous:
                continue
            image_file = parts[2]
            self.img_paths.append(os.path.join(root_dir, image_file))
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return (img, 0)


def create_datagen(data_dir, batch_size=8):
    transform = Compose([Grayscale(), ToTensor()])
    dataset = NonDefectImageDataSet(data_dir, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',
                        required=False,
                        default="Class8/Train",
                        help="Please specify the train directory")
    parser.add_argument('--val_dir',
                        required=False,
                        default="Class8/Test",
                        help="Please specify the test directory")
    parser.add_argument(
        "--log_interval", type=int, default=10,
        help="how many batches to wait before logging training status"
    )
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        help="Please specify the number of epochs")
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=4,
                        help="Please specify the batch_size")
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=4,
                        help="Please specify the batch_size")

    parser.add_argument(
        "--log_dir", type=str,
        default=f'tensorboard_logs_{datetime.now().strftime("%d%m%Y_%H-%M")}',
        help="log directory for Tensorboard log output"
    )
    parser.add_argument(
        '--load_weight_path', type=str,
        help="Please specify the weight path that needs to be loaded.")

    parser.add_argument(
        '--save_graph', action='store_true',
        help="Specify this if you want to save the network graph.")

    args = parser.parse_args()

    optimizer = Adam
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = F.mse_loss
    train_loader = create_datagen(args.train_dir, args.train_batch_size)
    val_loader = create_datagen(args.val_dir, args.val_batch_size)
    model = AnomalyAE()

    train(model, optimizer, loss, train_loader,
          val_loader, args.log_dir, device, args.epochs,
          args.log_interval,
          args.load_weight_path, args.save_graph)
