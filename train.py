# %%
from enum import Enum
import glob
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
import re
from PIL import Image
from skimage import io
from skimage import color

# %%
ANNOTATIONS_FOLDER_PATH = "data/annotations"
IMAGES_FOLDER_PATH = "data/images"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#--- hyperparameters ---
N_EPOCHS = 20
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
LR = 0.0001
NUM_CLASSES = 14

# %%
class CSVDataLoader(Dataset):
    """CSV data loader."""

    def __init__(self, txt_files_path, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the txt files with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        Label = Enum("Label", 'baby bird car clouds dog female flower male night people portrait river sea tree')
        all_files = glob.glob(txt_files_path + "/*.txt")
        li = []
        for filename in all_files:
            regex_pattern = ".*/(.*).txt"
            label = re.search(regex_pattern, filename).group(1)
            df = pd.read_csv(filename, index_col=None, header=None, names=['image_number'])
            breakpoint()
            df["label"] = Label[label].value - 1
            df["image_path"] = df["image_number"].apply(lambda x: os.path.join(root_dir, f"im{x}.jpg"))
            li.append(df)
        self.df = pd.concat(li, axis=0, ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def build_path(self, relative_path):
        return os.path.join(self.root_dir, relative_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.build_path(self.df.loc[self.df.index[idx], "image_path"])
        image = Image.open(img_path).convert("RGB")
        im_torch = transforms.ToTensor()(image)
        label = self.df.loc[self.df.index[idx], "label"]
        label_tensor = torch.tensor(label, dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        sample = {'image': im_torch, 'label': label_tensor}
        return sample

# %%
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(180),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Values aquired from dataloaders/plant_master_dataset_stats.py
    transforms.Normalize(mean=[0.09872966, 0.11726899, 0.06568969],
                         std=[0.1219357, 0.14506954, 0.08257045])
])

master_dataset = CSVDataLoader(
  txt_files_path=ANNOTATIONS_FOLDER_PATH, 
  root_dir=IMAGES_FOLDER_PATH,
  transform=None
)

train_size = int(0.80 * len(master_dataset))
test_size = len(master_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(master_dataset, [train_size, test_size])

train_set_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)
test_set_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

resnet18_model = resnet18(num_classes=NUM_CLASSES).to(device)

optimizer = optim.SGD(resnet18_model.parameters(), lr=LR, momentum=0.75)
loss_function = torch.nn.CrossEntropyLoss()

# %%
# training

training_losses = []
training_accuracies = []

for epoch in range(N_EPOCHS):
    total_train_loss = 0
    train_correct = 0
    total = 0

    for batch_num, batch in enumerate(train_set_dataloader):
        data, target = batch['image'].to(device), batch['label'].to(device)

        optimizer.zero_grad() 

        output = resnet18_model(data)
        train_loss = loss_function(output, target)
        train_loss.backward()
        optimizer.step()
        
        pred = output.max(1, keepdim=True)[1]

        correct = pred.eq(target.view_as(pred)).sum().item()
        train_correct += correct
        total += data.shape[0]
        total_train_loss += train_loss.item()

        if batch_num == len(train_set_dataloader) - 1:
            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                  (epoch, batch_num + 1, len(train_set_dataloader), train_loss / (batch_num + 1), 
                   100. * train_correct / total, train_correct, total))

    # Training loss average for all batches
    training_losses.append(total_train_loss / len(train_set_dataloader))
    training_accuracies.append((100. * train_correct / total))

plt.plot(range(N_EPOCHS), training_losses, label = "Training loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
plt.show()

plt.plot(range(N_EPOCHS), training_accuracies, label = "Training accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

# %%

# test
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for batch_num, batch in enumerate(test_set_dataloader):
        data, target = batch['image'].to(device), batch['label'].to(device)
        
        output = resnet18_model(data)
        test_loss += loss_function(output, target).item()
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        test_correct += correct
        total += data.shape[0]

        test_loss /= len(test_set_dataloader.dataset)

print("Final test score: Loss: %.4f, Accuracy: %.3f%%" % (test_loss, (100. * test_correct / total)))

# %%
