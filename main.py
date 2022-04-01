from torch.utils.data import Dataset, DataLoader
import os, re, glob, torch
import pandas as pd
from enum import Enum
from PIL import Image
from torchvision import transforms
from models.simplenn import Net

ANNOTATIONS_FOLDER_PATH = "data/annotations"
IMAGES_FOLDER_PATH = "data/images"
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64

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
        # all_files = glob.glob(txt_files_path + "/*.txt")
        all_files = ['/'.join([txt_files_path, fname]) for fname in os.listdir(txt_files_path)]
        li = []
        for filename in all_files:
            regex_pattern = ".*/(.*).txt"
            label = re.search(regex_pattern, filename).group(1)
            df = pd.read_csv(filename, index_col=None, header=None, names=['image_number'])
            df["label"] = Label[label].value - 1
            df["image_path"] = df["image_number"].apply(lambda x: '/'.join([root_dir, f"im{x}.jpg"]))
            li.append(df)
        self.df = pd.concat(li, axis=0, ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def build_path(self, relative_path):
        return os.path.abspath(relative_path)

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



from models.resnet import resnet18
resnet18_model = resnet18(num_classes=14)
optimizer = torch.optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.75)
loss_function = torch.nn.CrossEntropyLoss()

from tqdm import tqdm
training_losses = []
training_accuracies = []
for epoch in range(1):
    total_train_loss = 0
    train_correct = 0
    total = 0
    for batch_num, batch in enumerate(tqdm(train_set_dataloader)):
        data, target = batch['image'], batch['label']
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




# ------------------------    

batch = next(iter(train_set_dataloader))
X, y = batch['image'], batch['label']
model = Net()
pred = model.forward(X)


breakpoint()