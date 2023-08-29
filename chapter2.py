import torch.nn
import torchvision
from torchvision import transforms

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

train_data_path = "./train/"
transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)

val_data_path = './val/'
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transforms)
test_data_path = './test/'
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms)

from torch.utils import data

batch_size = 64
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)

from torch import nn
import torch.nn.functional as F


class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


simplenet = SimpleNet()

import torch.optim as optim

optimizer = optim.Adam(simplenet.parameters(), lr=0.001)


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="mps"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], target)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch:{},Training loss:{:.2f},Validation Loss:{:.2f},accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                              valid_loss,
                                                                                              num_correct / num_examples))


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("mps")

simplenet = simplenet.to(device)
print(device)

train(simplenet, optimizer, nn.CrossEntropyLoss(), train_data_loader, val_data_loader, device=device)
