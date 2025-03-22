# Course Outline

1. Training Robust Neural Networks
2. Images and CNN
3. Sequences and RNN
4. Multi-input and Multi-Output Architectures

#  Training Robust Neural Networks


## PyTorch Dataset

```Python
from torch.utils.data import Dataset
import pandas as pd

class WaterDataset(Dataset):
    def  __init__(self, csv_path):
        # inherits from torch Dataset
        super().__init__()
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # all but last are the features, and the last one is the label
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return features, label

```

## DataLoader Setup

```Python

from torch.utils.data import DataLoader

dataset_train = WaterDataset("water_tarin.csv")
# pass to pytroch data loader
dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=2)

# get one barch from the data loader
features,labels = next(iter(dataloader_train))
```

## Model definition

```Python
import torch.nn as nn

# Sequential (Quick but less flexible)
net_seq = nn.Sequential(
    nn.Linear(9, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

# Class-based model (Preferred for customization, debugging, layers like BatchNorm)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x

net = Net()
```

## Training Loop

Loss function - conventionally called the criterion
BCELoss - for binary classification
SGD - optimizer

```Python
import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(1000):
    for features, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(features)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward() # gradients contain information about direction and size of the changes required to minimize the loss
        optimizer.step() # pass the gradients to the optimizer
```

How an optimizer works?
eg consider 
Parameters [1,0.5] and graidents [0.9, -0.2]
Optimizer takes these and outputs parameter updates
eg. [-0.5, 0.5], notices the positive gradient meant negative update, and a negative gradient positive update. The amount is based on the optimizer used.

SGD depends on learning rate, a predefined hyperparameter
Very simple, so rarely used in practice,
Using the same LR on a paramater cannot be optimal so **Adaptive Gradient(Adagrad)** adaps learning rate for each paramater

`optim.Adagrad(net.parameters(), lr = 0.01)`
- good for sparse data but may use decrease in lr too fast
`optim.RMSprop(net.parameters(), lr =0.01)` - update each parameter based on the size of the previous gradients
- updates based on the size of previous gradients
`optim.Adam(..)`
- most widely used
- RMSprop + gradient momentum


## Evaluation

- once model is trained evaluate on test data


```Python
from torchmetrics import Accuracy

acc = Accuracy(task="binary")
net.eval() # put in eval mode and iterate over test data batches with no gradients

with torch.no_grad():
    for features, labels in dataloader_test:
        outputs = net(features)
        preds = (outputs >= 0.5).float()
        acc(preds, labels.view(-1, 1))

accuracy = acc.compute()
print(f"Accuracy: {accuracy}")
```

-----------------------------
Optimizers Summary
-----------------------------
SGD: Simple, fast. Rare in practice.
Adagrad: Good for sparse data, fast decay
RMSProp: Handles varying gradient sizes
Adam: Best default. Combines momentum + RMSProp

Usage:
optimizer = optim.Adam(net.parameters(), lr=0.01)
 
---

## 🔻 Vanishing Gradients
- Gradient become extremely small during backward pass.
- Early layers receive **tiny weight updates**, making them learn slowly or not at all.
- Common in deep networks using **sigmoid or tanh** activations.

### 🔬 Why it happens?
For sigmoid/tanh:
- Derivatives ∈ (0, 1)
- Multiple small values → exponentially smaller gradients as layers increase.

### 🔍 Impact
- Slows convergence.
- Traps training in local minima.
- Network becomes **hard to train**.

---

## 🔺 Exploding Gradients
- Gradients grow exponentially through layers.
- Causes **numerical instability**, large parameter updates.

### 🔍 Impact
- Diverging loss.
- Model weights become **NaN or Inf**.

---

## 🔧 Solutions

### 1. ✅ Better Weight Initialization
- Goal: Maintain gradient variance across layers.
- **Xavier Initialization** (Glorot) for tanh.
- **He/Kaiming Initialization** for ReLU/ELU.

```python
import torch.nn.init as init

init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```

### 2. ✅ Use Better Activation Functions
- **ReLU**: Simpler and avoids saturation in positive range.
- **Leaky ReLU / ELU / GELU**: Prevents dying neuron problems and keeps gradients flowing.

### 3. ✅ Batch Normalization
- Normalize layer outputs → keeps activation distributions stable.
- Helps mitigate both vanishing and exploding gradients.
```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        return x
```

### 4. ✅ Gradient Clipping
- Limits gradient values during backprop.
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---


## Weight Initialization

When we create a torch layer, its parameters stored in the weight attribute get initialized to random values.

Good initialization ensures:
- variance of layer inputs = variance of layer outputs
- variance of gradients the same before and after a layer
- achieving this for each depens on the activtaion function

[]Find out Why

```Python
import torch.nn.init as init

# add this snippet in the initization after initializing layers as self.fcx = nn.Linear(...)

init.kaiming_uniform_(net.fc1.weight)
init.kaiming_uniform_(net.fc2.weight)
init.kaiming_uniform_(net.fc3.weight, nonlinearity="sigmoid")

# Use He initialization (kaiming) for ReLU
```

- `nn.functional.relu` is the most commonly used activation function
- suffers from dying neurons
- ELU : non zero gradients for negative values, helps against dying neurons, average output around zero- helps against vanishing gradients

## Batch Normalization

- applied after a layer where output is first normalized, making sure output distribution is normal, then scales and shifted
-  normalized outputs are not passed further; rather, they are rescaled again using learned parameters!
- model learns optimal inputs distribution for each layer
    - faster loss decrease
    - helps against unstable gradients


```Python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        # Add two batch normalization layers
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)
        
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity="sigmoid")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)

        # Pass x through the second set of layers
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)

        x = nn.functional.sigmoid(self.fc3(x))
        return x

# BatchNorm reduces internal covariate shift and stabilizes gradients.
```

## Summary

- Make a dataset class using torch dataset, then set up the dataloader_train
- train by making a net class, which inherits from nn.Module, and define init and forward pass
- in init make sure to add in layers for batch normalization, and use the appropriate weight initialization method, such as kiaming for RELU to help with vanishing and exploring gradients problems
- set up the training loop by choosing the appropriate criterion/loss function and the optimizer, number of epochs.
- once trained, test the model. Remember to set `net.eval()`. Choose appropriate `Accuracy` metric. For features and labels in the testing set, making predictions, compare and get accuracy.

# Images and CNN

## Loading and Transforming Image Data
```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

# this is a pytorch dataset like the pervious WaterDataset, can create a dataLoader from it and get a dataset sample
dataset_train = ImageFolder("data/clouds_train", transform=train_transforms)
```
- **Why this way?** `ImageFolder` auto-labels data from directory names.
- `ToTensor` converts image to PyTorch format (C×H×W).
- `Resize` ensures uniform input size for CNN.

## DataLoader and Displaying Images
```python
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=1)
image, label = next(iter(dataloader_train))
print(image.shape) # torch.Size([1,3,128,128]) batch size, three color channels, images height and width
image = image.squeeze().permute(1, 2, 0) # 128,128,3 
plt.imshow(image)
plt.show()
```
- `permute` reorders tensor from (C,H,W) to (H,W,C) for display.

## Data Augmentation
```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])
```
- **Why?** Increases dataset diversity → better generalization.

## Convolutional Layer

- Linear layers: too many parameters, increase overfitting, do not recognize spatial patterns
- convolution: compute dot product of input patch  with filter and compute the sum
- add padding to the input (zeroes)
    - maintains spatial dimensions, and ensures border pixels are treated equally to others
- at each position perform convolution resulting in a feature map
    
```python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```
- Learns spatial features, fewer weights than linear layers.
- `padding=1` preserves input size.

## Max Pooling Layer
```python
nn.MaxPool2d(kernel_size=2)
```
- Reduces dimensionality and computation, retains strong features.
- used after convolutional layer, in each 4 by 4 grid, only take the largest values

## CNN Architecture Design

- lesser params than linear layers, 3x3 filter - 9, even if many filters, different from linear layer where everything is connected

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # convulution, activation, pooling, applied twice and flattened
        self.feature_extractor = nn.Sequential(
            # input image 3x64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # output 32 features maps, 32 64 64
            nn.ELU(),
            nn.MaxPool2d(2), # dimentions halfed, 32 32 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 32 32
            nn.ELU(),
            nn.MaxPool2d(2), # 64 16 16
            nn.Flatten()
        )
        self.classifier = nn.Linear(64*16*16, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x) # single linear layer
```
- Separate **feature extractor** and **classifier** is modular design.
- `Flatten` bridges CNN to fully connected layer.

## Training Loop for Classifier

What should not be augmented? eg hand written letters, eg W and M
Choose augmentation with data and task in mind
For cloud classiication, random rotation, horizontal flip, and auto contrast adjustment is good
Multiclass classification tasks, so Cross-Entropy loss

```Python
train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((218, 128))
])
```
```python
import torch.optim as optim

net = Net(num_classes=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```
- Standard training steps.
- `Adam` is robust, adaptive learning rate.

## Test Data Loader (No Augmentation)
```python
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])
dataset_test = ImageFolder("clouds_test", transform=test_transforms)
```

## Evaluation with Metrics
Previously predicted based on accuracy
Precisoin: Fraction of correct positive predictions, out of those predicted positive how many are actually positive, out of those you classified as having cancer how many actually have cancer, higher precision - better at avoiding false positives

Recall: fraction of all positive examples correctly predicted, out of those who have cancer, how many have you labelled as having cancer? higher recall - better at true positives

For multiclass classification
- precision: out of all those who were predicted as being cumulus clouds, how many are actually cumulus clouds? so how much can I trust you when you say something is cumulus
- recall: out of all the cumulus examples, how many were correctly predicted, how much of my data of cumulus can I expect you to correctly label?

7 cloud classes- 7 precision an recall scores
- micro average: global calculation
- macro average: mean of per-class Metrics
- weighted average: weighted mean of perclass metric

```Python
from torchmetrics import Recall

recall_per_class = Recall(task = "multiclass", num_classes=7, average=None)
recall_micro = Recall(task = "multiclass", num_classes=7, average="micro")
recall_macro = Recall(task = "multiclass", num_classes=7, average="macro")
recall_weighted = Recall(task = "multiclass", num_classes=7, average="weighted")
```
Micro: imbalanced dataset
macro: care about performance on small classes even if they have fewer datapoints
weighted: consider errors in large classes as more important

weighted: 

Evaluation loop

```python
from torchmetrics import Precision, Recall

metric_precision = Precision(task="multiclass", num_classes=7, average="macro")
metric_recall = Recall(task="multiclass", num_classes=7, average="macro")

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = outputs.max(1)
        metric_precision(preds, labels)
        metric_recall(preds, labels)

precision = metric_precision.compute()
recall = metric_recall.compute()
```
- **Macro averaging**: treats each class equally.

## Per-Class Recall
```python
metric_recall = Recall(task="multiclass", num_classes=7, average=None)
recall = metric_recall.compute()

recall_per_class = {k: recall[v].item() for k, v in dataset_test.class_to_idx.items()}
```
- Gives insights into model weaknesses per class.

## Summary