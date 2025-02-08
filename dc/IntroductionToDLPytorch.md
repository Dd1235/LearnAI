Introduction to Deep Learning with Pytorch

# Outline

1. Introduction to Pytorch
2. Training Our First Neural Network with Pytorch
3. Neural Network Architecture and Hyperparameters
4. Evaluating and Improving Models

# Ch 1

- models require large amounts of data, at least 100000s data points
- tensors are the basic data structure in Pytorch
```Python
import torch
a = torch.tensor([1, 2, 3]) # create from a list
np_arr = np.array(mylist)
tensor_from_np = torch.from_numpy(np_arr) # create from numpy array

a.shape
a.dtype
a.device
# can do
a + b
a * b # elementwise multiplication
a @ b # matrix multiplication
```
- tensors can be moved to GPU

```Python
import torch.nn as nn
input_tensor = torch.tensor([0.3471, 0.2865, 0.1456]) # input tensor with three features

# deine the first linear layer
linear_layer = nn.Linear(
    in_features = 3,
    out_features = 2
)
output = linear_layer(input_tensor) 
# output is a pytorch tensor, with two elements, grad_fn = <AddmmBackward0> matrix mult followed by addition
``` 

- Each linear layer has a .weight property and a .bias property
- `linear_layer.weight` is a tensor with shape (2, 3) and `linear_layer.bias` is a tensor with shape (2,)
- the weights represent the coefficients of the linear equation, and the bias is the intercept

- networks with only linear layers are called fully_connected
- each neuron in one layer is connected to each neuron in the next layer

- Stack layers using   `nn.Sequential()`
```Python
model = nn.Sequential(
    nn.Linear(10,18),
    nn.Linear(18, 20),
    nn.Linear(20, 5)
)
```
# Ch 2


- Activatoin functions add non-linearity to the networks
- pre-activation output is passed to the activation function
- sigmoid activation function, Binary Classification task, based on a threshold like 0.5 it gives you a class label 0 or 1
- tanh activation function, similar to sigmoid, but ranges from -1 to 1

- ReLU activation function, most commonly used, it is linear for positive values and zero for negative values

- Softmax activation function, used for multi-class classification, it converts the output to a probability distribution

```Python
sigmoid = nn.Sigmoid()
output = sigmoid(input_tensor)

model = nn.Sequential(
    nn.Linear(6,4),
    nn.Linear(4,1),
    nn.Sigmoid()
)
```

Softmax -> multi-class classification

- takes N-element vector as input and outputs a vector of same size, but a probability distribution

![soft max](softmax.png)

```Python
probabilities= nn.Softmax(dim = -1) # -1 : applied to the input tensor's last dimension
output_tensor = probabilities(input_tensor)
```
Here is a comprehensive set of notes in `.md` format, covering everything from the document, including code snippets:




## What is a Forward Pass?
- A **forward pass** refers to the process of passing input data through a network.
- **Key Steps**:
  1. Input data is propagated through each layer.
  2. Computations are performed at each layer.
  3. Outputs of each layer are passed to the next layer.
  4. The output of the final layer is the **prediction**.
- **Used for:** Both training and prediction.

### Possible Outputs:
1. **Binary Classification**: A single probability between 0 and 1.
2. **Multiclass Classification**: A distribution of probabilities summing to 1.
3. **Regression**: Continuous numerical predictions.

---

## Is There a Backward Pass?
- A **backward pass** or **backpropagation** is used during training to update the model's parameters (weights and biases).
- **Training Loop Steps**:
  1. Forward propagate input data.
  2. Compare outputs to true values (ground truth).
  3. Backpropagate to update weights and biases.
  4. Repeat until the model learns useful patterns.

---

## Binary Classification: Forward Pass Example
```python
import torch
import torch.nn as nn

# Create input data of shape 5x6
input_data = torch.tensor([
    [-0.4421,  1.5207,  2.0607, -0.3647,  0.4691,  0.0946],
    [-0.9155, -0.0475, -1.3645,  0.6336, -1.9520, -0.3398],
    [ 0.7406,  1.6763, -0.8511,  0.2432,  0.1123, -0.0633],
    [-1.6630, -0.0718, -0.1285,  0.5396, -0.0288, -0.8622],
    [-0.7413,  1.7920, -0.0883, -0.6685,  0.4745, -0.4245]
])

# Create binary classification model
model = nn.Sequential(
    nn.Linear(6, 4),  # First linear layer
    nn.Linear(4, 1),  # Second linear layer
    nn.Sigmoid()      # Sigmoid activation function
)

# Pass input data through the model
output = model(input_data)
print(output)
```

### Output Example:
```plaintext
tensor([[0.5188], [0.3761], [0.5015], [0.3718], [0.4663]],
       grad_fn=<SigmoidBackward0>)
```

---

## Multiclass Classification: Forward Pass Example
```python
# Specify the number of classes
n_classes = 3

# Create multiclass classification model
model = nn.Sequential(
    nn.Linear(6, 4),  # First linear layer
    nn.Linear(4, n_classes),  # Second linear layer
    nn.Softmax(dim=-1)  # Softmax activation function
)

# Pass input data through the model
output = model(input_data)
print(output.shape)  # Output shape: torch.Size([5, 3])
print(output)
```

---

## Regression: Forward Pass Example
```python
# Create regression model
model = nn.Sequential(
    nn.Linear(6, 4),  # First linear layer
    nn.Linear(4, 1)   # Second linear layer
)

# Pass input data through the model
output = model(input_data)
print(output)
```

### Output Example:
```plaintext
tensor([[0.3818], [0.0712], [0.3376], [0.0231], [0.0757]],
       grad_fn=<AddmmBackward0>)
```

---

## Using Loss Functions to Assess Model Predictions
### Why Do We Need a Loss Function?
- A **loss function** provides feedback during training by:
  - Taking in model predictions and ground truth.
  - Outputting a single scalar (the loss).

---

### One-Hot Encoding Example
```python
import torch.nn.functional as F

# One-hot encoding
print(F.one_hot(torch.tensor(0), num_classes=3))  # Output: tensor([1, 0, 0])
print(F.one_hot(torch.tensor(1), num_classes=3))  # Output: tensor([0, 1, 0])
print(F.one_hot(torch.tensor(2), num_classes=3))  # Output: tensor([0, 0, 1])
```

---

### Cross Entropy Loss in PyTorch
```python
from torch.nn import CrossEntropyLoss

# Scores and one-hot target
scores = torch.tensor([[-0.1211, 0.1059]])
one_hot_target = torch.tensor([[1, 0]])

# Define cross-entropy loss
criterion = CrossEntropyLoss()
loss = criterion(scores.double(), one_hot_target.double())
print(loss)  # Example output: tensor(0.8131, dtype=torch.float64)
```

---

## Backpropagation Concepts
- Backpropagation calculates local gradients for each layer.
- It starts with the loss gradient w.r.t. the output layer, propagating backward.

### Example: Backpropagation in PyTorch
```python
from torch.nn import CrossEntropyLoss

# Create the model and forward pass
model = nn.Sequential(
    nn.Linear(16, 8),
    nn.Linear(8, 4),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
)
prediction = model(torch.randn(1, 16))

# Calculate the loss and compute gradients
criterion = CrossEntropyLoss()
loss = criterion(prediction, torch.tensor([1]))
loss.backward()

# Access gradients
print(model[0].weight.grad)
print(model[0].bias.grad)
```

---

## Updating Model Parameters
```python
# Learning rate
lr = 0.001

# Example parameter update
weight = model[0].weight
weight_grad = model[0].weight.grad
weight = weight - lr * weight_grad

bias = model[0].bias
bias_grad = model[0].bias.grad
bias = bias - lr * bias_grad
```

---

## Gradient Descent Using PyTorch Optimizers
```python
import torch.optim as optim

# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Update weights after calculating gradients
optimizer.step()
```

---

## Training a Neural Network
### Steps:
1. Create a model.
2. Choose a loss function.
3. Create a dataset.
4. Define an optimizer.
5. Run the training loop.

### Training Loop Example
```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset and dataloader
features = torch.randn(10, 4)
target = torch.randn(10, 1)
dataset = TensorDataset(features, target)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create model, loss, and optimizer
model = nn.Sequential(
    nn.Linear(4, 2),
    nn.Linear(2, 1)
)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for data in dataloader:
        # Reset gradients
        optimizer.zero_grad()
        feature, target = data
        # Forward pass
        pred = model(feature)
        # Compute loss and gradients
        loss = criterion(pred, target)
        loss.backward()
        # Update parameters
        optimizer.step()
```

---

## Summary
- **Forward pass**: Computes predictions.
- **Backward pass (backpropagation)**: Computes gradients for parameter updates.
- **Training loop**: Combines forward pass, loss computation, gradient computation, and parameter updates iteratively.

---

# Ch 3

## Limitations of Softmax and Sigmoid functions

- Sigmoid produced a gradient almost zero for very large or very small values
- Causes function to saturate, and the model stops learning
- Softmax is used for multi-class classification, but it is not robust to outliers
- can lead to **Vanishing gradients** or **Exploding gradients** during backpropagation

## ReLU
- rectified linear unit
- f(x) = max(0, x)
- does not saturate for positive values
- does not have vanishing gradient problem
- but can have dying ReLU problem, where the neuron never activates
- ` relu = nn.ReLU()`

## Leaky ReLU

- f(x) = x if x > 0, else alpha * x
- alpha is a small constant
- prevents dying ReLU problem
- 0.01 is default in PyTorch
- `leaky_relu = nn.LeakyReLU(negative_slope = 0.05)`


## Example Code ReLU and Leaky ReLU

```Python

# Create a ReLU function with PyTorch
relu_pytorch = nn.ReLU()

# Apply your ReLU function on x, and calculate gradients
x = torch.tensor(-1.0, requires_grad=True)
y = relu_pytorch(x) # tensor(0., grad_fn=<ReluBackward0>)
y.backward()

# Print the gradient of the ReLU function for x
gradient = x.grad # tensor(0.)
print(gradient)

# Create a leaky relu function in PyTorch
leaky_relu_pytorch = nn.LeakyReLU(negative_slope = 0.05)

x = torch.tensor(-2.0) # tensor(-2.) 
# Call the above function on the tensor x
# -2 * 0.05 = -0.1
output = leaky_relu_pytorch(x) # tensor(-0.1000)
print(output)

```

## Deeper dive into neural network Architecture


- input and output layer dimensions are fixed
    - input layer is determined by the number of features
    - output layer is determined by the number of classes
- increasing number of hidden layers = increasing model capacity


### Counting the number of parameters

```Python
model = nn.Sequential(
    nn.Linear(8,4), 
    nn.Linear(4,2)
)
```
- first layers has 4 neurons, each neuron has 8 weights and 1 bias, so 9 features, therefore 36 parameters
- second layer has 2 neuons with 5 parameters each, so 10 parameters
- total = 46 parameters

- `.numel()` returns the number of elements in the tensor
- `model.parameters()` returns an iterator over all model parameters

```Python   
total = 0
for parameter in model.parameters():
    total += parameter.numel()
print(total)
```

## Learning Rate and Momentum

- Stochastic Gradient Descent (SGD) is the most common optimization algorithm
- Learning rate is a hyperparameter that controls the step size
- Momentum is a hyperparameter that helps accelerate SGD in the relevant direction and dampens oscillations

`sgd = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)`

- momentum helps with non convex functions and helps escape local minima

## Layer Initialization and Fine Tuning

- weights are initialized randomly
` print(layer.weight.min(), layer.weight.max())`
- nn.init module provides functions to initialize weights
- `nn.init.uniform_(layer.weight)` values from 0 to 1

- Transfer learning: resuinga  model trained on a first task for a second similar task
- fine-tuning is  a type of transfer learning
    - load weights from a pre-trained model
    - train it with a smaller learning rate
    - can freeze some layers
    - freeze early layers of network and fine-tune layers closer to output layer

```Python
for name, param in model.named_parameters():
    if name == '0.weight':
        param.requires_grad = False
```

## Example snippets for Layer Initialization and Fine Tuning

```Python
for name, param in model.named_parameters():    
  
    # Check if the parameters belong to the first layer
    if name == '0.weight' or name == '0.bias':
      
        # Freeze the parameters
        param.requires_grad = False
  
    # Check if the parameters belong to the second layer
    if name == '1.weight' or name == '1.bias':
      
        # Freeze the parameters
        param.requires_grad = False
```

```Python
layer0 = nn.Linear(16, 32)
layer1 = nn.Linear(32, 64)

# Use uniform initialization for layer0 and layer1 weights
nn.init.uniform_(layer0.weight)
nn.init.uniform(layer1.weight)

model = nn.Sequential(layer0, layer1)
```

# Ch 4

## A deeper dive into loading data

`torch.utils.data` provides DataLoader, TensorDataset, etc.

```Python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())

sample = dataset[0]
input_sample, label_sample = sample

batch_size = 2 # number of samples in each batch
shuffle = True # shuffle the data
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

# each element of dataloader is a tuple of input and label

for batch_inputs, batch_labels in dataloader:
    print(batch_inputs) # 2 since batch size is 2
    print(batch_labels) 
```

```Python
# Load the different columns into two PyTorch tensors
features = torch.tensor(dataframe[['ph', 'Sulfate', 'Conductivity', 'Organic_carbon']].to_numpy()).float()
target = torch.tensor(dataframe['Potability'].to_numpy()).float()

# Create a dataset from the two generated tensors
dataset = TensorDataset(features, target)

# Create a dataloader using the above dataset
dataloader = DataLoader(dataset, shuffle=True, batch_size=2)
x, y = next(iter(dataloader))

# Create a model using the nn.Sequential API
model = nn.Sequential(
    nn.Linear(4,2),
    nn.Linear(2,1)
)
output = model(features)
print(output)
```

## Evaluating Model Performance

- Loss
    - Training
    - Validation
- Accuracy
    - Training
    - Validation

- in case of classification, accuracy measures how well the model correctly predicts 

```Python
training_loss = 0
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    training_loss += loss.item() 
epoch_loss  = training_loss / len(trainloader)
```

- After the training epoch, we iterate over the validation set and calculate the average validation loss

```Python
validation_loss = 0
model.eval() # set model to evaluation mode
with torch.no_grad(): # no gradient calculation 
    for i,date in enumerate(validationloader, 0):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        validation_loss += loss.item()
epoch_loss = validation_loss / len(validationloader)
model.train() # set model back to training mode
```

- Overfitting when validation loss is high but training loss is not

```Python
import torchmetrics

# incase of a classification problem
metric = torchmetrics.Accuracy(task="multiclass", num_classes = 3)
for i, data in enumerate(dataloader, 0):
    features, labels = data
    outputs = model(features)
    acc = metric(outputs, labels.argmax(dim=-1))
acc = metric.compute() # on all data
metric.reset()
```

## Fighting overfitting

### Regularization using a dropout layer

- randomly a fraction of input neurons is set to zero at each update
- makes sure model doesn't rely too much on any one neuron

```Python
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Dropout(p = 0.2)
)
features = torch.randn((10, 8))
```

 - p is the probability that any one neuron is set to zero
 - usually added after the activation function


### Regularization with weight decay

- `optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 1e-4)`
- weight_decay is a hyperparameter that controls the amount of regularization, typically between 0 and 1
- additional term to teh parameter update step, encourage smaller weights
- higher, the less likely model will overfitting

### Data Augmentation

- typically upload to image data where images are rotated, flipped, zoomed, etc
- increases the size of the training set
- `torchvision.transforms` provides a variety of transformations

### Improving Model Performance

Step 1: Create a model that overfits the training set, so that you ensure problem is solvable, and sets a baseline
Step 2: Reduce overfitting, improve performances on validaiton set
Step 3: Fine-tune Hyperparameters

Step 1:

modify the training loop to overfit a single data point (batch size of 1)
    
```Python
features, labels = next(iter(trainloader))
for i in range(1e3):
    outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

- should reach 1.0 accuracy and 0 loss
- helps finding bugs in the code
- Goal: minimize the training loss
    - create large enough model
    - keep default learning rate

Step 2:

- make it generalize well to maximize validation accuracy
- Experiment with:
    - Dropout
    - Weight Decay
    - Data Augmentation
    - Early Stopping
    - Reducing model capacity

- Keep track of each hyperparameter and report maximum validation accuracy

Step 3:
    - you are happy with the model performance
    - fine-tune hyperparameters
    - if you have the computational resources, use grid search

```Python
for factor in range(2,6):
    lr = 10**-factor
```
- or random search, often leads to better results

```Python
factor = np.random.uniform(2,6)
lr = 10 ** -factor
```

```Python
values = []
for idx in range(10):
    # Randomly sample a learning rate factor between 2 and 4
    factor = np.random.uniform(2,4)
    lr = 10 ** -factor
    
    # Randomly select a momentum between 0.85 and 0.99
    momentum = np.random.uniform(0.85, 0.99)
    
    values.append((lr, momentum))
```