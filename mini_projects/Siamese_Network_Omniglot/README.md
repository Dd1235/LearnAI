Attempt at implementing [this](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) paper.

This is a work on progress, I will slowly remove the simplifications based from the paper.
[deleted the dataset from here]
# Siamese Networks Intuition

Siamese networks consist of twin neural networks/ towers with same weights and architecutre(CNN for image classification). If you have two images you pass each through each of the towers to generate embeddings. Then you use a fully connected layer, and, some distance measure to get the similarity between the two images. So your neural network essentially learns a "similarity function", rather than what makes up a class.

So say you have a bunch of images of apples and oranges, a multi class classifier would learn what an apple/orange is supposed to look like. But the siamese network will learn a way to compare the images, such that two apples get a higher similarity score, and an apple and an orange lower. Now during testing, if given some images, of apples, and oranges, and a query image of apple, it will tell you that the query image falls under the same class as the apple image. Something to be noted is that, now if I give it an image of an apple, orange, and banana, and query with a banana image, it would tell me that the query comes under the same class as the banana.


# Siamese Network for One-Shot Classification on Omniglot 

This project implements a Siamese Neural Network for one-shot image classification using the [Omniglot dataset](https://github.com/brendenlake/omniglot). It is inspired by the paper **"Siamese Neural Networks for One-shot Image Recognition"** by Koch et al. (2015), and built using **PyTorch**.

---

##  Project Highlights

- **Architecture**: Siamese Convolutional Neural Network (CNN)
- **Objective**: Learn a similarity function to classify unseen characters with only one example per class
- **Approach**:
  - Custom CNN used to generate image embeddings
  - Contrastive Loss to train the similarity function
  - L2 distance metric to compare image embeddings
- **Evaluation**:
  - One-shot **20-way classification**
  - Accuracy achieved: **~80%**

---

## Results

| Metric                      | Value      |
|----------------------------|------------|
| Accuracy (20-way one-shot) | ~80%       |
| Loss Function              | Contrastive|
| Optimizer                  | Adam       |
| Epochs                     | 5          |
| Distance Metric            | L2         |

During testing, the model is given a **query image** and 20 candidate images (one from each class). It must choose the one most similar to the query. The model is trained only on classes different from those seen during testing.

---

## Simplifications and Deviations from the Original Paper

- Used smaller CNN for faster training
- No hyperparameter tuning or early stopping
- No affine distortions (used only `ToTensor()` and `Resize()`)
- Used Adam optimizer instead of SGD with momentum
- L2 distance instead of L1
- Skipped weight initialization and visualization (e.g. t-SNE/PCA)

---

## Files

- `SiameseNetwork.ipynb`: Main training and evaluation notebook
- `Omniglot Dataset/`: Preprocessed dataset for training and evaluation
- `oneshot1.pdf`: Notes/understanding of the paper and concepts
- `README.md`: You're reading it!

---

## How to Run

```bash
# Setup your environment (conda or venv)
pip install torch torchvision matplotlib

# Run the notebook
jupyter notebook SiameseNetwork.ipynb
```
---

## Improvements to be done

- Add visualization of embeddings using **t-SNE** or **PCA**
- Hyperparameter tuning and learning rate scheduling
- Use more complex CNN backbone like ResNet
- Implement affine transformations for data augmentation like described
- Try Binary Cross Entropy + L2 regularization


# Attempt at Understanding the Paper

**Problem**: Little data available — what if there's only one image per class (for one-shot learning)?  
**Solution**: Learn a similarity function. When given new data, find the image it is most similar to and assign the same class.

**Doubt**: So a vanilla Siamese network can’t find new classes? Or is there a threshold beyond which it assumes this is a new class?  
**Doubt**: Say A and ß (German alphabet) — then B would be misclassified as a German alphabet character, right?  
**Doubt**: How would these models work when languages have similar scripts? Like Telugu-Kannada, or English-German-French-Spanish, etc.?

**Understanding**:
- Instead of learning what characterizes a class, you compare how similar the image is to existing images to determine if the two images are of the same class.
- It is different from multi-class classification. If you train it on apples and oranges, it will still be able to group two bananas together during testing.



### Siamese Neural Networks

A type of neural network designed to determine if two inputs are similar.

- First introduced for signature similarity.
- Two subnetworks (twin networks/towers) with the same architecture and shared weights process two input images and output feature representations. Then, a similarity metric computes the distance between them.

For image verification, CNNs are used as subnetworks because they are good at:
- Capturing local spatial features
- Preserving spatial hierarchy
- Being translation-invariant

`image (x1) -> CNN -> embedding (f(x1))`


### CNN Used (this is a mess, refer to Figure 4 of the paper):

- **Conv1**: kernel size 10x10, 64 filters → 64 feature maps  
  Input: 105x105 → Output: 96x96 (no padding)  
  + ReLU

**Doubt**: Why no padding? Isn’t it better to capture edge information too?

- **MaxPool1**: 64 @ 2x2 → Output: 48x48
- **Conv2**: 128 filters, 7x7 → 48 - 7 + 1 = 42x42  
  + ReLU
- **MaxPool2**: Supposed to be 128 @ 2x2 → Output should be 21x21  
  > (Why does the paper mention 64 here? Shouldn’t it be 128?)
- **Conv3 + ReLU**: 128 @ 4x4 → Output: 18x18
- **MaxPool3**: 128 @ 2x2 → Output: 9x9
- **Conv4 + ReLU**: 256 @ 4x4 → Output: 6x6

- Flatten: 256 x 6 x 6 = 9216  
- FC layer: Projects to a lower-dimensional feature vector

Then, fully connected + sigmoid, L1 Siamese distance.  
L1 distance = element-wise absolute difference between the two vectors  
Feature vector: 4096  
Final fully connected layer + sigmoid → Output: 1x1

ReLU is element-wise: `max(0, x)`



### Loss

- We use **cross-entropy loss** (for binary classification)
- There is an **L2 regularization** term (weight decay) added to prevent overfitting by penalizing large weights



### Weight Initialization

- Initialize all CNN weights from normal distribution:  
  `N(0, 0.001)`  
- Biases: `N(0.5, 0.001)`  
- FC layer:  
  - Weights: `N(0, 0.2)` (larger std deviation)  
  - Biases: same as above



### Hyperparameters

- **Learning Rate**:  
  `w_new = w_old - η * gradient`  
  Different learning rates allowed for each layer but decayed uniformly across the network by 1% per epoch

- **Momentum**:  
  - Starts at 0.5 for every layer  
  - Increases linearly every epoch until a certain value  
  - Helps avoid getting stuck in local minima

```
velocity = μ·v - η·grad  
w = w + velocity
```



### Other Stuff

- Each network trained for a **maximum of 20 epochs**
- Monitored **one-shot validation error** on a set of **320 one-shot learning tasks** generated randomly
- **Early stopping**: If validation error didn’t decrease for 20 epochs, training was stopped. The model from the best epoch (lowest validation error) was saved.



### Hyperparameter Optimization

- Tuned: Learning rate, momentum, L2 regularization
- Used **Bayesian optimization**: models the performance curve and smartly picks the next hyperparameters to try (more data-efficient than grid/random search)



### Affine Distortion

- To improve generalization, apply random transformations to characters during training
- Each image pair is randomly transformed by:
  - Rotation
  - Shear
  - Scaling
  - Translation  
  etc.

