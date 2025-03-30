Attempt at implementing [this](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) paper.

# Current implementation

For the sake of faster training time
- used a smaller CNN
- did not implement hyperparameter tuning, used contrasitive loss instead of BCE+L2 regularization
- did not implemented early stopping, and used smaller epochs (5)
- used Adam instead of sgd, since paper used sgd with momentum gradually increasing from 0.5 and learning rate decaying by 1% each epoch, also L2 weight decay
- did not implement affine distortions, just ToTensor() and Resize()
- did not visualize embeddings with pca or tsne


# Result

One shot 20 way accuracy is 80%

That is during the testing time, you are given 1 query image, and 1 example each from 20 differnt classes (support set), 
it has to pick one that is most similar to.
Training is done on characters different from what it sees here, where it learns how to compare two images.



# Attempt at Understanding the paper


Problem: little data available, what if one image per class(for one-shot learning)
Solution: learn the similarity function, when given new data, finds the image it is most similar too to assign class

doubt: so a vanilla siamese network can't find new classes? Or you have a threshold beyond which assume this is new class?
doubt: say A and ß (German alphabet), then B would be misclassified as German alphabet is it?
doubt: how would these models work out when languages have similar scripts? Telugu Kannada, or English German French Spanish etc.

Understanding: 
    - instead of learning what characterizes a class, you compare how similar the image is to existing images to determine if the two images are of the same class
    - It is different from multiclass classification. If you train it on apples and oranges, it will still be able to put together two bananas while testing.

Siamese neural networks: type of nn designed to determine if two inputs are similar

    - First introduced for signature similarity

    - Two subnetworks/twin networks/towers with same architecture and weights process two input images and output feature representation, then use a similarity metric to compute distance

For image verification, CNN subnetworks since they are good at capturing local spatial features, preserve spatial hierarchy, and are translation-invariant.

image (x1) -> CNN -> embedding (f(x1))

CNN used:

conv1: kernel size 10x 10, 64 filters, so output 64 feature maps.
consider 105x105 input, then 105-10+1 = 96x96 (no padding), each feature maps
+ ReLU

doubt: why no padding? isn't it better to give edges also implementing

Max Pool1 : 64@2x2 so 48x48 output

Conv2 128 filters, 7x7, so 48-7+1, 42x42 feature maps
+ ReLU

MaxPool2: 64@2x2, so output 21x21 (how 64 should be 128??)

Conv3 + relu : 128 @ 4x4 so 18x18 output

MaxPool3 64@2x2 9x9

Conv4 + relu 256@ 4x4 so 6x6

Faltten 256x6x6 = 9216

FC layer projects to lower dimension


fully connected + sigmoid, L1 siamese dist 

L1 distance: element wise absolute differnce between two vectors


feature fector 4096,
fully connected + sigmoid ( to get 0 or 1)
output 1x1

RELU is element wise, max(0,x)

(this is a mess, refer to figure 4 of paper)

We use cross-entropy loss (for binary classification)
There is an L2 regularization term (weight decay) that is added to cross entropy loss to get total loss to prevent overfitting by penalization large weights

Intialize all weights in cnn from normal distributoin wtih zero mean and 0.001 standard deviation. Biases - N(0.5, 0.001). 
FC - same biases, but weights N(0, 0.2), so larger standard
deviation

Learning rate: wnew = wold - $ \eta $ * gradient

allowed different learning rate for each layer, but decayed uniformly across the network by one percent

Momentum - start at 0.5 every layer, increasing linearly each epoch till a certain value
With momentum there is a velocity term, velocity accumulates previous gradients, builds speed,helps making sure you don't get stuck at local minima.

velocity = μ·v - η·grad, then w = w + velocity

Each network - maximum 20o epochs, monitored one-shot  alidation error on a set of 320 one-shot learning tasks generated randomly 

Early stopping: if validation error did not decrease for 20 epochs, stopped and used the parameters of the model at teh best epoch according to one-shot validation error.

Hyperparameter optimization:

Learning rate, momentum, L2 regularization
Bayseiam optimization: models the performance curve adn smartly picks the next hyperparameters to try, more data-efficient that grid search

Affine Distortion: to help generalize, add random distorsions to characters, each image is a pair is randomly transormed, by rotation, adding shear, scale, transaltion etc.

