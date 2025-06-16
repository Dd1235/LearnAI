# CNN Architectures

Review: LeNet-5

Was used to apply digit recognition.

AlexNet - One of the first deep CNNs. (2012 paper)
- started the covnet spree.
- first use of ReLU
- they used norm layers (not common anymore)
- heavy data augmentaitons (flipping, jittering, cropping, etc.)
- dropout 0.5
- btach size 128
- SGD with momentum
- learning rate 1e-2 reduced by 10 manually when val accuracy plateaus
- L2 weight decay 5e-4
- 7 CNN ensemble: 18.2% -> 15.4%

VGGNet

- much deaper networks with much smaller filters
- 8 layers in alex net to 16-19 layers in VGG
- Only 3x3 filters with stride 1 and padding 1
- 2x2 max pooling with stride 2

- stack of 3x3 conv(stride 1) layers has the same receptive field as one 7x7 conv layer with stride 1

- fewer parameters 3*(3^2 C^2) vs 7^2 C^2 for C channels per layer
- Total memory footprint is about 96MB per image just for a forward pass
- 138M parameters

GoogleNet

- 22 layers
- computational efficiency
- efficient "inception" module
- no fc layers
- only 5 million parameters (12x less than alexnet)
- "inception module": design a good local network topology(network within a network) and then stack these modules on top of each other
- previous layer -> bunch of convolutoins or max pooling, 1x1 conv, 3x3 conv, 5x5 conv, 3x3 max pooling -> filtern concatenation
- problem here is hte depth size of the output

- bottle neck layers are added to decrease depth
- adds 1x1 conv before 3x3 and 5x5 convs to reduce the number of input channels
- after the pooling layer, 1x1 conv is used to reduce the number of channels

- so first a stem network, conv conv pool etc, then multiple inception modules stacked on top of each other, then classifier output, removed the fc layers
- auxillary outputs to inject additional gradients at lower layers.

ResNet

- much much deeper using residual connections
- 152 layers
- we expect deeper network with parameters to do better on training set since it might be more prone to overfitting
- But it does worse on both training adn validation sets
- Hypothesis: the problem is an optimization problem, deeper models are harder to optimize (not an overfitting one)

- The deeper model should be able ot perform at least as well as the shallower model. A solution by construction is copying hte learned layers from teh shallower model and setting additional layers to identity mappings.
 
- Solution: Use network layers to fit a residual mapping instead of directly trying to fit the desired underlying mapping.
- so if normally we get H(x) out, H(x) = F(x) + x, so we use the layers to fit F(x) instead of H(x)
- its like we learn teh residual from our input x
- for deeper networks (50+) they also use bottle neck layers

- use batch norm after every conv layer
- Xavier/2 initialization
- SGD + momentum
- learning rate 0.1, reduced by 10 when validation accuracy plateaus
- mini-batch size 256
- L2 weight decay 1e-5
- no dropout 

- look into ResNext (lmao love the pun)
- also Network in Network (NIN) , stochastic depth etc, interesting ideas. also the auxillary outputs thing is interesting. FractalNet too. Dense net.

Efficient Networks 
- Squeeze net


# RNN

- look at the paper draw: a recurrent neural network for image generation
- x-> rnn(curcular loop,update hidden state) -> output
- h(t) = f_W(h_t-1, x_t) = tanh(W_hh * h_t-1 + W_xh * x_t + b_h)
-  Seq to Seq: many to one + one to many
- encode input seq in a single vector, then produce output sequence from a single input vector


- we do truncated backpropagation through time. Run forward and backward through chunks of the sequence instead of whole sequence.

- image captioning
- image captioning with attention - cnn gives a grid of vectors instead of just one, that give a vector for each spatial location, so in addition to sampling hte vocabulary, gives a distribution over the locations where it wants to look. Which goes back to the image and gets the features from that location. So it can focus on different parts of the image while generating the caption.

- so distribution over vocab and distribtion over L locations  taking in z and y (z is the image features, y is the previous word in the caption) -> attention mechanism

- soft attention vs hard attention

- soft attention: use the distribution over locations to get a weighted sum of the features from the image, then use that to generate the next word in the caption
- hard attention: sample a location from the distribution, then use that location to get the features from the image, then use that to generate the next word in the caption. This is harder to train since we need to use REINFORCE or some other method to train it.

## lec 11

- look at mask rcnn, and some r cnn, fast rcnn, faster rcnn.

## lec 12 visualizing and understanding 

- use pca or tsne to plot the features of the last say 1x1x4096 vector you get at the end and look at the clusters
- for the very first layer, since its directly connected to the input, you cand directly visualize each of the kernels.

- for immediate layers, can visualize teh activation maps instead.

- saliency maps
- guided    backpropagation
- gradient ascent


(go through slides its a time waste to type all this i guess)