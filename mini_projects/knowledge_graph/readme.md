# Graph NN

An image can be thought of as a grid graph where each pixel is connected to 8 of its neighbours.

In a CNN, its like you look at the neighbours of a region, and interpret something about it. Then next layer, look at the neighbours of hte neighbours.

Maybe those with similar characteristics would be closer.
Graph NN - node embeddings, space of all possible embeddings - embedding space, closer ones would be closer.

So we need an objective distance that we need forr a loss function.

Backpropagate the loss and update the node embeddings.

Message passing: take information from neighbours, aggregte, and update the node. 

# Paper

- 