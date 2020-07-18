# DEPICT-PyTorch
A PyTorch Implementation of DEPICT cluster loss

Ghasedi Dizaji, K., Herandi, A., Deng, C., Cai, W. and Huang, H., 2017. Deep clustering via joint convolutional autoencoder embedding and relative entropy minimization. In Proceedings of the IEEE international conference on computer vision (pp. 5736-5745).

DEPICT is an unsupervised discriminative clustering algorithm. At each step it performs an expectation maximization where the true density is assumed to be known by approximating the gradient to find a close form solution.

The original paper is implemented in Theano, uses 3 architectures and follows a pretty complicated training pipeline. This implementation is using PyTorch and is a pretty straightforward batch-wise implementation of DEPICT using 2 architectures only. Despite its simplicity, it performs really well.

# Visualiztions

Trained an autoencoder using reconstruction loss and depict clustering loss. Obtained T-SNE plots on 5000 samples in input space and encoding space.

![before](https://github.com/kartikeya-badola/DEPICT-PyTorch/blob/master/before.png)

T-SNE Plot at input space. Notice how 4 and 9 as well as 3, 5 and 8 have mixed up clusters (as expected since handwritten 4 and 9 are pretty similar looking. Same goes for 3,5 and 8)

![after](https://github.com/kartikeya-badola/DEPICT-PyTorch/blob/master/after.png)

T-SNE Plot at encodings of the same samples. Notice how the seperation between the clusters have increased dramatically (even between 4 and 9 and 3, 5 and 8). This was only after 20 epochs.

