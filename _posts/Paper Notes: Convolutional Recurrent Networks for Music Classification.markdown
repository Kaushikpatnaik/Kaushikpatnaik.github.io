
Notes and review of the paper: "Convolutional Recurrent Networks for Music Classification"

Summary:
The paper compares few different deep learning architecture choices for music classification. Music classification is a task where the objective is to predict tags corresponding to a short (10 sec) music clip. Tags could be related to the artist, music genre, mood, album etc. 

The paper carries out carefully controlled experiments (around parameter size, training times etc) to show that an architecture combining 2d convolutions and Bi-directional LSTMs performs better than pure convolutional architectures. 

1. What is the general idea they are using to solve the problem.

Convolutional layers to capture short term dependencies across time and frequency space, and LSTMs to capture long term dependencies. A combination of both these model types helps solve model far better than just the combination of a single model.

2. What are the model types they are comparing. Does it match the setup of the problem. Any obvious drawbacks?

The paper tests four architectures combining 1d convolutions, 2d convolutions, fully connected layers and LSTMS. The four architectures are
* i) k1c2
    - Two dimensional convolution (c2) with one dimensional kernel (k1). Max pooling follows convolutional laers. Followed by two fully connected layers. The convolution is two dimensional as you move the 1d kernel across both the dimensions.
* ii) k2c1
    - One dimensional convolution (c1) with two dimensional kernel (k2). Max pooling follows convolutional layers. Output of convolutional layers flattened and followed by two fully connected layers. Here the 2d kernel completly covers one of the dimensions (i.e it is wide convolution) and the convolution is carried out across the other dimension. 
* iii) k2c2
    - Two dimensional convoltion with two dimensional kernel. Max pooling follows convolutional layers. No fully connected layers follow it. Traditional CNN. As in traditional CNN, the kernel is much smaller than the input and it is translated over the entire input dimensions.
* iv) CRNN
    - k2c2 followed by two LSTM layers. The authors use a one directional LSTM, and the last layer is connected to the output.

All convolutional layers use batch normalization and exponential learning units (ELUs). 

Insert Diagram on model types.

The models are scaled by controlling the number of parameters to be 0.1M, 0.2M, 0.5M, 1M and 3M with 2% tolerance. The models are kept of similar depth (to enable models to use features of same depth/hierarchy while changing the number of such features available).

Insert Diagram on model size. 

However, within each model there are several ways to change number of parameters (increase num of features vs decreasing max pooling size, increasing size of RNN hidden units vs increasing conv features etc). These minor details have been omitted in the paper. 

The models are trained using ADAM with default parameters. Binary cross entropy is used as the loss function. All models are trained with early stopping - the training is stopped if there is no improvement of AUC on the validation set while iterating the whole training data once. 

3. What kind of experiments are they running? What kind of results are they getting?

On a single dataset from Last.fm, their proposed CRNN model achieves 



4. What previous work are they building on
5.  
6. What are some of the key proofs and assumptions on the data. Any drawbacks from real world application of the model. 
7. Starting with raw data, what steps does one have to do to recreate the results.