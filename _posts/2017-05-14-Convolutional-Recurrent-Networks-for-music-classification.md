---
layout: post
title:  "Convolutional Recurrent Networks for Music Classification"
date:   2017-05-16 10:10:00 -0700
categories: paper summary
---

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

![Model]({{ site.url }}/_assets/music_rnn_models.png)

The models are scaled by controlling the number of parameters to be 0.1M, 0.2M, 0.5M, 1M and 3M with 2% tolerance. The models are kept of similar depth (to enable models to use features of same depth/hierarchy while changing the number of such features available).

![Model Size]({{ site.url }}/_assets/music_rnn_results.png)

However, within each model there are several ways to change number of parameters (increase num of features vs decreasing max pooling size, increasing size of RNN hidden units vs increasing conv features etc). These minor details have been omitted in the paper. 

The models are trained using ADAM with default parameters. Binary cross entropy is used as the loss function. All models are trained with early stopping - the training is stopped if there is no improvement of AUC on the validation set while iterating the whole training data once. 

3. What kind of experiments are they running? What kind of results are they getting?

Songs are trimmed to 30-60s preview clips. The audio signals are trimmed to 29 seconds at the centre of preview clips and down- sampled them from 22.05 kHz to 12 kHz using Librosa. Log-amplitude mel-spectrograms are created from this downsampled data. The number of mel-bins is 96 and the hop-size is 256 samples, resulting in an input shape of 96×1366.

On a single dataset from Last.fm, their proposed CRNN model beats state of the art results using 0.5 million parameters. The CRNN model consistently performs better than other alternatives such as k2c2, k2c1, and k1c2, given the same number of parameters. k2c2 follows in performance, with minimal increase in AUC scores(1st digit of decimal). k2c2 however is slightly better than the remaining two models(unit difference). Given that the run time for a single epoch of the CRNN model is 2x -4x of the k2c2 model, it seems ideal to start off with k2c2 model for most cases. 

![Results]({{ site.url }}/_assets/music_rnn_auc.png)

5. What previous work are they building on

6. What are some of the key proofs and assumptions on the data. Any drawbacks from real world application of the model. 

No key proofs in the paper. The assumptions around using a combination of CNNs and LSTMs to capture short term and long term dependencies seems reasonable. However, given that k2c2 achieves nearly the same accuracy, the model choice is not entirely justified. The application from music classification can be extended to other time series or sequential applications. However, the pre-processing steps seem specific to short time high frequency signals(like speech, music etc). Appropriate techniques for lower frequency/sampling signals needs to be investigated.

7. Starting with raw data, what steps does one have to do to recreate the results.

The paper is straightforward to implement and test out. The model options and parameters and pre-processing steps are clearly explained.
