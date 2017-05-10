---
layout: post
title:  "End to End Sequence Labeling via Bi-Directional LSTM-CNNs-CRF"
date:   2017-05-09 10:10:00 -0700
categories: paper summary
---

Overall Summary:
The first end to end sequence labelling model that outperforms models that have a combination of hand-crafted transformations and deep learning.

The basic model is comprised of three things
* A word embedding derived from character embeddings using convolution
* Bidirectional LSTM
* CRFs for sequence labeling/decoding

It's interesting that they apply the same model to two different kinds of problems, which share a common basis of being sequence labeling tasks. Obtains near SOTA accuracy on two well known datasets - 97.55% on POS and 91.21% for NER

The previous best for POS was Semi-supervised condensed nearest neighbor for POS tagging, with accuracy of 97.50%.
The previous best for NER was based on local and global linguistic features, with accuracy of 91.20%.

## 1. What problem is the paper solving?

Sequence Labeling: Given a sentence(a sequence of words) and associated labels, train a model to predict labels. The paper focusses on NLP sequence labeling tasks of predicting Parts of Speech(POS) and Named Entity Recognition(NER).

## 2. What is the general idea they are using to solve the problem? 

Instead of using linguistic features like previous works, they aim to train a truly end to end model using deep learning.

To this effect, they start off with concantation of character and word embeddings to represent each word. The character representations are made from convolutions of individual char representations that are pre-trained.

These are then fed to a bi-directional LSTM to capture the temporal order characteristics.

The output states are then fed to a CRF system. The intentin of CRF is to use/capture conditional probabilities in labels ie. a Noun has a higher probability of being followed by a Verb or Adjective. The objective is to decode the labels as a sequence rather than one at a time.

Dropout (probability=0.5) is used to add regularization to the model while connecting between the larger components described above.

## 3. What previous work are they building on?

There have been papers using CNN-BLSTMs (Chiu et. al, Named Entity Recognition with Bidirectional LSTM-CNNs), and BLSTMS-CRFs (Huang et al. Bidirectional lstm-crf models for sequence tagging) before. The SOTA papers tend to use liguistic features in addition to the DL models. BLSTMs are by now well known in most language tasks as most state of the art papers use them. The inclusion of CRFs and CNNs on  the other hand are very recent developments.

## 4. What is the model? 

Pre-trained character and word embeddings. The character embeddings that represent a word are gathered together(with padding if required). A convolution filter, post dropout, is applied with kernel size of 3 chars and 30 feature maps. A max pooling layer across the characters then constructs the character representation of the word. 

![CNN Layer]({{ site.url }}/_assets/CNN-LSTM-CRF-CNN-Embedding.png)

The above character representation is then concated together with pre-trained word embeddings like Glove, Google Word2Vec. 

Thus a sentence now represents a sequence of word level embeddings, which can be fed to a Bi-Directional LSTM post dropout. The paper uses 200 hidden units, with initial states as zeros.

Finally the output of the Bi-directional LSTM layer at each step is inputted to the CRF layer post dropout.

The model architecture is shown below
![Model Architecture]({{ site.url }}/_assets/CNN-LSTM-CRF-Overall-model.png)

They train the model using simple SGD with momentum of 0.9 and batch size of 10. They lower the learning rate based on epochs as lr_t = lr_0/(1+tau*t), with tau = 0.5 and t being the epoch being completed. Gradient clipping of 5.0 is applied(not sure if just for BLSTMs, or for all). 

## 5.What experiments they are running. Does it match the setup of the problem. Any obvious drawbacks?

The authors choose two tasks for evaluating their model: Parts of Speech Tagging and Named Entity Recognition. 

For POS tagging they use the english portion of the Wall Street Journal (PTB), which contains 45 different POS tags. They adopt standard training and testing splits as previous work.
For NER they perform experiments on the english data from CoNLL 2003 shared task. This dataset contains four different kinds of named entities: PERSON, LOCATION, ORGANIZATION, and MISC. 

No Pre-processing was done one any of the datasets.

The authors dive into the performance improvements due to their model choices by 
a) comparing RNNs with LSTMs and BiDirectional LSTMs
b) Inclusion of CNN pre-processing step vs running a simple model with BiLSTMs and CRFs
c) CNN-LSTMs without CRFs and models with CRFs
d) Dropout vs No-Dropout
e) Generalization ability by looking at OOV vs Non-OOV words

## 6.What kind of results are they getting. Any drawbacks?

![NER Score]({{ site.url }}/_assets/CNN-LSTM-CRF-NER-Score.png)
![POS Score]({{ site.url }}/_assets/CNN-LSTM-CRF-POS-Score.png)
![Comparison of Models]({{ site.url }}/_assets/CNN-LSTM-CRF-Comaprison of models.png)

Their results are on par with previous SOTA models which use a combination of linguistic features and deep learning to achieve good performance. Their main advantage is the truly end to end system from raw text to predicted labels. 

The improvement on the NER and POS datasets are minimal (0.01 and 0.05), but are notable for being purely DL based results. Among the rest of the abalation studies, the most prominent results are from the inclusion of the CNN modules in the model. The rest of the comparisons show minimal improvement. It's hard to say if the model performance will improve with larger datasets. However the thoughts behind the model composition seem to work out.

## 7. What are some of the key proofs and assumptions on the data. Any drawbacks from real world application of the model. 

No key proofs, purely experimental paper. No inherent assumptions. It is unclear to me how the CRF training and decoding takes place, and whether it can be replaced by a generic decoder like a sequence to sequence network. The choice of the number of feature maps in the initial convolutional layer seems arbitary, and its unclear if max pooling is just over the 

## 8. Starting with raw data, what steps does one have to do to recreate the results.

Given a corpus, we need to
* Split into training, validation and testing sets
* Convert each word to their respective embedding based on Glove or Google word2vec
* Initialize the character embeddings with random uniform sampling from [-sqrt(10), sqrt(10)]
* Create the convolutional and BLSTMs layers as described in sections above, with random weight initialization as per Glorot et al. 2010
* Join the output of the BLSTM to the CRF layer
* Set the training to maximize the likelihood of observed labels given the BLSTM output and CRF weights
* Train based on the description given in sections above
* At test time, decode the output sequence using Viterbi
