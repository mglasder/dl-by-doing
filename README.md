# Deep Learning by doing: Implementations of Deep Learning models in Pytorch

Welcome to Deep Learning by doing! In this repository, I explore the practical implementation of deep learning 
algorithms from (scientific) papers, blog posts and the like. Join me as I delve into cutting-edge techniques, unravel complex concepts, 
and build easy-to-read code in PyTorch and PyTorch Lightning.

## Implementations

#### Bengio et al. (2003) - Neural Probabilistic Language Models

```
|–– dl-by-doing
|   |–– dl  
|   |   |–– ...
|   |   |-– bengio2003
|   |   |–– ...

```

#### Ioffe and Szegedy (2015) - Batch Normalization

```
|–– dl-by-doing
|   |–– dl  
|   |   |–– ...
|   |   |-– batch_norm
|   |   |–– ...

```

## Abstracts

#### Bengio et al. (2003) - Neural Probabilistic Language Models

https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

Abstract: 

A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language. 
This is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be 
tested is likely to be different from all the word sequences seen during training. Traditional but very successful 
approaches based on n-grams obtain generalization by concatenating very short overlapping sequences seen in the 
training set. We propose to fight the curse of dimensionality by learning a distributed representation for words which 
allows each training sentence to inform the model about an exponential number of semantically neighboring sentences. 
The model learns simultaneously (1) a distributed representation for each word along with (2) the probability function 
for word sequences, expressed in terms of these representations. Generalization is obtained because a sequence of words 
that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a 
nearby representation) to words forming an already seen sentence. Training such large models (with millions of 
parameters) within a reasonable time is itself a significant challenge. We report on experiments using neural 
networks for the probability function, showing on two text corpora that the proposed approach significantly improves 
on state-of-the-art n-gram models, and that the proposed approach allows to take advantage of longer contexts.

    
#### Ioffe and Szegedy (2015) - Batch Normalization

https://arxiv.org/abs/1502.03167

Abstract:

Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during 
training, as the parameters of the previous layers change. This slows down the training by requiring lower learning 
rates and careful parameter initialization, and makes it notoriously hard to train models with saturating 
nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer 
inputs. Our method draws its strength from making normalization a part of the model architecture and performing the 
normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less 
careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to
a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer 
training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, 
we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error 
(and 4.8% test error), exceeding the accuracy of human raters.