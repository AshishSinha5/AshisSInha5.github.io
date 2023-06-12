---
layout: post
title: Understanding Graph Convolutions
date: 2022-06-10 17:30
category: blog
headerImage: false
tag:
- algorithm
- gnn
- theory
---

{%- include mathjax.html -%}

## Leveraging Graphs for Machine Learning

Whenever we start building our implementing Machine Learning Algorithms to solve a particular problem, we start with the data at hand. For the most part, the data is tabular coming from single or multiple sources. We build our models using these data points with each row acting as a single sample. A typical workflow looks as follows - 


*insert pic*

There may be other forms of data structure available to us namely - 
- Image data structures - the images are of fixed dimensions and we apply either convert the pixel information to tabular form or apply convolutions on top of cubical images (RGB values) to get to extract the localized features of the images.

    *insert pic*
- Graph data structures - this kind of data structure usually arises when there is a direct relationship (edges) between the samples (nodes). E.g.

    - Citation network data where each node of the graph represents a document and each edge represents a citation relationship.

    - Social network data where each node is an individual and the edges represent a relationship between the individuals, eg. Friends, connections, etc. Unlike image data, the shape of these data structures varies with the number of nodes and edges and cannot be converted deterministically into a tabular structure. We can also not apply standard image convolutions due to the same reason.

The application of Machine learning on graphs leveraging their structure is actively researched in the community and we have come up with convolution neural networks that operate directly on graphs. To look closely into these kinds of neural networks we first investigate how a standard convolution works.

# What is a convolution?

**Convolution** is a mathematical operation on two functions ($f$ and $g$) that produces a third function ($f*g$) that expresses how the shape of one is modified by the other. 

For a convolution operation, we have a kernel $f$ and an input signal $g$, we reverse the input signal to create a _queue_ that slides into the kernel to compute the convolved value or _vice versa_. 

![1-D convolution](/assets/images/understanding_graph_convolutions/1D_convolution.gif)
*Fig.1 -  1D Convolution (flipping the kernel and sliding it across the input signal) [source](https://e2eml.school/convolution_one_d.html).*

For image processing, we use convolution to extract salient features of the image e.g. edges, lines, etc. We use cascading convolution operation for building CNNs which in turn are used to solve a variety of computer vision problems.

![2D Convolution](/assets/images/understanding_graph_convolutions/330px-2D_Convolution_Animation.gif)

*Fig.2 - 2D-Convolution used in CNNs [source](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution).*

# Generalizing convolutions for graph structure

- Adjacency matrix, incidence matrix, laplacian 
    - uses of laplacian
- laplacian as an operator 
    - examples 
- Spectral convolution 
    - Normalization of laplacian 
    - Fourier convolution
    - Drawbacks
- Tchebyshev's polynomial 
- Fast spectral convolution on graphs 


# Implementation in Python

# Implementation in PyTorch


# Reference
- Thomas N. Kipf, & Max Welling. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
- Xinye Chen (2021). Understanding Spectral Graph Neural Network.
- https://en.wikipedia.org/wiki/Convolution
- https://distill.pub/2021/understanding-gnns/
- https://distill.pub/2021/gnn-intro/
- https://csustan.csustan.edu/~tom/Clustering/GraphLaplacian-tutorial.pdf
- https://betterexplained.com/articles/intuitive-convolution/
