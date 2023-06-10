---
layout: post
title: Understanding Graph Convolutions
date: 2022-03-06 17:30
category: blog
headerImage: false
tag:
- algorithm
- gnn
- theory
---

{%- include mathjax.html -%}

# Leveraging Graphs for Machine Learning

Whenever we start building our implementing Machine Learning Algorithms to solve a particular problem, we start with the data at hand. For the most part, the data is tabular coming from single or multiple sources. We build our models using these data points with each row acting as a single sample. A typical workflow looks as follows - 

*insert pic*

There may be other forms of data structure available to us namely - 
    - Image data structures - the images are of fixed dimensions and we apply either convert the pixel information to tabular form or apply convolutions on top of cubical images (RGB values) to get to extract the localized features of the images.
    *insert pic*
    - Graph data structures - this kind of data structure usually arises when there is a direct relationship (edges) between the samples (nodes). E.g. - 
        - Citation network data where each node of the graph represents a document and each edge represents a citation relationship.
        - Social network data where each node is an individual and the edges represent a relationship between the individuals, eg. Friends, connections, etc. Unlike image data, the shape of these data structures varies with the number of nodes and edges and cannot be converted deterministically into tabular structure. We can also not apply standard image convolutions due to the same reason.

The application of Machine learning on graphs leveraging their structure is actively researched in the community and we have come up with convolution neural networks that operate directly on graphs. To look closely into these kinds of neural networks we first investigate how a standard convolution work.

# What is a convolution?

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
