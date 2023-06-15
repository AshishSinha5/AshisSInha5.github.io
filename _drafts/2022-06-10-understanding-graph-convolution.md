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

![1-D convolution](/assets/images/understanding_graph_convolutions/1D_convolution.gif){:style="display:block; margin-left:auto; margin-right:auto"}
<div align = 'center'>Fig.1 -  1D Convolution (flipping the kernel and sliding it across the input signal) <a href = "https://e2eml.school/convolution_one_d.html">[source]</a>.</div>

For image processing, we use convolution to extract salient features of the image e.g. edges, lines, etc. We use cascading convolution operation for building CNNs which in turn are used to solve a variety of computer vision problems.


![2D Convolution](/assets/images/understanding_graph_convolutions/330px-2D_Convolution_Animation.gif){:style="display:block; margin-left:auto; margin-right:auto"}
<div align = 'center'>Fig.2 - 2D-Convolution used in CNNs 
<a href="https://en.wikipedia.org/wiki/Convolution#Discrete_convolution">[source]</a>.</div>


We note these convolution work as feature extractors and operate on inputs of fixed size. We aim to generalize these convolutions to graphs where we work with variable-size inputs. These graph convolutions will act as *feature aggregators* for a node and its neighbors. 

# Generalizing convolutions for graph structure

We define graphs as data structures consisting of tuple $G = (V, E)$ where $V$ is a finite set of elements and $E$ is a subset of $(V \times V)$. Graphs occur naturally in many domains such as social networks, citation networks, molecular structure, routes,  etc.

The study of graphs is one of the main focus areas in mathematics and lately, the ML community is trying to leverage the native graph structure of the data to extract some patterns out of it.

In this section, we aim to generalize convolutions on graphs which will enable us to perform various graph machine learning techniques on it. We begin by laying down some foundation.

For the entire section we'll consider a graph $G = (V, E)$ with $V = \\{v_1, v_2, v_3, ..., v_n\\}$ with  $\|V\| = n$ and $\|E\| = m$.

Graphs are represented in many ways - 
- Adjacency Matrix
- Adjacency List
- Edge List

## Adjacency Matrix
Adjacency Matrix is a $n \times n$ matrix $A_G = a_{ij}$ where 

$$
\begin{equation*}
    a_{ij} = \begin{cases}
    w_{ij}, & \text{if $e_{ij} \in E$. } \\
    0, & \text{otherwise}
    \end{cases}
\end{equation*}
$$

For an undirected graph, $A$ is symmetric.

## Functions on a graph

Given a graph $G$ with vertex set $V$ with $k$ dimensional feature vector of vertex $i$ as $v_i \in R^k$. A function on this graph is defined as - 
$$
f : V \rightarrow R^k \\
f = (f(v_1), f(v_2), f(v_3), ..., f(v_k))
$$

## Incidence Matrix
Incidence Matrix is a $|E| \times |V|$ $(m \times n)$ matrix $\triangledown$ where -

$$
\begin{equation*}
  \triangledown_{ev} =\begin{cases}
    -1, & \text{if $v$ is the initial vertex of edge $e$.}\\
    1, & \text{if $v$ is the terminal vertex of edge $e$.}\\
    0, & \text{if $v$ is not in $e$}
  \end{cases}
\end{equation*}
$$
*add example.*

Incidence matrix is known as a discrete differential operator, i.e. for any function $f$ on the vertex set $V$, $(\triangledown f)e_{ij} = f(v_j) - f(v_i)$



## Laplacian 
Laplacian of a matrix a given by 

$$
L = \triangledown^T\triangledown
$$

and is related to the adjacency matrix $A$ by the following relation  - 

$$
L = D - W
$$


where $D = D_{ii} = d(i)$ and $W$ is the edge weight matrix.
When we talk about the laplacian we usually mean normalized Laplacian matrix given by - 

$$
\begin{align*}
L =& D^{-1/2}LD^{-1/2} \\
  =& I - D^{-1/2}WD^{-1/2}
\end{align*}
$$

### Laplacian as an operator and quadratic form

The laplacian can viewed as an operator $Lf = (D - A)f$ and we can prove that -

$$
Lf(v_i) = \Sigma_{j = 1}^{n} w_{ij}(f_i - f_j)
$$

It can also be viewed as a quadratic form - 

$$
f^TLf = \Sigma_{j = 1}^{n}w_{ij}(f_i - f_j)^2
$$

We see that 

### Laplacian as an operator

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
