---
layout: post
title: Evidential Deep Learning
date: 2022-03-06 17:30
category: blog
headerImage: false
tag:
- algorithm
---

{%- include mathjax.html -%}

# Introduction
# Example
Let's take an example of building a classification model. We start with a set of annotated data for training the model and predict on previously unseen data. E.g. we can train an image classification model predicts whether a given handwritten digit belongs to one of the ten classes representing integer values from 0 to 9.

Leveraging open-sourced deep learning library, we build a multi-class classification model for the given problem statement. A sample of training data looks something like this - 

![Sample dataset for the Classification task](/assets/images/mnist_sample.png)

Predicting labels for some of the images -

|--|--|--|
|![Sample Four](/assets/images/sample_four_pred_true.png)|![Sample Five](/assets/images/sample_five_pred_true.png)|![Sample eight](/assets/images/sample_eight_pred_true.png)|

Now if we provide some samples which do not match the distribution of data that the model saw during training it may misclassify those samples. E.g. when we rotate the same images by some degrees we get - 

|--|--|--|
|![Rot Four](/assets/images/rot_four_pred_true.png)|![Rot Five](/assets/images/rot_five_pred_true.png)|![Rot eight](/assets/images/rot_eight_pred_true.png)|

As we see in the above example the rotated images of digits are misclassified two out of three times. The below visualization gives us an extensive view of the above problem - 

![EDL-Rot-one](/assets/images/edl_rot_1.png)

We see that the model predicts wrong labels for the digit one with high confidence as we increase the rotation angle of the image. To solve the above problem, an **evidential deep learning** technique is proposed where the model is incorporated with the capability of assessing the **uncertainty** of the input distribution.

# What is Evidential deep learning

The above problem of predicting the wrong labels with high confidence can be solved by introducing a mechanism to assess the uncertainty of the model output. There are two types of uncertainties - 

- **Aleatory uncertainty**: This is the type of uncertainty that is inherent in a system or process and arises due to the natural randomness or variability of the system. For example, the outcome of a coin toss or a dice roll is aleatory uncertainty.

- **Epistemic uncertainty**: This type of uncertainty arises due to a lack of knowledge or information about a system or process. It can occur when there are limitations in measurement techniques, incomplete or inaccurate data, or when the system is poorly understood.

# How do we account for uncertainty



# Applications
# Demo
# References
