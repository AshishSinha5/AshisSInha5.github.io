---
layout: post
title:  "Negative Sampling"
date:   2021-06-16 11:15:31 +0530
---

### Introduction

Say you are training a NLP model with 100s of millions of tokens in your vocabulary. Let us assume for the sake of simplicity that the NLP model is a bigram model. <br>
To end goal of a bigram model is to find the conditional probability that the token t<sub>1</sub> will appear next to token t<sub>0</sub> i.e the probability P(t<sub>1</sub>|t<sub>0</sub>)