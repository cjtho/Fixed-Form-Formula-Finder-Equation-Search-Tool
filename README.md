# Fixed-Form-Formula-Finder-POC

## Introduction
This project explores the potential of artificial intelligence to provide insights and solutions for deriving formulas from integer sequences.

## Background
A 'fixed form formula' or a closed-form expression is a mathematical equation comprising a finite number of terms. These expressions are crucial for gaining mathematical insights and improving computational efficiency within their respective domains. They offer significant advantages by simplifying complex calculations and providing quick solutions to otherwise laborious problems.

Consider the sequence $(0, 1, 3, 6, 10, 15, 21, 28, ...)$. Each term here represents the sum of all integers from 0 up to the term's position. Calculating the 10,000th term manually or through straightforward computation would be impractical. However, with a closed-form formula, we can determine this value almost instantaneously for any position within the sequence.

## Data Preperation

### Generating Mathematical Expressions
To construct syntactically correct mathematical expressions, we utilize a tree-like structure that simplifies the generation process. The challenge, however, is in ensuring these expressions are semantically valid. Our solution involves a refined brute-force method where expression trees are generated under specific conditions and continually produced until a valid one is identified. This method is resource-intensive but remains efficient as long as the rate of data generation exceeds the training speed of our model, allowing us to reliably produce meaningful mathematical expressions.

### Integer Representation in Neural Networks
Handling integers of varying magnitudes in neural networks presents a unique challenge, as traditional normalization methods are often inadequate. To address this, we developed an 'integer embedding' technique. In this approach, integers are converted to a base-N representation (where N=10 for our project), with their sign encoded separately. The resulting vector is then scaled to facilitate processing. However, a challenge arises because these vectors can vary significantly in length. To standardize input without resorting to excessive padding—which would introduce a lot of redundant data—we implemented a sequence-to-sequence model within our architecture. This model treats each integer individually, transforming it into a new, dense representation akin to an NLP embedding. This innovative system allows the neural network to effectively handle any integer, learning the most efficient representations for predictive modeling.

## Model Construction


## Training

We implement a curriculum learning approach, because of the nature of the problem. If we were to give the model





