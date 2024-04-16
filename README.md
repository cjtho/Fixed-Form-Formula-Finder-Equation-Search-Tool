# Fixed-Form-Formula-Finder-POC

## Introduction
What is the formula for the sequence $(1, 3, 5, 7, 9, 11, 13)$? The odds; $2n+1$, easy right? How about $(0, 1, 3, 6, 10, 15, 21)$? A bit trickier, but many will recognize these as the triangular numbers, represented by the formula $\frac{n(n+1)}{2}$. Now, consider the sequence $(0, 2, 4, 8, 13, 20, 29)$. Stumped, I bet. Yet, there’s likely a small voice telling you there’s something familiar about it. Imagine an AI that could analyze these integer sequences and precisely identify the functions describing them. This project explores the potential of artificial intelligence to provide insights and solutions for deriving formulas from integer sequences. We aim to answer the intriguing question: Can AI help us determine which functions are **probably** present in a given sequence?

P.S. The answer was $\text{Fib}(n) + \text{Tri}(n)$, combining Fibonacci and triangular numbers. :)


## Background
A 'fixed form formula' or a [closed-form expression](https://en.wikipedia.org/wiki/Closed-form_expression) is a mathematical equation comprising a finite number of terms. These expressions are crucial for gaining mathematical insights and improving computational efficiency within their respective domains. They offer significant advantages by simplifying complex calculations and providing quick solutions to otherwise laborious problems.

Consider the sequence $(0, 1, 3, 6, 10, 15, 21, 28, ...)$. Each term here represents the sum of all integers from 0 up to the term's position. Calculating the 10,000th term manually or through straightforward computation would be impractical. However, with a closed-form formula $\frac{n(n+1)}{2}$, we can determine this value almost instantaneously for any position within the sequence.

## Data Preparation

### Generating Mathematical Expressions
To construct syntactically correct and semantically valid mathematical expressions, we utilized a tree-like structure. The generation process faces the challenge of ensuring these trees form valid mathematical expressions. We employed a refined brute-force method where trees are generated under stringent conditions and tested for validity. This method, while resource-intensive, efficiently keeps pace with our model's training speed, ensuring a steady supply of meaningful data.

### Integer Representation in Neural Networks
Handling integers of varying magnitudes presents a unique challenge in neural networks, as traditional normalization methods fall short. Our 'integer embedding' technique converts integers into a base-N representation (N=10 for this project), encoding their sign separately. This approach helps standardize input lengths without excessive padding, reducing redundant data. Our model treats each integer vector as a unique entity, transforming it into a dense representation that enhances the neural network's ability to learn and predict effectively.

## Model Construction
The design of the model was constrained by the computing power of my laptop. However, there were several key aspects to address:

### Integer Embeddings
To enhance the representation of each base-N integer vector, we applied a multi-layer bidirectional LSTM followed by a dense layer with batch normalization. This approach aims to condense the information into a richer representation that the model can utilize more effectively.

### Sequence Encoding
Once each integer is transformed into a rich vector, we obtain a sequence of vectors. We opted to use another multi-layer bidirectional LSTM network as an encoder to capture the entire sequence's information comprehensively.

### Decoding
Given that the task was merely to predict the presence of tokens, we employed a straightforward dense layer with batch normalization as a feed-forward network for decoding the encoded structure.

### Explanation
The rationale behind this unusual architecture was influenced by the original goals and constraints. Initially, we intended to employ the cutting-edge technology of transformers for an encoding-decoding architecture. This model would not only predict the presence and count of tokens but also their order in a prefix traversal, effectively generating the mathematical expression. However, due to computational limitations, we had to modify the model's design and objectives.

## Training
We adopted a curriculum learning approach due to the complex nature of the problem. Introducing the model to any mathematical expression from the outset could be either too challenging or slow the learning process. Therefore, we divided the complexity into two aspects: the number of terms in a mathematical expression and the complexity of the underlying functions. The idea was that the model should first master simpler tasks like addition before tackling more complex functions, such as Fibonacci sequences or binomial coefficients.

In practice, our training regimen began with a loop where the model was exposed to expressions across all complexity levels. This initial broad exposure was intended to help the model generalize its learning. Following this, the model was repeatedly trained on simpler tasks tailored to its current level in the curriculum until its performance plateaued.

## Results
Here are the results of our training:
![image](https://github.com/cjtho/Fixed-Form-Formula-Finder-POC/assets/151635991/80e94f66-3bcb-4fb9-a352-668664f1e095)

### Formula Predictions Analysis for Small Data Entries
**Correct Predictions:**
- **Triangular Numbers**: Identifies multiplication
- **Odd Numbers**: Identifies addition
- **Square Numbers**: Identifies multiplication

**Failed Predictions:**
- **Triangular Numbers**: 
  - **Missed**: Slight addition
  - **Missed**: Floor division
- **Odd Numbers**:
  - **Missed**: Multiplication is necessary for the computation

### Formula Predictions Analysis for Larger Data Entries
With a larger dataset, the prediction for **Triangular Numbers** improves:
- **Now Predicts**: Multiplication, addition, and floor division
- **Incorrectly Predicts**: Subtraction, which does not apply

### Observations on Odd Numbers Formula
The model's strong prediction of addition for **Odd Numbers** might be influenced by an assumption of$n+n$rather than the actual$2 \times n$. This suggests the model could be misinterpreting the multiplication of two as repetitive addition.

## Future Considerations
### Limitations of Neural Networks in Exact Problem Domains
This project, a proof of concept, highlights a fundamental issue with using neural networks in certain problem domains—those requiring exactitude. Neural networks, while excellent general approximators, are disadvantaged when precise outcomes are necessary. For example, when a neural network processes the number 2, it does not perceive it as an exact integer but rather as an approximation akin to a 'feeling' of 2. This inherent characteristic places the architecture at a disadvantage for tasks that demand exact solutions, unlike in domains where there is a direct correspondence between elements, such as in language translation.

### Challenges in Data Preparation
The data preparation phase experienced significant challenges that likely influenced the model's performance. The method used for generating random expression trees may introduce bias, preventing the model from observing a true distribution of possible relationships. Furthermore, these trees often generate expressions that need simplification, complicating the learning process. For instance, if the model correctly predicts $N+1$ but the actual label is $N−(−1)$, it is unfairly penalized. Developing a method for simplifying expressions during the data preparation phase is essential to reduce redundancy and bias in training data.

### Proposed Enhancements in Model Architecture and Loss Function
To address the issue of exactitude versus approximation, I suggest that neural networks should not serve as the definitive solution for this project. Instead, they could be optimized to provide probabilistic insights into the existence, counts, and placements of tokens, leveraging their strengths in pattern recognition. This approach requires advances in computing power, architecture (such as transformers), and training data quality. Additionally, adapting the loss function to more appropriately reward or penalize the model could significantly enhance performance, taking into account the subtleties of mathematical expression recognition.

### Integration of Deliberate Modeling Techniques
A more deliberate modeling approach may offer a superior solution for the complex problem space of this project. Framing the task as a type of game where an agent interacts with an environment of integers could allow for more strategic and nuanced problem-solving. Using a reinforcement learning agent to explore and construct expression trees provides the advantage of making deliberate actions based on exploration and feedback. Unlike neural networks, such agents have the capability to 'go back' and revise their strategies, potentially leading to more accurate and robust solutions. Lastly, it can utilize the predictions of the aforementioned neural network model so to give it a 'kick start' in the right direction.
