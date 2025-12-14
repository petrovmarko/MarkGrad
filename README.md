# 🧠 MarkGrad: A Minimal Autograd Engine

MarkGrad is a small automatic differentiation engine that was created by me from the ground up to comprehend the inner workings of backpropagation.  
It is perfect for learning and experimentation because it emphasizes accuracy and clarity over performance.

Motivated by engines in the micrograd style by @karpathy.

---

## ✨ Characteristics

    1. Scalar-based automatic differentiation  
    2. Keeps the directed acyclic graph of computations
    3. Reverse-mode autodiff backpropagation  
    4. Overloading operators (`+`, `-`, `*`, `/`, `**`)  
    5. Common nonlinearities are supported (ReLU, will add more).  
    6. Implementation that is clear and readable  
    7. Currently it supports only "sample-by-sample" prediction (no batch)

---

## Not implemented yet
Here are some things I really want to implement from scratch as well, and will do in the future if I have time.

    1. Convolutional layers
    2. Support for higher dimensional automatic training (batches)
    3. More nonlinearities

---

## Installation

Just clone the repository and you are good to go. No dependencies apart from random and cmath. You can install them with 
```
pip3 install cmath
```
```
pip3 install random
```
Then just clone the repo.
```
git clone https://github.com/petrovmarko/MarkGrad
```
```
cd MarkGrad
```

---

## Example Usage

```python
import Scalar, engine

X_train = [Scalar(x) for x in range(100)] # X 
y_train = [Scalar(x ** 2 + 3 * x) for x in range(100)] # y 

model = engine.NeuralNet(lr = 0.01) # learning rate of 0.01
epochs = 100 # epochs for training

for epoch in range(epochs):
    preds = [Model(x) for x in X_train] # get predictions
    # calculate MSE loss
    loss = Scalar(0)
    for j in range(len(preds)):
        loss = loss + (preds[j] - y_train[j]) ** 2
    loss = loss / len(y_train) # automatic cast from int to Scalar class
    
    # Clean the gradients
    model.zero_grad()
    
    # propagate backward
    loss.backward()

    # tune parameters
    model.step()

    print(f'Epoch {epoch} | loss {loss}')

```
