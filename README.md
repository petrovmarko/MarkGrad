# MarkGrad: A Minimal Autograd Engine

MarkGrad is a small automatic differentiation engine that was created by me from the ground up to comprehend the inner workings of backpropagation.  
It is perfect for learning and experimentation because it emphasizes accuracy and clarity over performance. Right now it can feel a bit slow especially when training multiple layers, and I plan on 
implementing a background c++ engine that will make the computation faster.

Still not done in full.

---

## Characteristics

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

Just clone the repository and you are good to go. No dependencies apart from random and math. You can install them with 
```
pip3 install math
```
```
pip3 install numpy
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
from scalar import Scalar
from engine import Layer, NeuralNet

y_train = [(x**2) + 1 for x in range(100)] # y 

model = NeuralNet(lr = 0.001)
model.add_layer(Layer(in_feature=1, out_features=10, scale = 1, activation = 'relu'))
model.add_layer(Layer(in_feature=10, out_features=20, scale = 0.1, activation = 'relu'))
model.add_layer(Layer(in_feature=20, out_features=1, scale = 0.01, activation='None'))

epochs = 300 # epochs for training

for epoch in range(epochs):
    X_train = list(range(100))
    preds = [model(x) for x in X_train] # get predictions
    # calculate rMSE loss
    loss = Scalar(0)
    for j in range(len(preds)):
        loss = loss + (preds[j] - y_train[j]) ** 2
    loss = loss / len(y_train) # automatic cast from int to Scalar class
    loss = loss ** 0.5
    # Clean the gradients
    model.zero_grad()
    
    # propagate backward
    loss.backward()

    # tune parameters
    model.step()
    print(f'Epoch {epoch} | loss {loss}')

print([model(x) for x in range(100)])
    

    

```
