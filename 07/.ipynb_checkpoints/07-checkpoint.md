# 1.Fashion MNIST
---
- We're gonna use fashion MNIST datasets in 'keras' library for training model.
```python title:'Fashion MNIST'
import keras
(train_input, train_target), (test_input, test_target) =\
keras.datasets.fashion_mnist.load_data()
```
- `load_data()`: split train, test datasets.

```python title:'Check shape'
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)
```
![[1.Artificial Neural Network (ANN).png|500]]


```python title:'Show images'
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize = (10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
```

```python title:'Check target'
print(train_target[:10])

# return: [9 0 0 3 0 2 7 2 5 5]

import numpy as np
print(np.unique(train_target, return_counts = True))

# return: 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))
```
- Each number represents a specific class.
- Each classes have 6000 samples.

# 2.Classifying fashion items using logistic regression
---
- Use Stochastic Gradient Descent(SGD)
- For using SGDClassifier, datas should be preprocessed(Standardization).
- SGDClassifier handles only 1-dim array.
```python title:'Normalization data'
train_scaled = train_input / 255.0 # normalization
train_scaled = train_scaled.reshape(-1, 28 * 28) # reshape
print(train_scaled.shape) # return: (60000, 784)
```

```python title:'Evaluating the performance using cross_validate'
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss = 'log_loss', max_iter =5, random_state = 42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs = -1)
print(np.mean(scores['test_score']))
```

# 3.Artificial Neural Network(ANN)
---
![[1.Artificial Neural Network (ANN)-1.png]]

# Tensorflow & Keras
---
- Tensorflow: a deep learning library that google made. ==(keras' backend)==
- Keras: tensorflow's API
![[1.Artificial Neural Network (ANN)-2.png|400]]
# 4.Making ANN Model
---
```python title:'Take out datas from train datas'
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)
```

### Input Layer
```python title:'Input layer'
inputs = keras.layers.Input(shape = (784,))
```
![[1.Artificial Neural Network (ANN)-3.png]]
> [!important] 
> input() is a python function that takes input from clients.
> => USE INPUT<font color="#ff0000">S</font>

### Dense Layer
```python title:'Dense layer'
dense = keras.layers.Dense(10, activation = 'softmax')
```
![[1.Artificial Neural Network (ANN)-4.png]]

### Model
```python title:'Model'
model = keras.Sequential([inputs, dense])
```
![[1.Artificial Neural Network (ANN)-5.png]]
> [!caution] 
> Don't forget the fact that <span style="background:#fff88f">each intercepts are added in each neurons</span>

# 5.Compile
---
- Before fitting model, we need to set model.
  $\Rightarrow$ loss_function **MUST BE** defined.
```python title:Compile
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```

### Loss_function
![[1.Artificial Neural Network (ANN)-6.png]]
- You have to change target array to one-hot encoding, but in kerase, you can use integer target array **as it is**.
  $\Rightarrow$ <span style="background:#affad1">sparse_categorical_crossentropy</span>

### Metrics
- To print accuracy in each epochs, use `{python icon}matrix = ['accuracy']` 

# 6.Training
---
```python title:'Fit'
model.fit(train_scaled, train_target, epochs = 5)
```
<center><img src="1.Artificial Neural Network (ANN)-7.png"/></center>

```python title:Evaluate
model.evaluate(val_scaled, val_target)
```
<center><img src="1.Artificial Neural Network (ANN)-8.png"/></center>
