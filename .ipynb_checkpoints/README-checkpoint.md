### What is k-NN algorithm?
- Judging data with the k nearest datas
- k is usually **odd** number (if k datas in the class have the target in same ratio, there is no single answer)
### Features
- k-NN doesn't need to train model. All of the training are just saving datas.
- Calculating distance with **Euclidean Distance**
```python title="kNeighbors"
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier() # generate object

kn.fit(fish_data, fish_target) # Training the model, fish datas have 49 datas
kn.score(fish_data, fish_target) # Accuracy
kn.predict([[30,600]]) # Predicting data

print(kn._fit_X)
print(kn._y)

kn49 = KNeighborsClassifier(n_neighbors = 49) #  Refer to 49 data points
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
```

> What is the difference between k:5 and 49?
> > A larger k value doesn't necessarily mean better results.
> > K means the model judge the data with using ==k nearest datas==.
> > If k is 49, it means  all, so it'll always have the same result (numerous target). 

### Seperating train set & test set
```python title:'Making data sets'
# parameters should be tuple
fish_data = np.column_stack((fish_length, fish_weight)) 
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 4)
```
- `train_test_split` is seperating datas with train, test sets.
- `{python icon}train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify = fish_target, random_state = 4)` 
  If you use **stratify**, it seperates data with fitting class ratio
#### Caution
- Stratify must be use as the target data 

### About Scale...
- If there is a large difference in scale between the x-axis and the y-axis, unexpected results may occur. (The larger scale tends to dominate the overall influence)
  -> **Data Preprocesing** must be needed
#### Standard score
##### Standard Deviation
$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$
##### Standard Score
$$z = \frac{x - \mu}{\sigma}$$
```python title:'Data preprocessing'
mean = np.mean(train_input, axis = 0) # average
std= np.std(train_input, axis = 0) #axis = 0 : column
train_scaled = (train_input - mean) / std # Standard Score
```
#### Caution
- Data preprocessing must always be determined **based on the train_input**.
- When scaling other datas, it must be converted **based on the standard deviation, mean about train_input**.

### Memo
`{python icon} fish_data = [[l, w] for l, w in zip(length, weight)]`


# 📝 3. k-NN Regression

**Created:** 2025-07-04
**Subject:** Machine Learning
**Chapter/Topic:** 3. k-NN Regression
**Source:** 혼자 공부하는 머신러닝 & 딥러닝
**Difficulty:** 🟡 Medium
**Status:** 🟡 Learning

---

## 📚 Main Content

### Key Concepts
#### k-NN Regression 
- k-NN algorithm 
	- Predict the class of a new sample based on the majority class.
- k-NN Regression 
	- Predict the class with using k neighbors target and calculate average
#### Setting datas
```python title:'Setting datas'
import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
- Sklearn training sets must be the 2-dim array
  -> using **reshape** function 
```python title:'Reshape'
# data seperate
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)

# the input datas must be 2nd dim array in sklearn
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)
```
- if use -1, fill other dimension first and rest of the elements are fitted to proper dimension.

#### Coefficient of Determination
$$\begin{equation}
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
\end{equation}$$
$$\begin{equation}
\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i , \ \text{Average of real values}
\end{equation}$$
$$\begin{equation}
\hat{y}(\mathbf{x}) : \text{Predict value}, \ y_i:\ \text{Real value}
\end{equation}$$
```python title:'R^2'
# training model
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()

knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))
```
- Result: 0.992809406101064

#### Overfitting vs Underfitting
```python title:'Overfitting, Underfitting'
print(knr.score(train_input, train_target))
```
- Result: 0.9698823289099254
- Score
	- Train sets > Test sets
		- **Overfitting**
	- Train sets < Test sets
		- **Underfitting**
		- This means the model is too **simple**
		- ==Causes==
			- the data sets are too small
- Current model is now **Underfitting!**
#### Solution of Underfitting
- Make model complicated
	- Reduce the number of neighbors
		- the model will become more sensitive
```python title:'Solution of Underfitting'
# train model with n_neighbors = 3
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
knr.score(test_input, test_target)
```
 - Result: 0.9804899950518966 / 0.9746459963987609

### 📈 Examples & Practice
#### Understanding Overfitting & Underfitting
```python title:'Understanding overfitting and underfitting'
# test
x = np.arange(5,45).reshape(-1, 1)
for n in [1, 5 ,10]:
    knr.n_neighbors = n
    knr.fit(train_input, train_target)
    prediction = knr.predict(x)

    plt.scatter(train_input, train_target)
    plt.plot(x, prediction)
    plt.title(f'n_neighbors = {n}')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
```

![[Pasted image 20250704184453.png]]
- As n increases, the model become simple

### 🔗 Connections to Other Topics
<!-- Link to related concepts -->
- **Prerequisites:** [[2. k-NN Algorithm]] 
- **Related Topics:** [[2. k-NN Algorithm]]
- **Applications:** [[]]

---

## 🤔 Questions & Doubts
<!-- Things you don't understand or want to clarify -->
- ❓ What is the EXACT meaning of "Making model Complicate?"
- ❓ 
- ❓ 

---

## 💡 Personal Insights
<!-- Your own thoughts, patterns you noticed, etc. -->
- 
- 
- 

---

## 📋 Summary
<!-- Key takeaways in bullet points -->
- 
- 
- 

---



``` 