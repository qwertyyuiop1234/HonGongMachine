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

