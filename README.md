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

### Memo
`{python icon} fish_data = [[l, w] for l, w in zip(length, weight)]`

