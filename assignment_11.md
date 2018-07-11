# Machine Learning - Assignment 11

## Introduction
> Describe the prediction tasks and the evaluation metric used in the Kaggle competition. Use this evaluation metric in your study (this might be an unusual metric which you might need to implement yourself).

For this assignment, we have used the Global Terrorism Database by the START consortium. We have analysed the dataset systematically in the previous assignment and have found interesting correlations according to hostage / kidnapping data. Not only the kidnapping data has significantly increased with $r = 0.035$ with $p = 0.0078 < 0.05$ over the last decade, the outcome of a kidnapping seems to be predictible, as shown in figure 1.



![Figure 1: Increase of kidnappings over years](https://i.imgur.com/xDO2Z9d.png)



This yet alarming observation of increasing kidnappings and hostiges let us ask if it is possible to predict the outcome of a hostige or kidnapping, as well as the possible percentage of survivers of this attack to be able to get a better understanding of such. It might be also possible to find ways to release more hostiges and increase the number of survivers in future.

We have started to re-investigate the data in more detail, to make sure, that we understand the connections between single variables in detail, which we will describe in more detail in section 2.


## Data Preparation and Selection
> Describe your data preparation steps including normalization.
> > Describe clearly which data you use for training, validation, and testing, and how you optimize parameters.

For this assignment, we have reduced the used dataset columns to the following columns as feature input which might correlate with the kidnapping data in total:

- iyear
- extended
- nhostkid
- nhours
- ndays
- ransom
- ransompaid
- ransompaidus
- nreleased
- gname_id
- ndied

And we have used the following columns as classification labels:
- hostageoutcome

First, we have filtered the rows by the type of attack, which has to be category 5 or 6 which stands for hostages or kidnapping alone or in combination or had the value $1$ (for *yes*) in *ishostkid*. We then removed the rows which have contained broken data (NaNs) in our prediction vectors *hostageoutcome* and *nreleased*. Then, we also have removed all rows which have the value *-99* in *nreleased*. Finally, we added another filter step where we kept out every row whose number of hostages (*nhostkid*) is smaller than the number of released hostages (*nreleased*).

The second step was to enrich the dataset with additional variables. We have assigned an index to every single terrorist group name, as well as normalised the number of released hostages by dividing it by *nhostkid*. The figure 2 shows clearly, that there are many correlations in our chosen feature set.

![Figure 2: Correlations of chosen columns](https://i.imgur.com/RR4HPsm.png)

Semantic unknown reasons or numbers in the data were assigned as $-9$ or $-99$ in values by the dataset vendor. Also, there were many $NaN$ values where there were no data available. Because both cases means literally the same, we have assigned the value $-1$ to all incidents of $-9$, $-99$ or $NaN$.

The last preparation step was to separate the dataset to training, validation and test set. We decided to use 60 % of the refined dataset as training, 20 % as validation and remaining 20 % as test set.

The size of our dataset were therefore:

| Set | Number of features (X) | Number of labels (y) | Number of data|
|-|-|-|-|
|Training|11|2|3623|
|Validation|11|2|1208|
|Test|11|2|1208|

## Prediction Architectures
> Investigate at least three different prediction methods. For instance, for classification you might use SVMs, logistic regression, kNN, decision trees, etc. Among them must be at least one method which was presented in class. Also investigate and document different parametrizations (kernel type, distance measure, etc.)

### Fully Connected Neural Network

We have trained a model that should predict the hostkidoutcome and nreleased_p by given informations as shown below. X is the input vector containing informations of the columns that is written in X.


X = ['iyear', 'extended', 'nhostkid', 'nhours', 'ndays', 'ransom', 'ransompaid', 'ransompaidus', 'nreleased', 'gname_id', 'ndied']

y = ['hostkidoutcome', 'nreleased_p']

#### Model Architecture
```python
model = Sequential()
model.add( Dense(input_dim, input_dim=input_dim, kernel_initializer="uniform", activation='tanh', name='layer_0'))
model.add(Dense(128, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_1'))
model.add(Dense(256, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_2'))
model.add(Dense(256, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_3'))
model.add(Dense(128, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_4'))
model.add(Dense(output_dim, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_5'))

opt = Optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06, decay=0.0)
model.compile(optimizer=opt, loss='mse')
```
#### Dataset
We have splitted our dataset into training,- (60%), validation,- (20%) and test-set(20%).

We also normalized each column of the data-set to the interval [0,1] in order to interpret the training results correctly.

#### Training
![](https://i.imgur.com/MSsVci3.png)


#### Results
We evaluated the model with the test-set and achieved a score of ```0.012780360795406391```


## Results
> Illustrate your results with adequate plots.

> Discuss your findings: which model would you recommend to use? Are the overall results satisfactory? Could your predictive model be used in practive? Use a lot of skepticism and point to limitations, potential problems, and shortcomings in your approach. Can you hypthesize any causal effects? Or do important explanatory variables seem to be missing?

> If possible, compare your restults to the results of other participants in the Kaggle competition

We cannot compare our results to other submitted results, because we have trained our model on a unique dataset composition dataset and a unique goal as well.

## Conclusion
> Provide short concluding remarks about your findings. Which types of methods achieved the best results? Where would you continue future work?
> 

In our project we have firstly identied the most correlated columns of the dataset. We use these to train our model and we have achieved a great results : ```0.012780360795406391```

We would continue to change the composition of the dataset (input vector) to investigate if it affects positively/negatively on the predicted result.
