## a) Introduction

We have chosen the Global Terrorism Database, Version 2 [1] by the START consortium. It contains information on more than 170,000 Terrorist Attacks.

> "The Global Terrorism Database (GTD) is an open-source database including information on terrorist attacks around the world from 1970 through 2016 (with annual updates planned for the future). The GTD includes systematic data on domestic as well as international terrorist incidents that have occurred during this time period and now includes more than 170,000 cases. The database is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism (START), headquartered at the University of Maryland." [2]

The dataset contains information about the following issues:

- ID and Date
- Incident information
- Incident location
- Attack information
- Weapon information
- Target/Victim information
- Perpetrator information
- Causalities and consequences:

More information can be found [here on Kaggle](https://www.kaggle.com/fabsta/terrorism-in-europe-and-germany/data).

## b) Data analysis

The dataset contains 135 columns in total from terrism attacks between 1970 and 2017, excluding 1993 because of a data lost. Some of the columns contain string data or categorial numbers.

For additional data, we have added an additional column which contains an assigned ID to every `gname` – the terrorist's group name to every attack. Also, we have added a column which contains a binary value if the nationality of the fatalities is the same as the country's ID to compare it later with Pearson correlation coefficient.

We have calculated the mean, median and variances of all numeric columns, as well as the probability for standard deviation with Shapiro-Wilk test. As expected, none of the columns is deviated like a standard deviation. The exact values are listed in appendix. We than have plotted the values as histograms and scatter plots.

As third step, we have then plotted the variables in pairs as heatmaps to get an idea for potential correlations. It is apparent that the most attacks happened in the last 10 years, according to figure 1 below. Also, the nationality of fatalities is most of the time the country of the attack, which is visible as a clear coloured line in figure 2.

![Figure 1: Plot of ID of events and year to ](https://i.imgur.com/yIn2Tqc.png)


![Figure 2: Plot of country versus nationality of targets/victims](https://i.imgur.com/WUixBuY.png)

### Interesting features and subgroups

#### Subgroup 1: Terrorism attacks and country

The following figure 3 shows the distribution of terrorism attacks over countries. The countries are decoded as id. As shown in the plot below, there are three extremes marked. Interval of country ids 90-100 has 27.4k attacks, 140-150 19.2k attacks and 0-10 15.4k attacks.

![Figure 3: Histogram of country IDs](https://i.imgur.com/GYL0KDS.png)

The figure 4 below visualizes the location of the terrorism attacks globally given by its longitude and latitude. A noteworthy finding is the high density of attacks in the area of middle-west-asian, middle asian and at the west-coast of middle america.
![Figure 4: Geographical heatmap of terrorism attacks](https://i.imgur.com/4HRcmyh.png)

#### Relation of the two plots:

At the histogram it can be seen that the most attacks occurs globally in the country id from 90 to 100. (Please take note, the interval correlates with the chosen bin size.) countries that can be looked up with the mentioned IDs are *Hungary, Iceland, India, Indonesia, Iran, Iraq, Ireland, Israel, Italy, Ivory Coast*. Since the IDs are given to countries according to the alphabetical order of the country name, countries can also considered into the local density while there are no attacks.

### Possible Correlations

We have calculated the Pearson correlation coefficient for each numeric column. For this we have replaced every incidents of `NaN` with the column's average mean. With this, we have calculated then a `n x n` matrix, where `n`is the number of columns. Columns which contain strings were left out (and have a coefficiency of 0 in our matrix). We then have plotted the matrix with Matplotlib and explored the peaks of the coefficent matrix (see figure 5 for illustration).

![Figure 5: interactive selection of coefficients](https://i.imgur.com/iP83XdE.png)

Therefore, we have chosen the following positive and negative coefficients as interesting and worth to explore in more detail, whose *coef = < -0.3* or *coef = > 0.3* and filtered out semantic meaningless correlations:

| Column 1 | Column 2 | Coefficient | Explanation
| - | - | - | - |
|iyear|longitude|0.54| The year intends slightly, where the terrorist attack might occur |
|extended|ishostkid|0.36| If the hostage were kidnapped for longer than 24 hours. |
|attacktype1|weaptype1|0.65| There is a tendency, where the type of attack leads to the chosen weapon type. |
|nkill|nkillus|0.44| The potential that the fatalities were US citizens. |
|nkill|nkillter|0.35| The potential that the fatalities were terrorists |
|nkillus|nwound|0.75| The chance of having a US fatality in the number of injuries. |
|natlty1|country|0.59| The nationality of the target/victim might be also the country of attack.|
|natlty_differ|country|0.38| The potential that the nationality of target/victim differs from the country of attack. |
|ndays|nreleased|0.33|The potential that on increasing days of hostage / kidnapping results in a higher number of released targets/victims.|

For example we had a **strong correlation coefficient of 0.982** for the columns *targtype* and *targsubtype*, but this contains no semantic value, since every target type is defined also by a set of target subtypes.

## d) Clustering

Based on the correlations, it is not feasable to identify useful correlation to classify. Beside trivial correlations (for instance **geo-location of the the attack and distribution of affected countries** or **weapontype and attacktype**) we did not find any other correlations that is interesting enough to classify.

## e) Reduction of dimensions

We have seen that some columns were not used or are incomplete. Therefore, we have reduced and augmented undefined cells with the following method:

- Delete **attacktype2\*, attacktype3\*** (broken data).
- Replace NaN's with mean of the list to prevent data loss in **longitude** and **latitude**.
- Replace NaN to 0.0 in **alternatives.json**, and set NaN to "Undefined" in **alternatives_txt.json**.
  - Key: 0.0 is coded as "Undefined"
- Delete **claims2.json** and **claims3.json** (broken data)
- Delete **claimmode2\*.json** (broken data)
- Delete **claimmode3\*.json** (broken data)
- Delete **gname2.json**, **gname3.json** (broken data)
- Replace NaN with "Undefined" in **gsubname.json**.
- Replace Nan with "Undefined" in **addnotes.json**.
- Replace Nan with "Undefined" in **approxdate.json**.
- Replace Nan with -1 in **claimed.json**.
- Replace Nan with -1 in **claimmode.json**.
- Replace Nan with -1 in **claimmode_txt.json**.
- Replace Nan with "Undefined" in **corp1,- 2,- 3.json**
- Replace Nan with -1 in **divert.json**.


## g) Conclusion

The dataset we have chosen on kaggle provides a huge amount of information we need to pre-process before. This includes the following steps:

- deleting datasets that consists non-processable entries e.g. *NaN*
- or 'rescueing' datasets where small batches contains an invalid type.
  - Those we have replaced with its mean (by numerical datasets)
  - or "Undefined" by descriptions.
  - In "categorial" datasets with have etablished our own encoding for instance -1 if the entry is invalid.

We analysed the data by plotting the histograms and scatter plots via matplotlib and computed the correlation for multiple variables using *Pearson correlation coefficient*.

We found that the dataset contains a huge set of attacks. Based on the correlations, the location of the attack is an indicator for a feature variable, thus relevant data for the prediction for future common terrorism attacks. Logically, based on historical changes, the year might be an additional feature for the prediction of future attacks.

Alternatively, it is possible to use the data of hostage / kidnapping attacks, such as the number of days, the number of rescued, the type of rescue, as well as the country to predict, how likely it is to rescue all targets / victims.

## h) References

- **[1]** START CONSORTIUM: Global Terrorism Database, Version 2,  https://www.kaggle.com/START-UMD/gtd (last visit on 29th of June, 2018, 20:02)
- **[2]** DOCUMENTATION FOR GTD DATABASE https://www.start.umd.edu/gtd/downloads/Codebook.pdf

## Appendix A: Table of columns

| id | columname | mean | median | variance | W | normality |
|-|-|-|-|-|-|-|
| 0 | eventid | 200177632735.86 | 200712130009.50 | 1727753934559308032.00 | 0.82 | 0.00 |
| 1 | iyear | 2001.71 | 2007.00 | 172.77 | 0.85 | 0.00 |
| 5 | extended | 0.04 | 0.00 | 0.04 | 0.20 | 0.00 |
| 7 | country | 132.53 | 98.00 | 12734.63 | 0.71 | 0.00 |
| 9 | region | 7.09 | 6.00 | 8.70 | 0.90 | 0.00 |
| 13 | latitude | 23.40 | 31.13 | 345.53 | 0.92 | 0.00 |
| 14 | longitude | 26.35 | 42.24 | 3337.68 | 0.91 | 0.00 |
| 20 | crit2 | 0.99 | 1.00 | 0.01 | 0.06 | 0.00 |
| 21 | crit3 | 0.88 | 1.00 | 0.11 | 0.38 | 0.00 |
| 23 | alternative | 1.29 | 1.29 | 0.07 | 0.31 | 0.00 |
| 28 | attacktype1 | 3.22 | 3.00 | 3.58 | 0.76 | 0.00 |
| 34 | targtype1 | 8.40 | 4.00 | 44.20 | 0.85 | 0.00 |
| 36 | targsubtype1 | 46.87 | 39.00 | 904.80 | 0.93 | 0.00 |
| 40 | natlty1 | 127.73 | 107.00 | 7736.05 | 0.79 | 0.00 |
| 42 | targtype2 | 10.20 | 10.20 | 1.94 | 0.29 | 0.00 |
| 44 | targsubtype2 | 28.37 | 28.37 | 41.16 | 0.24 | 0.00 |
| 50 | targtype3 | 9.88 | 9.88 | 0.20 | 0.05 | 0.00 |
| 52 | targsubtype3 | 55.07 | 55.07 | 4.01 | 0.04 | 0.00 |
| 71 | claimed | 0.03 | 0.00 | 0.80 | 0.19 | 0.00 |
| 75 | claimmode2 | 7.22 | 7.22 | 0.02 | 0.02 | 0.00 |
| 78 | claimmode3 | 7.09 | 7.09 | 0.01 | 0.01 | 0.00 |
| 80 | compclaim | -6.42 | -6.42 | 0.48 | 0.15 | 0.00 |
| 81 | weaptype1 | 6.43 | 6.00 | 4.63 | 0.60 | 0.00 |
| 85 | weaptype2 | 6.74 | 6.74 | 0.34 | 0.26 | 0.00 |
| 87 | weapsubtype2 | 10.67 | 10.67 | 3.49 | 0.29 | 0.00 |
| 89 | weaptype3 | 6.87 | 6.87 | 0.05 | 0.07 | 0.00 |
| 91 | weapsubtype3 | 11.51 | 11.51 | 0.62 | 0.07 | 0.00 |
| 93 | weaptype4 | 6.24 | 6.24 | 0.00 | 0.00 | 0.00 |
| 95 | weapsubtype4 | 10.79 | 10.79 | 0.03 | 0.00 | 0.00 |
| 98 | nkill | 2.39 | 1.00 | 121.02 | 0.15 | 0.00 |
| 99 | nkillus | 0.05 | 0.00 | 22.15 | 0.00 | 0.00 |
| 100 | nkillter | 0.48 | 0.00 | 10.69 | 0.08 | 0.00 |
| 101 | nwound | 3.20 | 0.00 | 1092.44 | 0.04 | 0.00 |
| 103 | nwoundte | 0.10 | 0.00 | 1.25 | 0.04 | 0.00 |
| 109 | ishostkid | 0.06 | 0.00 | 0.20 | 0.17 | 0.00 |
| 113 | ndays | -31.89 | -31.89 | 613.75 | 0.10 | 0.00 |
| 122 | hostkidoutcome | 4.62 | 4.62 | 0.24 | 0.27 | 0.00 |
| 124 | nreleased | -28.72 | -28.72 | 188.78 | 0.21 | 0.00 |
| 130 | INT_LOG | -4.58 | -9.00 | 20.64 | 0.65 | 0.00 |
| 131 | INT_IDEO | -4.51 | -9.00 | 21.44 | 0.67 | 0.00 |
| 132 | INT_MISC | 0.09 | 0.00 | 0.34 | 0.22 | 0.00 |
| 133 | INT_ANY | -3.98 | 0.00 | 22.01 | 0.68 | 0.00 |
| 135 | natlty_differ | 0.13 | 0.00 | 0.11 | 0.39 | 0.00 |
| 136 | gname_ids | 2477.41 | 3050.00 | 1070155.58 | 0.79 | 0.00 |# Machine Learning - Assignment 11

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
