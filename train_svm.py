#!/usr/bin/env python

import numpy as np
import pandas as pd

# Plotter library
import seaborn as sns
import matplotlib.pyplot as plt

# Own libraries
import data_preparation as prep


def filter_data(df):
    # Filter for only kidnapping data (1st, 2nd or 3rd attack type)
    kidnap_cats = [5, 6]
    df = df[df.attacktype1.isin(kidnap_cats) | df.attacktype2.isin(kidnap_cats) | df.attacktype3.isin(
        kidnap_cats) | df.ishostkid == 1]

    # Drop attacktype columns. They aren't needed anymore
    df = df.drop(['attacktype1', 'attacktype2', 'attacktype3', 'ishostkid'], axis=1)

    # Filter also broken data from our classes
    df = df[df.hostkidoutcome.notnull()]

    # Filter data for NaN nreleased or value -99
    df = df[df.nreleased.notnull()]
    df = df[df.nreleased != -99]

    # Filter also data where nhostkid is lower than nreleased
    df = df[df.nhostkid >= df.nreleased]

    return df


def augmentate_data(df):
    # Add an ID group for gname to the DataFrame
    df['gname_id'], _ = prep.str_to_index_arr(df['gname'])

    # Add a normalisation for how many of the hostage victims survived
    df['nreleased_p'] = np.divide(df.nreleased, df.nhostkid)

    # Add a column of died hostages
    df['ndied'] = np.subtract(df.nhostkid, df.nreleased)

    # Drop all string columns and keep only numeric ones.
    df = df._get_numeric_data()

    return df


def handle_NaN_in_data(df):
    from sklearn.preprocessing import Imputer
    fill_NaN = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index

    df = imputed_df

    return df


def set_NaN_to_value(df, value):
    return df.replace(np.nan, value)


def set_unknown_to_NaN(df, unknown_values):
    for unknown_value in unknown_values:
        df.replace(unknown_value, np.nan)

    return df


def visualize_data(df, path='', suffix=''):
    # First: a plot about number of kidnapped persons
    sns.set(style="darkgrid", color_codes=True)

    g1 = sns.jointplot(
        'iyear',
        'nhostkid',
        data=df,
        kind="reg",
        color='r',
        size=7,
        xlim=[1970, 2016]
    )
    g1.set_axis_labels('Years', 'Number of kidnapped victims')

    g1.savefig(path + 'interaction-iyear_nhostkid' + suffix + '.png')
    g1.savefig(path + 'interaction-iyear_nhostkid' + suffix + '.pdf')
    # Outcomes vs percentage of released victims

    g2 = sns.violinplot(
        x='hostkidoutcome',
        y='nreleased_p',
        data=df,
        hue='ransom'
    )

    g2.figure.savefig(path + 'interaction-hostkidoutcome_nreleased_p' + suffix + '.png')
    g2.figure.savefig(path + 'interaction-hostkidoutcome_nreleased_p' + suffix + '.pdf')

    ### Correlation

    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g3 = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=.3,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    g3.figure.savefig(path + 'correlation_full' + suffix + '.png')
    g3.figure.savefig(path + 'correlation_full' + suffix + '.pdf')


def train(X, Y):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)


def train_svm(X, y, C=1.0):
    '''
    Trains the SVM with X as input and y as output data

    Input:
    - X: Input vector with features
    - y: Output vector with one label column
    - C: SVM regularisation parameter
    '''

    from sklearn.svm import SVC

    svm_model = SVC(kernel='linear', C=C, decision_function_shape='ovr')
    svm_model.fit(X, y)

    return svm_model


def predict_svm(model, X, y):
    Z = model.predict(X)

    return Z


if __name__ == "__main__":
    #####
    # The purpose of our classifier is to predict the hostkidoutcome category and a percentage of released persons.
    # Y: hostkidoutcome, npreleased
    # X: extended, iyear, gname_id, nhostkid, ndays, ransom, ransompaid, ishostkid
    #####

    ### Data filtering

    # Read data and exclude cols
    # @Snippet: To exclude: lambda x: x not in ["eventid","imonth","iday", "attacktype2","claims2","claimmode2","claimmode3","gname2"]
    df = prep.read_data('globalterrorismdb_0617dist.csv',
                        usecols=['nreleased', 'attacktype1', 'attacktype2', 'attacktype3', 'extended', 'iyear', 'gname',
                                 'nhostkid', 'nhours', 'ndays', 'ransom', 'ransompaid', 'ransompaidus', 'ishostkid',
                                 'hostkidoutcome'])
    df = filter_data(df)
    df = augmentate_data(df)

    # We also have sometimes -99 or -9 as values when things were unknown. We have to replace them as well with NaNs
    df = set_unknown_to_NaN(df, [-9, -99])

    # We have a whole number of columns which contains NaNs for missing data. To overcome those, we simply use the sklearn Imputer to fill the NaNs with the mean values
    df = set_NaN_to_value(df, -1)

    print(df.head())

    # Plot data
    visualize_data(df, path="plots/")
    print('Resulting columns for training: \n{}\n'.format(df.columns))

    ### Separate set into train, validation, test by assigning each to the preferred class randomly.
    train = df.sample(frac=0.6, replace=True)
    validation = df.sample(frac=0.2, replace=True)
    test = df.sample(frac=0.2, replace=True)

    labels = ['hostkidoutcome', 'nreleased_p']
    X_train, Y_train, Y_train_columns = prep.separate_labels(train, labels)
    X_validation, Y_validation, Y_validation_columns = prep.separate_labels(validation, labels)
    X_test, Y_test, Y_test_columns = prep.separate_labels(test, labels)

    print('''
Table of set sizes:

| Set | Number of features (X) | Number of labels (y) | Number of data|
|-|-|-|-|
|Training|{}|{}|{}|
|Validation|{}|{}|{}|
|Test|{}|{}|{}|
    '''.format(
        len(X_train.columns),
        len(Y_train.columns),
        X_train.shape[0],
        len(X_validation.columns),
        len(Y_validation.columns),
        X_validation.shape[0],
        len(X_test.columns),
        len(Y_test.columns),
        X_test.shape[0]
    ))

    ########
    # Training

    # svm_model = train_svm(X_train, Y_train_columns[0])
    # prediction = predict_svm(svm_model,X_test)
