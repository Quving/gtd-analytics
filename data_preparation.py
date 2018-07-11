import numpy as np
import pandas as pd

# Read the data

def read_data(filename, usecols=None):

    df = pd.read_csv(filename, encoding = "ISO-8859-1", usecols=usecols)
    print('Finished reading csv file. Dimensions: {}'.format(df.shape))

    return df

def split_categories(df, categories):

    for category in categories:
        df = df[category].get_dummies()

def str_to_index_arr(arr):

    '''
    Takes an unordered list of strings, orders them, assigns an unique ID to every string and builds an equally long list of indexes respectively to the strings.
    '''
    strings = np.sort(np.unique(arr))
    ids = range(0, len(strings))

    id_dict = {key: index for key, index in zip(strings, ids)}
    id_arr = [id_dict[string] for string in arr]

    return id_arr, id_dict

def make_bool_arr(arr, conditions):

    return np.array(arr) == conditions

def separate_labels(df, labels):
    '''
    Separates the labels columns from the dataframe and returns the dataframe and the labels

    Returns:
    - The dataframe of X variables
    - The dataframe of labels (Y)
    - An array of each label column separated as array, ordered in the order of labels in labels array
    '''

    return df.drop(labels, axis = 1), df[labels], [df[label] for label in labels]

