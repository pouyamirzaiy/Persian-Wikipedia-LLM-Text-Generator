import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    df['alcohol'] = np.log10(df['alcohol'])
    bins = np.arange(0.0, 2.0, 0.05)
    df['alcohol'] = pd.cut(df['alcohol'], bins=bins)
    county_data = pd.get_dummies(df['alcohol'], prefix='alcohol').replace({False: 0, True: 1})
    df = df.join(county_data)
    df.drop('alcohol', axis=1, inplace=True)

    df['volatile acidity'] = np.log(df['volatile acidity'])
    bins = np.arange(-10, 2, -2)
    df['volatile acidity'] = pd.cut(df['volatile acidity'], bins=bins)
    county_data = pd.get_dummies(df['volatile acidity'], prefix='volatile acidity').replace({False: 0, True: 1})
    df = df.join(county_data)
    df.drop('volatile acidity', axis=1, inplace=True)

    bins = [x for x in range(0, 300, 50)]
    df['total sulfur dioxide'] = pd.cut(df['total sulfur dioxide'], bins=bins)
    county_data = pd.get_dummies(df['total sulfur dioxide'], prefix='total sulfur dioxide').replace({False: 0, True: 1})
    df = df.join(county_data)
    df.drop('total sulfur dioxide', axis=1, inplace=True)

    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=['Id', target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
