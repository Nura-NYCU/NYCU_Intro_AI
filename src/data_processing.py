import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/covid_fake_news.csv')

def load_data():
    global df
    print('Original Data')
    print(df.head(), end = '\n\n')

    print('Pre-Processed data')
    df['headlines'] = df['headlines'].str.lower()
    print(df.head(), end = '\n\n')
    
    print('Train and Test splits loaded...')
    X = df['headlines']
    y = df['outcome']
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    return x_train, x_test, y_train, y_test
