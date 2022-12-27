import numpy as np
import pandas as pd

def convert_data(df) -> list:
    pass




if __name__ == '__main__':
    #importation de données
    df = pd.read_csv("sample_submission.csv")
    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")

    #conversion de données

    #conversion en liste
    l = np.array(df)
    l_test = np.array(df_test)
    l_train = np.array(df_test)

    #print(l)
    #print(l_test)
    t = list(l_train[::1])
    print("liste ",t)

