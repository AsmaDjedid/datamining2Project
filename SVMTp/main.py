import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #L'import afin de dissiner des graphes
#asssss
class SVM:
    
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, nbr_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = nbr_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, x):
        n_features = x.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)

    
    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b
        return self.cls_map[idx] * linear_model >= 1


    def _get_gradients(self, constrain, x, idx):
            
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db


    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Déclaration de fonctions
    #create a list of column names with data type = string
  tringCollist = [i [0] for i in df.dtypes if i [1]== 'string']
      print stringCollist

     from pyspark.sql.functions import *
        #create a function that performs a countDistinct
     distinctlist = []
     def countDistinctCats (colName):
        count = df.agg(countDistinct(colName)).collect()
        distinctlist.append (count)
        #Apply function on evrey colum in stringCollit
    map( countDistinct, stringCollist)
    print distinctlist





if __name__ == '__main__':
    #importation de données
    df = pd.read_csv("sample_submission.csv")
    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")

    # afficher les dimensions de DataFrame (nombre de lignes et nombre de colonnes)
    print("dimension de DataFrame: {} rows, {} columns".format(df.shape[0], df.shape[1]))
    print("dimension de DataFrame de teste : {} rows, {} columns".format(df_test.shape[0], df_test.shape[1]))
    print("dimension de DataFrame de train : {} rows, {} columns".format(df_train.shape[0], df_train.shape[1]))

    # Générer des statistiques descriptives
    print(df.describe())
    print(df_test.describe())
    print(df_train.describe())

    # Renvoyer le nombre de valeurs uniques de chaque classe de "is_claim"
    print("Les valeurs de classe is_claim est \n ",df_train['is_claim'].value_counts(), '\n')

    # Renvoyer le pourcentage des voitures avec ou sans assurance dans  chaque classe
    count_no_is_claim = len(df_train[df_train.is_claim == 0])
    count_is_claim = len(df_train[df_train.is_claim == 1])
    print("Pourcentage des voitures sans assurance: {:.2f}%".format(
        (count_no_is_claim / (len(df_train.is_claim)) * 100)))
    print("Pourcentage des voitures avec assurance : {:.2f}%".format(
        (count_is_claim / (len(df_train.is_claim)) * 100)), "\n")

    #conversion de données
    data_train_convertis = convert_data(df_train)

    print(data_train_convertis)
    #conversion en liste
    l = np.array(df)
    l_test = np.array(df_test)
    l_train = np.array(df_test)

    #print(l)
    #print(l_test)
    t = list(l_train[::1])
    #print("liste ",t)

def visualisation_svm():#Definition de la fonction de visualisation

  def get_hyperplane_value(x_, w, b, offset): #Definir les valeur de l'hyperplan
    return (-w[0] * x_ + b + offset) / w[1] 

  fig = plt.figure()
  ax = fig.add_subplots(1,1,1)

  plt.title("visualtion")
  plt.scatter(x[:, 0] , x[:, 1] , marker="0" , color = '#88c999' )#Definition du nuage de point , specifier sa couleur 
  plt.xlabel('a')#Les noms des axes 
  plt.ylabel('b') 

  x0_1 = np.amin(x[:, 0]) #Definir le min et max des valeurs de debut et la fin
  x0_2 = np.amax(x[:, 0]) 

  x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0) #les paramètres de l'hyperplan
  x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

  x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
  x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

  x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1) 
  x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

  ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--") #Tracer y par rapport à x sous forme de ligne
  ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
  ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

  x1_min = np.amin(x[:, 1])
  x1_max = np.amax(x[:, 1])
  ax.set_ylim([x1_min - 3, x1_max + 3]) #Définition des limites d'affichage de l'axe y

  plt.show() #Affichage du graphe

  visualisation_svm()#Appel de la fonction
