import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv ('train.csv')
train.drop(['Artist Name','Track Name','instrumentalness','Popularity','key'], axis=1, inplace=True) #тк в последних 3 столбцах содержатся пустые значения, функция dropna() не удаляет их
train.dropna()

y=train['Class'].astype('int')
x = train.drop('Class',axis=1)
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.3,random_state=17)

from sklearn.neighbors import KNeighborsClassifier
first_knn=KNeighborsClassifier()
np.mean(cross_val_score(first_knn,x_train,y_train,cv=5))


#настраиваем max_depth
from sklearn.model_selection import GridSearchCV
knn_params = {'n_neighbors':list(range(50,100,5))}
knn_grid=GridSearchCV(first_knn,knn_params, cv=5, n_jobs=-1)
knn_grid.fit(x_train,y_train)
knn_grid.best_score_, knn_grid.best_params_
print('best score = ',knn_grid.best_score_, ', best params = ',knn_grid.best_params_)


knn_valid_pred=knn_grid.predict(x_valid)
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
accuracy=accuracy_score(y_valid,knn_valid_pred)
precision=precision_score(y_valid,knn_valid_pred,average='micro')
matrix=confusion_matrix(y_valid,knn_valid_pred)
print('accuracy = ',accuracy,', precision = ',precision,'\nmatrix = ',matrix)
