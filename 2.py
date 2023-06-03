import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
train = pd.read_csv ('train.csv')
train.drop(['Artist Name','Track Name','instrumentalness','Popularity','key'], axis=1, inplace=True) #тк в последних 3 столбцах содержатся пустые значения, функция dropna() не удаляет их

y=train['Class'].astype('int')
x = train.drop('Class',axis=1)
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.3,random_state=17)

nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
print('Metrics on Test Set:')
y_pred = nb_classifier.predict(x_valid)
print('Accuracy:', accuracy_score(y_valid, y_pred))
print('Precision:',precision_score(y_valid, y_pred, average='macro'))
print('Matrix:\n', confusion_matrix(y_valid, y_pred))
