import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import io
import requests
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = pd.read_csv("winequality-red.csv")

arraydata = data.values

X = arraydata[:, 0:11]
Y = arraydata[:, 11:12]


vs = 0.05
sd = 7

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=vs, random_state=sd)

model =
model.fit(Xtrain, Ytrain)
predictions = model.predict(Xtest)
print(Xtest.shape)
print("Model --> Logistic Regression")
print("Overall Accuracy: {}").format(accuracy_score(Ytest, predictions) * 100)

new_data = [(10,0.83,0.61,1.9,0.10,10,40,0.9966,3.97,0.99,9.5)]
new_array = np.asarray(new_data)

prediction = model.predict(new_array)
print prediction
y1= arraydata[1520:1599,11:12]
#print y1
plt.plot(y1,color= 'red')
plt.plot(Xtest)
plt.show()
#print(Xtrain.shape)