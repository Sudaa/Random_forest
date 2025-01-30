## import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## import dataset
df = pd.read_csv("Social_Network_Ads.csv")
X = df.iloc[:,[2,3]].values
Y = df.iloc[:,4].values

## splitting dataaset 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

## feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## fitting the classifier into the trainning set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 190, criterion = 'entropy', random_state= 0)
classifier.fit(X_train,Y_train)

## predicting the test set result
Y_pred = classifier.predict(X_test)

## confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(Y_test, Y_pred)

##visualize confusion matrix 
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No','Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.show()

print(classification_report(Y_test, Y_pred))
