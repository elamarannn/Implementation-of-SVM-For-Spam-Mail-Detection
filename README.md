# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Elamaran S E
RegisterNumber:  212222230036
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Result output:

![282243022-28a5795a-2580-433a-9443-e2f07c687b5e](https://github.com/elamarannn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497531/f3c8e44d-f51c-40a5-a248-3316e65e4c8d)
data.head():

![282243035-5cdf8c27-cb0a-43b9-86c0-a2db5671c78d](https://github.com/elamarannn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497531/c1b72c58-cf5c-440c-bb3a-ce7839d00bbd)
data.info():

![282243043-6c9e9c39-9def-41c3-993e-73ef4e37ac30](https://github.com/elamarannn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497531/29b35d78-006c-43ad-ad49-4fe46eea88f5)
data.isnull().sum():

![282243050-8cc08474-d436-4d1f-b9fd-300e03f40aca](https://github.com/elamarannn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497531/afa270a4-19d4-4a09-b0b7-c41de6d7dbcb)
Y_prediction value:

![282243051-4ae9ec2d-9001-432f-8bb9-9ae3de9e2311](https://github.com/elamarannn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497531/541430cf-9c49-45e8-a58e-0b107953917c)
Accuracy value:

![282243052-7d5ffca2-ba4e-4690-b5d1-524d12659f1c](https://github.com/elamarannn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497531/3eecdfe0-cd6c-4c39-bb57-3e0323b2769d)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
