#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm,preprocessing
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('train.csv')
le = preprocessing.LabelEncoder()
le.fit(data['id'])
transformed=le.transform(data['id'])
x_train = data
y_train = data['target']
x_train = x_train.drop(['target'],axis=1)
x_train['id']=transformed


# In[3]:


from sklearn.model_selection import cross_val_score
y_train = data['target']
model = svm.SVC(kernel='rbf',gamma=10,C=100,verbose=1)
#model.fit(x_train,y_train)
from sklearn.model_selection import train_test_split
x_train_split=np.array_split(x_train,10)
y_train_split=np.array_split(y_train,10)
print(y_train.shape,x_train.shape)


# In[ ]:


for i in range(0,100):
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(x_train_split[i],y_train_split[i],test_size=0.1,random_state=0)
    model.fit(Xtrain,Ytrain)
    print(model.score(Xtest,Ytest))


# In[ ]:




