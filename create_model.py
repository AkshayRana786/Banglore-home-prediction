#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


#Importing Dataset
df1 = pd.read_csv("New_House_Dataset.csv")
df1.head()


# In[3]:


# encode location column to use onehotencoder or dummy variable.
dummy = pd.get_dummies(df1['location'])
dummy


# In[4]:


# Concent the dummy variable.
df2 = pd.concat([df1,dummy.drop('other',axis=1)],axis=1)
df2


# In[5]:


df3 = df2.drop('location',axis=1)
df3.head(5)


# In[6]:


df3.shape


# In[7]:


#Set Indipendent variable.
X = df3.drop('price',axis= 1)
X.head()


# In[8]:


#Set Dipendent variable.
y = df3['price']
y.head()


# In[9]:


# Splite Training and testing data set.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# In[10]:


# Create LinearRegration model
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[11]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)

cross_val_score(LinearRegression(), X, y, cv = cv)


# In[12]:


#Serach muliple algorithum with its param

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_modle_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params' : {
                'normalize' : [True,False]
            }
        },
        'lasso' : {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,2],
                'selection' : ['random', 'cyclic']
            }
        },
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse','friedman_mse'],
                'splitter' : ['best','random']
            }
        }
    }
    
    scores = []
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv = cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model' : algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model','best_score', 'best_params'])

find_best_modle_using_gridsearchcv(X,y)


# In[15]:


X.columns


# In[16]:


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    return lr_clf.predict([x])[0]


# In[17]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[18]:


predict_price('1st Phase JP Nagar',1000,3,3)


# In[19]:


predict_price('Indira Nagar',1000,2,2)


# In[20]:


predict_price('Indira Nagar',1000,3,3)


# In[22]:


#Exporting a file as pickle.
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[24]:


# save column information in json file.
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]  #convert into lower case
}

with open("columns.json","w") as f:
    f.write(json.dumps(columns))

