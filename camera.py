#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pandas
#!pip install matplotlib
#!pip install scikit-learn


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data_train=pd.read_csv(r"C:\Users\hp\Desktop\camera_dataset.csv")


# In[3]:


data_train


# # Explortory Data Analysis

# In[4]:


data_train.shape


# In[5]:


data_train.info()


# In[6]:


data_train.isnull().sum()


# In[7]:


data_train.describe()


# In[62]:


data_train.plot(x='Price',y='Max resolution',kind='scatter')
plt.show()


# In[8]:


data_train.plot(x='Price',y='Low resolution',kind='scatter')
plt.show()


# In[9]:


data_train.plot(kind='box',figsize=(20,10))
plt.show()


# In[11]:


Y=data_train['Price']


# In[12]:


Y


# In[13]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[14]:


data_train=data_train.drop('Price',axis=1)
X=data_train


# In[15]:


X.head()


# In[16]:


X_std=std.fit_transform(X)

dataset_std=std.transform(data_train)


# In[17]:


X_std


# In[18]:


from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y)


# # Training The Model

# # 1) Decision Tree

# In[19]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[20]:


dt.fit(dataset_std,y_transformed)


# In[21]:


dt.predict(dataset_std)


# # 2) KNN

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[23]:


knn.fit(dataset_std,y_transformed)


# In[24]:


knn.predict(dataset_std)


# # 3) Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[26]:


lr.fit(dataset_std,y_transformed)


# In[27]:


lr.predict(dataset_std)


# # As we predicted on Test Data csv, we are not able to plot accuracy score as we dont have Ground Truth, so we are going to use only Train.csv and gonna split it into train and test

# In[28]:


X


# In[29]:


Y


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[32]:


X_train


# In[33]:


Y_train


# In[34]:


X_test


# In[35]:


Y_test


# In[36]:


from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
yt = lab.fit_transform(Y_train)


# # 1) Decision Tree

# #### We dont need to use Standard Scaler for DT since distance doesnt matter here

# In[57]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()


# In[58]:


dt.fit(X_train,yt)


# In[59]:


Y_pred=dt.predict(X_test).round()


# In[60]:


Y_pred


# In[61]:


Y_test


# In[62]:


from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# In[63]:


dt_ac=r2_score(Y_test,Y_pred)


# In[65]:


dt_ac


# # 2) KNN

# In[45]:


#### We need to use Standard Scaler for knn since distance matter here


# In[46]:


X_train_std=std.fit_transform(X_train)

X_test_std=std.transform(X_test)


# In[47]:


X_test_std


# In[48]:


from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y_train)


# In[49]:


from sklearn.ensemble import RandomForestRegressor


# In[50]:


rf_classifier = RandomForestRegressor(n_estimators=100, random_state=42)
rf_classifier.fit(X_train,Y_train)
y_pred = rf_classifier.predict(X_test)


# In[51]:


y_pred


# In[52]:


from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y_pred)


# In[53]:


rf_ac=r2_score(Y_test,y_pred)
rf_ac


# In[54]:


plt.bar(x=['dt','rf'],height=[dt_ac,rf_ac])
plt.xlabel("Algorithms")
plt.ylabel("R2 Score")
plt.show()


# In[ ]:


import pickle
# open a file, where you ant to store the data
file = open('camera.pkl', 'wb')

# dump information to that file
pickle.dump(rf_classifier, file)

