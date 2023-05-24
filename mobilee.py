#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!pip install pandas
#!pip install matplotlib
#!pip install scikit-learn


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data_train=pd.read_csv(r"C:\Users\hp\Downloads\Mobile-Price-Classification-main\Mobile-Price-Classification-main\train.csv")
data_test=pd.read_csv(r"C:\Users\hp\Downloads\Mobile-Price-Classification-main\Mobile-Price-Classification-main\test.csv")


# In[3]:


data_train


# In[4]:


data_test


# # Explortory Data Analysis

# In[5]:


data_train.shape


# In[6]:


data_test.shape


# In[7]:


data_train.info()


# In[8]:


data_test.info()


# In[9]:


data_train.isnull().sum()


# In[10]:


data_test.isnull().sum()


# In[11]:


data_train.describe()


# In[12]:


data_train.plot(x='price_range',y='ram',kind='scatter')
plt.show()


# In[13]:


data_train.plot(x='price_range',y='battery_power',kind='scatter')
plt.show()


# In[14]:


data_train.plot(x='price_range',y='fc',kind='scatter')
plt.show()


# In[15]:


data_train.plot(x='price_range',y='n_cores',kind='scatter')
plt.show()


# In[ ]:





# In[16]:


data_train.plot(kind='box',figsize=(20,10))
plt.show()


# In[17]:


X=data_train.drop('price_range',axis=1)


# In[18]:


X


# In[19]:


data_test=data_test.drop('id',axis=1)


# In[20]:


data_test.head()


# In[21]:


data_test.shape


# In[22]:


Y=data_train['price_range']


# In[23]:


Y


# In[24]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[25]:


X_std=std.fit_transform(X)

data_test_std=std.transform(data_test)


# In[26]:


X_std


# In[27]:


data_test_std


# # Training The Model

# # 1) Decision Tree

# In[28]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[29]:


dt.fit(X_std,Y)


# In[30]:


dt.predict(data_test_std)


# In[31]:


data_test


# # 2) KNN

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[33]:


knn.fit(X_std,Y)


# In[34]:


knn.predict(data_test_std)


# # 3) Logistic Regression

# In[35]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[36]:


lr.fit(X_std,Y)


# In[37]:


lr.predict(data_test_std)


# # As we predicted on Test Data csv, we are not able to plot accuracy score as we dont have Ground Truth, so we are going to use only Train.csv and gonna split it into train and test

# In[38]:


X


# In[39]:


Y


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[42]:


X_train


# In[43]:


Y_train


# In[44]:


X_test


# In[45]:


Y_test


# # 1) Decision Tree

# #### We dont need to use Standard Scaler for DT since distance doesnt matter here

# In[46]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[47]:


dt.fit(X_train,Y_train)


# In[48]:


Y_pred=dt.predict(X_test)


# In[49]:


Y_pred


# In[50]:


Y_test


# In[51]:


from sklearn.metrics import accuracy_score


# In[52]:


dt_ac=accuracy_score(Y_test,Y_pred)


# In[53]:


dt_ac


# # 2) KNN

# In[54]:


#### We need to use Standard Scaler for knn since distance matter here


# In[55]:


X_train_std=std.fit_transform(X_train)

X_test_std=std.transform(X_test)


# In[56]:


X_test_std


# In[ ]:





# In[57]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[58]:


knn.fit(X_train_std,Y_train)


# In[59]:


Y_pred=knn.predict(X_test_std)


# In[60]:


Y_pred


# In[61]:


Y_test


# In[62]:


knn_ac=accuracy_score(Y_test,Y_pred)


# In[63]:


knn_ac


# # 3) Logistic Regression

# In[64]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[65]:


lr.fit(X_train_std,Y_train)


# In[66]:


Y_pred=lr.predict(X_test_std)


# In[67]:


Y_pred


# In[68]:


lr_ac=accuracy_score(Y_test,Y_pred)


# In[69]:


lr_ac


# In[161]:


plt.bar(x=['dt','knn','lr'],height=[dt_ac,knn_ac,lr_ac])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()


# In[ ]:


import pickle
# open a file, where you ant to store the data
file = open('mobile.pkl', 'wb')

# dump information to that file
pickle.dump(lr, file)

