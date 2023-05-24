#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pandas
#!pip install matplotlib
#!pip install scikit-learn


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data_train=pd.read_csv(r"C:\Users\hp\Downloads\Car_Price_deployment-main\Car_Price_deployment-main\car data.csv")


# In[4]:


data_train


# # Explortory Data Analysis

# In[5]:


data_train.shape


# In[6]:


data_train.info()


# In[7]:


data_train.isnull().sum()


# In[8]:


data_train.describe()


# In[9]:


data_train.plot(x='Present_Price',y='Selling_Price',kind='scatter')
plt.show()


# In[ ]:





# In[10]:


data_train.plot(x='Present_Price',y='Kms_Driven',kind='scatter')
plt.show()


# In[11]:


data_train.plot(kind='box',figsize=(20,10))
plt.show()


# In[12]:


Y=data_train['Selling_Price']


# In[13]:


Y


# In[14]:


data_train=data_train.drop('Car_Name',axis=1)


# In[15]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[16]:


unique_values = data_train['Fuel_Type'].value_counts()
print(unique_values)


# In[17]:


data_train['Fuel_Type'] =data_train['Fuel_Type'].replace({'Petrol': '1'})
data_train['Fuel_Type'] =data_train['Fuel_Type'].replace({'Diesel': '2'})
data_train['Fuel_Type'] =data_train['Fuel_Type'].replace({'CNG': '3'})
data_train.head()


# In[18]:


unique_values = data_train['Seller_Type'].value_counts()
print(unique_values)


# In[19]:


data_train['Seller_Type'] =data_train['Seller_Type'].replace({'Dealer': '1'})
data_train['Seller_Type'] =data_train['Seller_Type'].replace({'Individual': '2'})
data_train.head()


# In[20]:


unique_values = data_train['Transmission'].value_counts()
print(unique_values)


# In[21]:


data_train['Transmission'] =data_train['Transmission'].replace({'Manual': '1'})
data_train['Transmission'] =data_train['Transmission'].replace({'Automatic': '2'})
data_train.head()


# In[22]:


data_train=data_train.drop('Selling_Price',axis=1)
X=data_train


# In[23]:


X.head()


# In[24]:


X_std=std.fit_transform(X)

dataset_std=std.transform(data_train)


# In[25]:


X_std


# In[26]:


from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y)


# # Training The Model

# # 1) Decision Tree

# In[27]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[28]:


dt.fit(dataset_std,y_transformed)


# In[29]:


dt.predict(dataset_std)


# # 2) KNN

# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[31]:


knn.fit(dataset_std,y_transformed)


# In[32]:


knn.predict(dataset_std)


# # 3) Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[34]:


lr.fit(dataset_std,y_transformed)


# In[35]:


lr.predict(dataset_std)


# # As we predicted on Test Data csv, we are not able to plot accuracy score as we dont have Ground Truth, so we are going to use only Train.csv and gonna split it into train and test

# In[36]:


X


# In[37]:


Y


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[40]:


X_train


# In[41]:


Y_train


# In[42]:


X_test


# In[43]:


Y_test


# In[44]:


from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
yt = lab.fit_transform(Y_train)


# # 1) Decision Tree

# #### We dont need to use Standard Scaler for DT since distance doesnt matter here

# In[45]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[46]:


dt.fit(X_train,yt)


# In[47]:


Y_pred=dt.predict(X_test).round()


# In[48]:


Y_pred


# In[49]:


Y_test


# In[50]:


from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# In[51]:


dt_ac=r2_score(Y_test,Y_pred)


# In[52]:


dt_ac


# # 2) KNN

# In[53]:


#### We need to use Standard Scaler for knn since distance matter here


# In[54]:


X_train_std=std.fit_transform(X_train)

X_test_std=std.transform(X_test)


# In[55]:


X_test_std


# In[56]:


from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y_train)


# In[57]:


from sklearn.ensemble import RandomForestRegressor


# In[58]:


rf_classifier = RandomForestRegressor(n_estimators=100, random_state=42)
rf_classifier.fit(X_train,Y_train)
y_pred = rf_classifier.predict(X_test)


# In[59]:


y_pred


# In[60]:


from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y_pred)


# In[61]:


rf_ac=r2_score(Y_test,y_pred)
rf_ac


# In[62]:


plt.bar(x=['dt','rf'],height=[dt_ac,rf_ac])
plt.xlabel("Algorithms")
plt.ylabel("R2 Score")
plt.show()


# In[63]:


import pickle
# open a file, where you ant to store the data
file = open('car.pkl', 'wb')

# dump information to that file
pickle.dump(rf_classifier, file)

