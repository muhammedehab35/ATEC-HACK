#!/usr/bin/env python
# coding: utf-8

# In[1219]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_text,plot_tree,DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
import pickle


# In[1220]:


warnings.filterwarnings('ignore')


# In[1221]:


df=pd.read_csv('C:\\Users\\Muhammed ehab\\OneDrive - ESPRIT\\Bureau\\MO dataset\\df.csv')


# In[1222]:


df.columns


# In[1223]:


df.shape


# In[1224]:


df.info()


# In[1225]:


df.isnull().sum()


# In[1226]:


df.sample(8)


# In[1227]:


df.head(3)


# In[1228]:


df.UV_Radiation.value_counts()


# In[1229]:


df.info()


# In[1230]:


df.Topography.value_counts()


# In[1231]:


trnsfrm=LabelEncoder()
df['Longitude']=trnsfrm.fit_transform(df['Longitude']) # E : 0 et W:1
df['Latitude']=trnsfrm.fit_transform(df['Latitude']) # N : 0 et S:1
df['Topography']=trnsfrm.fit_transform(df['Topography']) # N : 0 et S:1
df['UV_Radiation']=trnsfrm.fit_transform(df['UV_Radiation']) # low:1,hight:0 ,moderate:2


# In[1232]:


df.head(3)


# In[1233]:


df.info()


# In[1234]:


df.dtypes.value_counts().plot.pie()


# In[1235]:


for col in df.columns:
    plt.figure()
    sns.distplot(df[col])


# In[1236]:


sns.countplot(x='Water',data=df,hue='Latitude')


# In[1237]:


for col in df.columns:
    sns.lmplot(x='Latitude',y=col,hue='Temperature',data=df)


# In[1238]:


df['Latitude'].value_counts().plot(kind='bar')


# In[1239]:


df.columns


# In[1240]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True)


# # LONGITUDE MODEL

# In[1241]:


X=df[['Temperature','Topography','Water','UV_Radiation','Longitude_degree','Latitude_degree']]
Y=df['Longitude']


# In[1242]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20,random_state=0)


# In[1243]:


model1=LogisticRegression()
model2=model=MLPClassifier(hidden_layer_sizes=[128,64,32],learning_rate='constant',
                    learning_rate_init=0.002,max_iter=100,random_state=0)  
model3=model=DecisionTreeClassifier(max_depth=3,random_state=42,criterion='entropy')
model4=model=DecisionTreeClassifier(max_depth=4,random_state=42,criterion='gini')
model5=RandomForestClassifier(n_estimators=40,max_depth=3,random_state=0,bootstrap=False)
model6=MLPRegressor(activation='relu',hidden_layer_sizes=[128,64,32])
model7=LinearRegression()


# In[1244]:


my_models=[model1,model2,model3,model4,model5]


# In[1245]:


def evaluate_regression_models(model):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(f"Test Score: {model.score(X_test, Y_test)}, Train Score: {model.score(X_train, Y_train)}")


# In[1246]:


for model in my_models:
    evaluate_regression_models(model)


# In[1247]:


Logitude_model=model4
pickle.dump(Logitude_model,open('Logitude_model.pkl','wb'))
Logitude_model=pickle.load(open('Logitude_model.pkl','rb'))


# # LATITUDE MODEL

# In[1248]:


X=df[['Temperature','Topography','Water','UV_Radiation','Longitude_degree','Latitude_degree','DISSOLVED OXYGEN(mg/L).1']]
Y=df['Latitude']


# In[1249]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20,random_state=0)


# In[1250]:


for model in my_models:
    evaluate_regression_models(model)


# In[1251]:


Latitude_model=model2
pickle.dump(Latitude_model,open('Latitude_model.pkl','wb'))
Latitude_model=pickle.load(open('Latitude_model.pkl','rb'))


# # LONGITUDE_DEGREE MODEL

# In[1262]:


X=df[['Temperature','Water','DISSOLVED OXYGEN(mg/L).1','Latitude_degree']]
Y=df['Longitude_degree']


# In[1263]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20,random_state=0)


# In[1264]:


model6.fit(X_train, Y_train)
Y_pred = model6.predict(X_test)
print(f"Test Score: {model6.score(X_test, Y_test)}, Train Score: {model6.score(X_train, Y_train)}")


# In[1265]:


Longitude_degree_model=model6


# In[1266]:


pickle.dump(Longitude_degree_model,open('Longitude_degree_model.pkl','wb'))
Longitude_degree_model=pickle.load(open('Longitude_degree_model.pkl','rb'))


# # LATITUDE_DEGREE MODEL

# In[1267]:


X=df[['Temperature','Water','Longitude','Longitude_degree']]
Y=df['Latitude_degree']


# In[1268]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20,random_state=0)


# In[1269]:


model7.fit(X_train, Y_train)
Y_pred = model7.predict(X_test)
print(f"Test Score: {model7.score(X_test, Y_test)}, Train Score: {model7.score(X_train, Y_train)}")


# In[1270]:


Latitude_degree_model=model7
pickle.dump(Latitude_degree_model,open('Latitude_degree_model.pkl','wb'))
Latitude_degree_model=pickle.load(open('Latitude_degree_model.pkl','rb'))


# In[ ]:





# In[ ]:




