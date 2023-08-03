#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


import pandas as pd
df = pd.read_csv('crop_production.csv',na_values='=')
df


# In[3]:


df=df[df['District_Name'] == 'CUTTACK']
df.info()


# In[4]:


df


# In[5]:


df=df[df['Crop_Year']>=2000]
df

df= df.join(pd.get_dummies(df['District_Name']))
#df= df.join(pd.get_dummies(df['Season']))
df= df.join(pd.get_dummies(df['Crop']))
df= df.join(pd.get_dummies(df['State_Name']))
df
# In[6]:


plt.figure(figsize = (6,6))
segment = df['Season'].value_counts()
segment_label = df['Season'].unique()
color = ('LightPink', "LightBlue" , 'LightGreen','red','green','Gold')

plt.pie(segment,
       autopct = '%1.1f%%',
       labels = segment_label,
       explode = (0.06,0.05,0.05,0.07,0.08,0.05),
       shadow = True,
       colors = color);


# In[7]:


sns.catplot(data=df,x="Crop_Year",aspect=2,kind='count')


# In[8]:


df.describe()


# In[9]:


df['Yield'] = df['Production']/df['Area']
df


# In[10]:


df['Yield'] 
df


# In[11]:


corr=df[['Area','Production']].corr()
sns.heatmap(corr,annot = True , cmap = 'YlGnBu')


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sb

C_mat = df.corr()
fig=plt.figure(figsize = (6,6))

sb.heatmap(C_mat, vmax = .4, square = True)
plt.show()


# In[13]:


from sklearn import preprocessing


# In[14]:


# Creat x, where x the scores columns values as floats 
x=df[['Yield']].values.astype(float)
x
#creat a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

x_scaled

df['Yield']= x_scaled
df


# In[15]:


df=df.fillna(df.mean())


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


a=df


# In[18]:


b=df['Yield']


# In[19]:


a=df.drop('Yield',axis=1)


# In[20]:


len(a.columns)


# In[21]:


a.columns


# In[22]:


from sklearn.preprocessing import StandardScaler
TargetVariable=['Yield']
Predictors=['Crop_Year', 'Area','Production']
 
a=df[Predictors].values
b=df[TargetVariable].values
 
### Sandardization of data ###
#from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(a)
TargetVarScalerFit=TargetVarScaler.fit
 

 
#Split the data into training and testing set
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=1)
 
# Quick sanity check with the shapes of Training and testing datasets
print(a_train.shape)
print(b_train.shape)
print(a_test.shape)
print(b_test.shape)


# In[23]:


from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=2,random_state=0,n_estimators=100)
regr.fit(a_train,b_train)
b_pred= regr.predict(a_test)


# In[24]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score


# In[25]:


print('MSE=',mse(b_pred,b_test))
print('MAE=',mae(b_pred,b_test))
print('R2 Score =',r2_score(b_pred,b_test))


# # ANN

# In[26]:


# Tensorflow
import tensorflow as tf
from keras import callbacks
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras import models


# In[27]:


# importing the libraries
from keras.models import Sequential
from keras.layers import Dense
 
# create ANN model
model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=30, input_dim=3, kernel_initializer='normal', activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=40, kernel_initializer='normal', activation='tanh'))
 
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))
 
# Compiling the model
from keras import metrics 
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()])

#model.compile(loss = 'mean_squared_error',  
#optimizer = 'sgd', metrics = [mean_absolute_error])
 
 #Fitting the ANN to the Training set
model.fit(a_train, b_train ,batch_size = 5, epochs = 5, verbose=1)


# In[28]:


from sklearn.metrics import mean_absolute_error as mae
print('MAE=',mae(b_pred,b_test))

