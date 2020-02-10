#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Supervised Learning  - Linear Regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# In[14]:


df = pd.read_csv("Life2.csv", thousands=',')


# In[18]:


df


# In[16]:


df.plot(kind='scatter', x="GDP", y='Life_satisfaction')


# In[20]:


model = sklearn.linear_model.LinearRegression()  # select Model for train


# In[30]:


model.fit(df[['GDP']], df[['Life_satisfaction']])   # for 2d array  train


# In[37]:


model.predict([[60000]])  #predict(self, X) Predict using the linear model.  testing


# In[36]:


model.score(df[['GDP']], df[['Life_satisfaction']])   # how accurate the model   accuracy

