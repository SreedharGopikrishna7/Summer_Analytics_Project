#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np


# In[32]:


import warnings
warnings.filterwarnings('ignore')


# In[33]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[34]:


train = train.drop(['norating1'], axis=1)
test = test.drop(['norating1'], axis=1)
#train['norating1'] = train['star_5f']+train['star_4f']+train['star_3f']+train['star_2f']+train['star_1f']


# In[35]:


train['Gender']=train['title'].astype(str).str.contains('Women')
train['Gender_M']=train['title'].astype(str).str.contains('Men')


# In[36]:


test['Gender']=test['title'].astype(str).str.contains('Women')
test['Gender_M']=test['title'].astype(str).str.contains('Men')


# In[37]:


train.shape


# In[38]:


test.shape


# In[39]:


train.isnull().sum()


# In[40]:


test.info()


# In[41]:


train = train.drop(['maincateg','title', 'noreviews1'], axis=1)


# In[42]:


test = test.drop(['maincateg','title', 'noreviews1'], axis=1)


# In[43]:


pd.unique(train['platform'])


# In[44]:


train['platform'].value_counts()


# In[45]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train['company']= label_encoder.fit_transform(train['platform'])
train = train.drop(['platform'], axis=1)


# In[46]:


train['Gender']= label_encoder.fit_transform(train['Gender'])
train['Gender_M']= label_encoder.fit_transform(train['Gender_M'])


# In[47]:


test['Gender']= label_encoder.fit_transform(test['Gender'])
test['Gender_M']= label_encoder.fit_transform(test['Gender_M'])


# In[48]:


train.head()


# In[49]:


train['Gender'].value_counts()


# In[50]:


train['Gender_M'].value_counts()


# In[51]:


label_encoder = preprocessing.LabelEncoder()
test['company']= label_encoder.fit_transform(test['platform'])
test = test.drop(['platform'], axis=1)


# In[52]:


test.head()


# In[53]:


train.isnull().sum()


# In[54]:


test.isnull().sum()


# In[55]:


train = train.fillna(train.median())


# In[56]:


train.isnull().sum()


# In[57]:


test = test.fillna(train.median())


# In[58]:


train['norating1'] = train['star_5f']+train['star_4f']+train['star_3f']+train['star_2f']+train['star_1f']
test['norating1'] = test['star_5f']+test['star_4f']+test['star_3f']+test['star_2f']+test['star_1f']


# In[59]:


test.isnull().sum()


# In[60]:


X_train = train.drop(['price1','Offer %'], axis=1)
y_train = train.price1
X_test = test.copy()


# In[61]:


X_train.tail()


# In[62]:


y_train.head()


# In[63]:


from sklearn.model_selection import train_test_split
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# In[72]:


from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor

#rf_model = RandomForestRegressor(n_estimators=1000,random_state=2021, max_depth=100)
EXT_model = ExtraTreesRegressor(random_state = 42)
EXT_model.fit(X_train_train, y_train_train)
y_preds = EXT_model.predict(X_train_test)


# In[73]:


print(mean_absolute_error(y_preds, y_train_test))


# In[74]:


EXT_model.fit(X_train, y_train)


# In[75]:


y_preds = EXT_model.predict(X_test)


# In[76]:


id = test.id
output = pd.DataFrame(y_preds, columns =['price1']) 
output['id'] = id
output = output[['id', 'price1']]


# In[77]:


output.head()


# In[78]:


output.shape


# In[79]:


output.to_csv('Prediction11.csv', index=False)


# In[ ]:




