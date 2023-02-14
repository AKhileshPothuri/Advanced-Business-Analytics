#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[2]:


ap_df = pd.read_csv("BostonHousing.csv")


# In[3]:


ap_df.isna().sum()


# In[4]:


ap_df.describe()


# In[5]:


ap_df.info()


# In[6]:


ap_df.head()


# In[7]:


ap_df = ap_df.drop(columns=['MEDV'])
ap_df.head()


# In[8]:


for column in ap_df:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=ap_df, x=column)


# In[9]:


ap_df.hist(bins=30, figsize=(20, 15))


# In[10]:


for column in ap_df:
    print(column + " Mean :" + str(ap_df[column].mean()))
    print(column + " Median :" + str(ap_df[column].median()))
    print()


# In[11]:


ap_df['CAT. MEDV'].value_counts()


# In[12]:


ap_y = ap_df['CAT. MEDV']
ap_X = ap_df.drop(columns=['CAT. MEDV'])


# In[13]:


ap_X_train, ap_X_val, ap_y_train, ap_y_val = train_test_split(ap_X,ap_y, test_size=0.3,random_state=0)


# ## Random Forest

# In[14]:


ap_param ={ 
    'n_estimators': [50,80,100,150], 
    'max_features': range(1,12) 
} 
ap_rf = RandomForestClassifier(random_state=0) 
ap_grid_search = GridSearchCV(estimator=ap_rf, param_grid=ap_param,n_jobs=-1) 
ap_grid_search.fit(ap_X_train,ap_y_train) 


# In[15]:


ap_grid_search.best_params_ 


# In[16]:


ap_grid_search.score(ap_X_val,ap_y_val) 


# In[17]:


ap_final_model = ap_grid_search.best_estimator_
ap_feature_imp = pd.Series(ap_final_model.feature_importances_,index=ap_X.columns).sort_values(ascending = False)
ap_feature_imp


# ## SVM

# In[18]:


ap_y = ap_df['CAT. MEDV']
ap_X = ap_df.drop(columns=['CAT. MEDV'])
ap_X = StandardScaler().fit_transform(ap_X)


# In[19]:


ap_X_train, ap_X_val, ap_y_train, ap_y_val = train_test_split(ap_X,ap_y, test_size=0.3,random_state=0)


# ### Linear

# In[20]:


ap_linearSVM = svm.SVC(kernel='linear')
ap_linearSVM.fit(ap_X_train,ap_y_train)


# In[21]:


ap_linearSVM.C


# In[22]:


ap_linearSVM.score(ap_X_val,ap_y_val)


# ### Radial

# In[23]:


ap_radialSVM = svm.SVC(kernel='rbf')
ap_radialSVM.fit(ap_X_train,ap_y_train)


# In[24]:


ap_radialSVM.C


# In[25]:


ap_radialSVM.score(ap_X_val,ap_y_val)


# In[26]:


ap_param = {'C': [0.1,0.5, 1, 5, 10],
            'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'linear']}

ap_SVM=svm.SVC()
ap_grid = GridSearchCV(estimator=ap_SVM,param_grid=ap_param,verbose=3,cv=10)
         
ap_grid.fit(ap_X_train, ap_y_train)


# In[27]:


ap_grid.best_params_


# In[28]:


ap_grid.score(ap_X_val,ap_y_val)


# In[ ]:




