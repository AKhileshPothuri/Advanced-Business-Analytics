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


# In[2]:


ap_df = pd.read_csv("inq2022.csv")


# In[3]:


ap_df.head()


# In[4]:


ap_df.info()


# In[5]:


ap_df.describe(include = 'all')


# In[6]:


ap_df.isnull().sum()/len(ap_df)


# In[7]:


ap_df = ap_df.drop(columns=['ACADEMIC_INTEREST_1','ACADEMIC_INTEREST_2','IRSCHOOL','CONTACT_CODE1','CONTACT_DATE'])


# In[8]:


ap_df.describe(include='all')


# In[9]:


ap_df.info()


# In[10]:


ap_df.isna().sum()/len(ap_df)


# In[11]:


ap_df = ap_df.drop(columns=['satscore','telecq'])


# In[12]:


ap_df = ap_df.drop(columns=['LEVEL_YEAR'])


# In[13]:


ap_df.Enroll.value_counts()


# In[14]:


cat_columns = ['ETHNICITY','TERRITORY','Instate']


# In[15]:


for col in cat_columns:
    print(ap_df[col].value_counts())
print(len(ap_df))    
print(ap_df.isna().sum())


# In[16]:


ap_df['ETHNICITY'] = ap_df['ETHNICITY'].fillna('C') ## Most frequent occurence
ap_df['TERRITORY'] = ap_df['TERRITORY'].fillna('0') ## Null % is close to 0, so filling with least occurence value as it doesn't have much impact on the model
ap_df['sex'] = ap_df['sex'].fillna(1) ## Highest occurence
ap_df['avg_income'] = ap_df['avg_income'].fillna(ap_df['avg_income'].mean()) ## Right skewed so filling nulls with mean
ap_df['distance'] = ap_df['distance'].fillna(ap_df['distance'].mean()) ## Right skewed so filling nulls with mean


# In[17]:


ap_df =pd.get_dummies(ap_df, drop_first=True)

ap_df_tree = ap_df.copy() ## Making a replica for Decision Tree Model

ap_df.describe()


# In[18]:


ap_df.info()


# In[19]:


ap_df[['TOTAL_CONTACTS','SELF_INIT_CNTCTS','TRAVEL_INIT_CNTCTS',
            'SOLICITED_CNTCTS','REFERRAL_CNTCTS','CAMPUS_VISIT',
            'sex','mailq','premiere','interest','stucar','init_span','int1rat',
            'int2rat','hscrat','avg_income','distance']].skew(axis = 0, skipna = True)


# In[20]:


ap_df[['TOTAL_CONTACTS','SELF_INIT_CNTCTS','TRAVEL_INIT_CNTCTS',
            'SOLICITED_CNTCTS','REFERRAL_CNTCTS','CAMPUS_VISIT',
            'sex','mailq','premiere','interest','stucar','init_span','int1rat',
            'int2rat','hscrat','avg_income','distance']].hist(bins=30, figsize=(20, 15))


# In[21]:


ap_df['TOTAL_CONTACTS'] = np.where(ap_df['TOTAL_CONTACTS']>0,1,0) 
ap_df['SELF_INIT_CNTCTS'] = np.where(ap_df['SELF_INIT_CNTCTS']>0,1,0)


# In[22]:


ap_df.drop(columns=['Enroll']).hist(bins=30, figsize=(20, 15))


# Considering VIF threshold as 10

# In[23]:


ap_vif = pd.DataFrame()
ap_vif["feature"] = ap_df.drop(columns=['Enroll']).columns
  
ap_vif["VIF"] = [variance_inflation_factor(ap_df.drop(columns=['Enroll']).values, i)
                          for i in range(len(ap_df.drop(columns=['Enroll']).columns))]
  
print(ap_vif)


# In[24]:


ap_df.corr()


# In[25]:


ap_df = ap_df.drop(columns=['TOTAL_CONTACTS']) ### Highest VIF value


# In[26]:


ap_vif = pd.DataFrame()
ap_vif["feature"] = ap_df.drop(columns=['Enroll']).columns
  
ap_vif["VIF"] = [variance_inflation_factor(ap_df.drop(columns=['Enroll']).values, i)
                          for i in range(len(ap_df.drop(columns=['Enroll']).columns))]
  
print(ap_vif)


# In[27]:


ap_df.corr()


# In[28]:


ap_df = ap_df.drop(columns=['mailq']) ### Highest VIF value


# In[29]:


ap_vif = pd.DataFrame()
ap_vif["feature"] = ap_df.drop(columns=['Enroll']).columns
  
ap_vif["VIF"] = [variance_inflation_factor(ap_df.drop(columns=['Enroll']).values, i)
                          for i in range(len(ap_df.drop(columns=['Enroll']).columns))]
  
print(ap_vif)


# ## Regression Model

# In[30]:


ap_X_train, ap_X_val, ap_y_train, ap_y_val = train_test_split(ap_df.drop(columns=['Enroll']),ap_df['Enroll'], test_size=0.3,random_state=0)
ap_log_reg = sm.Logit(ap_y_train, ap_X_train).fit()
print(ap_log_reg.summary())


# In[31]:


ap_prediction_prob = ap_log_reg.predict(ap_X_val)
ap_prediction = list(map(round, ap_prediction_prob))
confusion_matrix(ap_y_val,ap_prediction)


# In[32]:


lr_auc = roc_auc_score(ap_y_val, ap_prediction_prob)
print('Logistic: ROC AUC=%.3f' % (lr_auc))


# In[33]:


lr_fpr, lr_tpr, _ = roc_curve(ap_y_val, ap_prediction_prob)
plt.plot(lr_fpr, lr_tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# ## Tree Model

# In[34]:


ap_X_train, ap_X_val, ap_y_train, ap_y_val = train_test_split(ap_df_tree.drop(columns=['Enroll']),ap_df_tree['Enroll'], test_size=0.3,random_state=0)
ap_dtree = tree.DecisionTreeClassifier(max_depth=4,min_samples_split=30)
ap_dtree = ap_dtree.fit(ap_X_train, ap_y_train)


# In[35]:


ap_r = export_text(ap_dtree, feature_names=list(ap_X_train.columns.values))
print(ap_r)


# In[36]:


plt.figure(figsize=[25,20])
tree.plot_tree(ap_dtree,
               feature_names=list(ap_X_train.columns.values),
               class_names=True,
               filled=True)
plt.show()


# In[37]:


ap_prediction =ap_dtree.predict(ap_X_val)
confusion_matrix(ap_y_val,ap_prediction)


# In[38]:


ap_dtree.score(ap_X_val,ap_y_val)


# In[39]:


ap_prediction_prob = ap_dtree.predict_proba(ap_X_val)
ap_tree_auc = roc_auc_score(ap_y_val, ap_prediction_prob[:,1])
print('Decision Tree: ROC AUC=%.3f' % (ap_tree_auc))


# In[40]:


tree_fpr, tree_tpr, _ = roc_curve(ap_y_val, ap_prediction_prob[:,1])
plt.plot(tree_fpr, tree_tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

