#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("winequality-white.csv" , sep=";")
df.head()


# In[3]:


df.shape


# In[4]:


df.columns.values


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# # Check Duplicate values

# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates(inplace = True)


# In[10]:


df.shape


# # Traget variable

# In[11]:


df.quality.unique()


# In[12]:


df.quality.value_counts()


# # Checking Outliers

# In[14]:


cols =df.columns.values
for i in cols:
  
    sns.boxplot(df[i],whis=1.5,color='red',orient='h')
    plt.show();
    


# # Dealing with outliers

# In[15]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[16]:


for column in df[cols].columns:
    lr,ur=remove_outlier(df[column])
    df[column]=np.where(df[column]>ur,ur,df[column])
    df[column]=np.where(df[column]<lr,lr,df[column])


# In[17]:


cols =df.columns.values
for i in cols:
    sns.boxplot(df[i],whis=1.5,color='blue',orient='h')
    plt.show();


# # Distribution and skewness

# In[21]:


df.hist(figsize=(20,30));


# # Checking correlation

# In[22]:


k = 12 #number of variables for heatmap
cols = df.corr().nlargest(k, 'quality')['quality'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True)


# # Visualize Bivariate data

# In[23]:


df_attr = ( df[cols])
sns.pairplot(df_attr, diag_kind='kde',hue='quality',height=2.5,)  
plt.show()


# In[24]:


df.columns


# In[26]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df.drop('quality', axis=1)
X = scaler.fit_transform(X)


# In[31]:


new_df = pd.DataFrame(X, columns=df.drop('quality',axis=1).columns)


# In[32]:


new_df.shape


# # Splitting the data

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_new=df[['alcohol', 'density', 'volatile acidity', 'chlorides']]
Y_new=df[['quality']]


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size=0.2, random_state=40)


# In[34]:


X_train.head()


# In[35]:


Y_train.head()


# In[36]:


from sklearn.preprocessing import StandardScaler
X_features = df
X = StandardScaler().fit_transform(X)


# In[37]:


X_features.head()


# # Model deployment

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics 


# In[39]:


X_train_LR = X_train[['alcohol']]
Y_train_LR = Y_train[['quality']]


# In[40]:


X_test_LR = X_test[['alcohol']]
Y_test_LR = Y_test[['quality']]


# In[41]:


lm = LinearRegression()
lm.fit(X_train_LR,Y_train_LR)


# In[42]:


Y_train_LR_predict = lm.predict(X_train_LR)
print('Y_train_predict = \n ',Y_train_LR_predict)
print()
Y_test_LR_predict = lm.predict(X_test_LR)
print('Y_test_predict = \n', Y_test_LR_predict)


# In[43]:


a_lm = lm.intercept_
b_lm= lm.coef_
print('a=',a_lm)
print()
print('b=',b_lm)


# # Evaluating the result

# In[44]:


from sklearn.metrics import mean_squared_error


# In[45]:


r_square = lm.score(X_train_LR,Y_train_LR)
print('R-square: ', r_square)
mse = mean_squared_error(Y_train_LR, Y_train_LR_predict)
print('MSE: ', mse)


# In[47]:


r_square = lm.score(X_test_LR,Y_test_LR)
print('R-square: ', r_square)
mse = mean_squared_error(Y_test_LR, Y_test_LR_predict)
print('MSE: ', mse)


# In[48]:


Y_train_LR_predict= np.round_(Y_train_LR_predict)
print(Y_train_LR_predict)


# In[49]:


print('R-square: ', lm.score(X_train_LR,Y_train_LR_predict) )
print('MSE:',mean_squared_error(Y_train, Y_train_LR_predict))


# In[52]:


Y_test_LR_predict = np.round_(Y_test_LR_predict)
print(Y_test_LR_predict)


# In[53]:


print('R-square: ', lm.score(X_test_LR,Y_test_LR_predict) )
print('MSE:',mean_squared_error(Y_test, Y_test_LR_predict))


# In[ ]:




