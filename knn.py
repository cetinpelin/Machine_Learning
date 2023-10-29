#!/usr/bin/env python
# coding: utf-8

# In[1]:


#KNN (K-Nearest Neighbors)
#1. Exploratory Data Analysis
#2. Data Preprocessing
#3. Model & Prediction
#4. Model Evaluation
#5. Hyperparameter Optimization
#6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)


# In[2]:


#1. Exploratory Data Analysis
df = pd.read_csv("diabetes.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe().T


# In[5]:


df["Outcome"].value_counts()


# In[6]:


#2. Data Preprocessing & Feature Engineering

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)


# In[7]:


#3. Model & Prediction
knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)


# In[8]:


#4. Model Evaluation

#Confusion matrix için y_pred:

y_pred = knn_model.predict(X)

#AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))


# In[9]:


#AUC
roc_auc_score(y, y_prob)


# In[10]:


cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])


# In[11]:


cv_results['test_accuracy'].mean()


# In[12]:


cv_results['test_f1'].mean()


# In[13]:


cv_results['test_roc_auc'].mean()


# In[15]:


#Başarı nasıl artar?
#1. Örnek boyutu artırılabilir
#2. Veri ön işleme
#3. Özelliklik mühendisliği(yeni veriler türetilebilir)
#4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()


# In[17]:


#5. Hyperparameter Optimization

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)


# In[18]:


knn_gs_best.best_params_


# In[21]:


#6. Final Model

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()


# In[22]:


cv_results['test_f1'].mean()


# In[23]:


cv_results['test_roc_auc'].mean()


# In[26]:


random_user = X.sample(1)

knn_final.predict(random_user)


# In[ ]:




