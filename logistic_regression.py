#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Diabetes Prediction with Logistic Regression

#İş problemi:

#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi
#modeli geliştirebilir misiniz?

#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitülerinde tutulan büyük veri setinin parçasıdır.
#ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenic şehrinde yaşayan 21 yaş ve üzerinde olan 
#Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal bagımsız 
#değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun 
#pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

#Değişkenler

#Pregnancies: Hamilelik sayısı
#Glucose: Glikoz
#BloodPressure: Kan basıncı
#Insulin : Insülim
#BMI : Beden kitle indeksi
#DiabetesPredigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
#Age : yaş (yıl)
#Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

#1. Exploratory Data Analysis
#2. Data Preprocessing
#3. Model & Prediction
#4. Model Evaluation
#5. Model Validation:Holdout
#6. Model Validation: 10-Fold Cross Validation
#7. Prediction for a New Observation


get_ipython().system('pip install --upgrade scikit-learn')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate



# In[3]:


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# In[4]:


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# In[5]:


def replace_with_thresholds(dataframe, variable):...


# In[6]:


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[7]:


#Exploratory Data Analysis

dataset = pd.read_csv("diabetes.csv")
dataset.head()


# In[8]:


dataset.shape


# In[9]:


#Target'ın Analizi
dataset["Outcome"].value_counts()


# In[10]:


sns.countplot(x="Outcome", data=dataset)
plt.show()


# In[11]:


100 * dataset["Outcome"].value_counts() / len(dataset)


# In[12]:


#Feature'ların Analizi

dataset.describe().T


# In[13]:


dataset["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()


# In[14]:


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


# In[15]:


for col in dataset.columns:
    plot_numerical_col(dataset, col)
    
cols = [col for col in dataset.columns if "Outcome" not in col]


# In[16]:


#Target vs Features
dataset.groupby("Outcome").agg({"Pregnancies": "mean"})


# In[17]:


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataset.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
    
for col in cols:
    target_summary_with_num(dataset, "Outcome", col)


# In[18]:


#Data Preprocessing  (Veri Ön İşleme)
dataset.isnull().sum()


# In[19]:


dataset.describe().T


# In[20]:


for col in cols:
    print(col, check_outlier(dataset, col))

replace_with_thresholds(dataset, "Insulin")


# In[21]:


for col in cols:
    dataset[col] = RobustScaler().fit_transform(dataset[[col]])


# In[22]:


dataset.head()


# In[23]:


#Model & Prediction

y = dataset["Outcome"]

X = dataset.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_


# In[24]:


log_model.coef_


# In[25]:


y_pred = log_model.predict(X)

y_pred[0:10]


# In[26]:


y[0:10]


# In[27]:


#Model Evaluation

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm =confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
    
plot_confusion_matrix(y, y_pred)
    


# In[28]:


print(classification_report(y, y_pred))


# In[29]:


#ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


# In[34]:


#Model Validation: Holdout

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)


# In[35]:


log_model = LogisticRegression().fit(X_train, y_train)


# In[38]:


y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]


# In[39]:


print(classification_report(y_test, y_pred))


# In[48]:


RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# In[46]:


#AUC
roc_auc_score(y_test, y_prob)


# In[52]:


#Model Validation: 10-Fold Cross Validation

y = dataset["Outcome"]
X = dataset.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)


# In[53]:


cv_results = cross_validate(log_model,
                             X, y,
                             cv=5,
                             scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


# In[56]:


cv_results['test_accuracy'].mean()


# In[57]:


cv_results['test_precision'].mean()


# In[58]:


cv_results['test_recall'].mean()


# In[59]:


cv_results['test_f1'].mean()


# In[60]:


cv_results['test_roc_auc'].mean()


# In[61]:


#Prediction for A New Observation

X.columns


# In[62]:


random_user = X.sample(1, random_state=45)
log_model.predict(random_user)


# In[ ]:




