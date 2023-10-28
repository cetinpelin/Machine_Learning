#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


# In[7]:


dataset = pd.read_csv("advertising.csv")
dataset.head()


# In[9]:


X = df[["TV"]]
y = df[["sales"]]


# In[10]:


reg_model = LinearRegression().fit(X, y)


# In[20]:


#y_hat = b +w*TV
# sabit (b -bias)
reg_model.intercept_[0]


# In[21]:


#tv'nin katsayısı (w1)
reg_model.coef_[0][0]


# In[22]:


reg_model.intercept_[0] + reg_model.coef_[0][0]*150


# In[23]:


#Modelin Görselleştirilmesi

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's':9},
               ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


# In[26]:


#Tahmin başarısı
#MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)


# In[28]:


y.mean()


# In[29]:


y.std()


# In[30]:


#RMSE
np.sqrt(mean_squared_error(y, y_pred))


# In[32]:


#MAE
mean_absolute_error(y, y_pred)


# In[33]:


#R-square
reg_model.score(X, y)


# In[37]:


X = df.drop('sales', axis=1)
y = df[["sales"]]


# In[44]:


#Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# In[45]:


reg_model = LinearRegression().fit(X_train, y_train)


# In[48]:


#Tahmin Başarısı Değerlendirme
#Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))


# In[49]:


#Train Rkare
reg_model.score(X_train, y_train)


# In[50]:


#Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[51]:


#Test Rkare
reg_model.score(X_test, y_test)


# In[52]:


#10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                X,
                                y,
                                cv=10,
                                scoring="neg_mean_squared_error")))


# In[53]:


#Simple Linear Regression with Gradient Descent from Scratch
#Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
        
    mse = sse / m
    return mse


# In[54]:


#update_weights
def update_weights(Y, b, w, X, learning_rate):
    
    m = len(Y)
    
    b_deriv_sum = 0
    w_deriv_sum = 0
    
    for i in range(0, m):
        y_hat = b + w * X[i]
        y= Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
        
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    
    return new_b, new_w
        
        


# In[55]:


#Train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    
    cost_history = []
    
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        
        if i % 100 ==0:
            print("iter={:d}  b={:.2f}  w={:.4f}  mse={:.4}".format(i, b, w, mse))
            
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    
    return cost_history, b, w


# In[56]:


dataset = pd.read_csv("advertising.csv")


# In[57]:


X = df["radio"]
Y = df["sales"]


# In[58]:


#hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000


# In[60]:


cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)


# In[ ]:




