#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
from math import pi
import matplotlib.pyplot as plt
import numpy.linalg as lg


# In[33]:


train_data= np.loadtxt('regTrain.txt')

X=train_data[:,0]
Y=train_data[:,1]


# In[34]:


def get_fi_Fourier(X,k):
    # print(X)
    lenX=len(X)
    fi=np.ones((lenX,1))
    for l in range(1,k+1):
        tmp=np.cos(2*pi*l*X)/l
        fi=np.append(fi,tmp.reshape(lenX,1),axis=1)
        tmp=np.sin(2*pi*l*X)/l
        fi=np.append(fi,tmp.reshape(lenX,1),axis=1)
    return fi


# In[62]:


def get_W(fi,Y):
    w=lg.inv(fi.T.dot(fi)).dot(fi.T).dot(Y)
    return w


# In[86]:


def get_predict(X,w,k):
    predict_y=0
    fi=get_fi_Fourier(X,k)
#     print(fi)
    predict_y=fi.dot(w)
    return predict_y


# In[91]:


X_test=np.arange(0,0.85,0.01)
# print(X_test)
for k in range(1,18,2):
    fi=get_fi_Fourier(X,k)
#     print("fi1=",fi[1,:])
    w=get_W(fi,Y)
#     print(w)
    predict_y=get_predict(X_test,w,k)
    plt.plot(X_test,predict_y,label="k=%s"%k)

plt.scatter(X,Y,label="train_data",marker='.',c="#87CEEB")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()




# In[83]:


fi=get_fi_Fourier(X_test,5)
fi[1]


# In[ ]:




