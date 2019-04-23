import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

train_data= np.loadtxt('regTrain.txt')

X=train_data[:,0]
Y=train_data[:,1]

test_data=np.loadtxt('regTest.txt')

def get_fi_Fourier(X,k):
    # print(X)
    lenX=len(X)
    fi=np.ones((lenX,1))

    for l in range(1,k+1):
        tmp=np.cos(2*pi*l*X)/l
        fi=np.append(fi,tmp.reshape(lenX,1),axis=1)
        tmp=np.sin(2*pi*l*X)/l
        fi=np.append(fi,tmp.reshape(lenX,1),axis=1)

    print("tmp=",[tmp])
    return fi


KRR= KernelRidge(kernel='rbf',gamma=0.1)
print(X)
print(Y)
KRR.fit(X,Y)
Y_pred=KRR.predict(test_data[:,0])

plt.plot(test_data[:,0],Y_pred)


