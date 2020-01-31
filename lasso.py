import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 

import matplotlib.pylab as plt

data = pd.read_csv('Cars93.csv')
Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])

#Normalización
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train=X[:50]
X_test=X[50:]
Y_train=Y[:50]
Y_test=Y[50:]
from itertools import combinations
m=np.arange(0,10)
b=[]
for i in range(10):
    comb=combinations(m,i+1)
    regresion = sklearn.linear_model.LinearRegression()
    a=[]
    for j in comb:
        regresion.fit(X_train[:,j], Y_train)
        a.append(regresion.score(X_test[:,j], Y_test))
    a=np.array(a)
    b.append(a)
c=[]
for i in b:
    c.append(np.max(i))
x=np.arange(1,len(c)+1)
for i in range(len(b)):
    a=np.ones(len(b[i]))*(i+1)
    plt.scatter(a,b[i])
plt.scatter(x,c,color='black',s=40,label='Maximos')
plt.xlabel('# de Variables')
plt.ylabel("$R²$")
plt.legend(loc=0.0)
plt.savefig('nparams.png')
plt.close()

#Lasso
x=np.logspace(-3,2,40)
y=[]
for i in x:
    lasso=sklearn.linear_model.Lasso(alpha=i)
    lasso.fit(X_scaled,Y)
    y.append(lasso.score(X_test,Y_test))
plt.scatter(np.log10(x),y)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.savefig('lasso.png')
plt.close()