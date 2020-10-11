import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def y(x):
     return 10+x**2 -15*np.cos(5*x)

search_space = np.arange(-10,10,0.01)
xs=list(y(i)for i in search_space)
n_particulas = 50
x1p = np.random.random(n_particulas)+8
x1pL=x1p
velocidad_x1=np.zeros(n_particulas)
x1_pg = 0
fx_pg=10000
fx_pL=np.ones(n_particulas)*fx_pg
c1, c2= 0.75,0.75

for i in range(0,100):
     fx = pd.DataFrame((y(i) for i in x1p))
     [val,idx]=fx.min(),fx.idxmin()

     if val.values<float(fx_pg):
          fx_pg = val
          x1_pg = x1p[idx]

     for j in range(0,n_particulas):
          if fx.iloc[j].values<fx_pL[j]:
               fx_pL[j]=fx.iloc[j].values
               x1pL=x1p[j]

     velocidad_x1 = velocidad_x1+c1*np.random.rand()*(x1_pg-x1p) + c2*np.random.rand()*(x1pL-x1p)
     x1p= x1p+velocidad_x1

plt.plot(search_space,xs)
plt.plot(x1_pg,fx_pg,'*')
plt.show()