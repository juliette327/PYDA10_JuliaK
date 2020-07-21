#!/usr/bin/env python
# coding: utf-8

# # Домашняя работа №1

# In[27]:


import numpy as np
from matplotlib import pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 10,7


# ## Задание 1

# Изобразите с помощью matplotlib. Изобразите точку x + 2y + 3z. Найдите угол между векторами x, y и x, z. 

# In[33]:


x = np.array([1,1])
y = np.array([2, 0])
z = np.array([0,2])


# In[20]:


m = np.array([x, y, z])
print(m)


# In[38]:


s = x + (2*y) + (3*z)
print(s)


# In[44]:


fig, ax = plt.subplots()
ax.scatter(s[0], s[1])
plt.show()


# In[58]:


x = np.array([1,1])
y = np.array([2, 0])
z = np.array([0,2])

ax = plt.axes()

plt.xlim( [0, 3] )
plt.ylim( [0, 3] )

ax.arrow( 0, 0, x[0], x[1], head_width=0.1, head_length=0.2, fc='k', ec='k' )
ax.arrow( 0, 0, y[0], y[1], head_width=0.1, head_length=0.2, fc='k', ec='k' )
ax.arrow( 0, 0, z[0], z[1], head_width=0.1, head_length=0.2, fc='k', ec='k' )

plt.show()


# In[59]:


def cosine( x, y ):
    xLength = np.linalg.norm( x )
    yLength = np.linalg.norm( y )
    
    return np.dot( x, y ) / ( xLength * yLength )
print(cosine( x, y ))


# In[60]:


def cosine( x, z ):
    xLength = np.linalg.norm( x )
    zLength = np.linalg.norm( z )
    
    return np.dot( x, z ) / ( xLength * zLength )
print(cosine( x, z ))


# ## Задание 2

# In[ ]:





# Найдите собственные значения и собственные вектора матриц (необходимо решение на numpy и решение по алгоритму на бумажке). Для матрицы 3x3 можно посмотреть на корни характеристического многочлена, посчитанные в numpy.

# In[9]:


m1 = np.array([[2, 2],
               [1, 3]])

m2 = np.array([[4, 1, -1],
               [1, 4, -1],
               [-1, -1, 4]])


# In[10]:


w1, v1 = np.linalg.eig(m1)
w2, v2 = np.linalg.eig(m2)


# In[6]:


for i in range(len(w1)):
    print("Собственное значение " + str(w1[i]))
    print("соответствующий ему собственный вектор " + str(v1[:,i]))


# In[8]:


for j in range(len(w2)):
    print("Собственное значение " + str(w2[j]))
    print("соответствующий ему собственный вектор " + str(v2[:,j]))


# ## Задание 3

# Найдите сингулярное разложение матрицы.

# In[12]:


m3 = np.array([[1, 2],
               [2, 3]])


# In[14]:


U, s, V = np.linalg.svd(m3)
n_component = 1
S = np.diag(s)[:, :n_component]
V = V[:n_component, :]
A = U.dot(S.dot(V))
print(A)


# In[ ]:




