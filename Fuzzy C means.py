import numpy as np
import math as mat
import random as rand
import matplotlib.pyplot as p
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

r = 5
center = [[1, 1], [-1, 1], [1, -1]]
X, ol = make_blobs(n_samples=r, centers=center,
cluster_std = 0.4, random_state=0)
X = StandardScaler().fit_transform(X)

#Defination of colors for the output
def pc(lst):
    cols = []
    for l in lst:
        if l == 1:
            cols.append('red')
        elif l == 2:
            cols.append('blue')
        elif l == 0:
            cols.append('#5ac18e')
        elif l == -1:
            cols.append('black')
        else:
            cols.append('yellow')
    return cols
    
    
#Recalculation of Centers 
def cen(u, d, n):
    c = np.empty((0,2))
    e = np.zeros((2))
    for i in range(n):
        s = 0
        sx = 0
        sy = 0
        for j in range(r):
            s += u[i][j]**2
            sx += (u[i][j]**2)*d[j][0]
            sy += (u[i][j]**2)*d[j][1]
        e[0]=sx/s
        e[1]=sy/s
        c=np.array(np.append(c, np.array([e]), axis=0))
    return(c)

#Calculating the euclidian distance
def dis(d, c, n):
    g = np.zeros((r,n))
    for i in range(r):
        for j in range(n):
            g[i][j] = mat.dist(d[i],c[j])
    return(g)

#Calculation of the fuzzy relationship matrix
def fuz(d, f, n):
    g = np.zeros((r,n))
    for k in range(r):
        for i in range(n-1):
            s = 1
            for j in range(n-i-1):
                s += (d[k][i+j]/d[k][i+j+1])**2
            g[k][i]=1/s
    for t in range(r):
        sum = 0
        for b in range(n-1):
            sum += g[t][b]
        g[t][n-1]= 1-sum
    w = np.zeros((n,r))
    for y in range(n):
        for h in range(r):
            w[y][h]=g[h][y]
    return(w)



def Fuzzy(n):
    u = np.zeros((n, r))
    for i in range(r):
        u[rand.randrange(n)][i] = 1
    c = cen(u, X, n)
    d = dis(X, c, n)
    u1 = fuz(d, 2, n)
    c1 = cen(u1, X, n)
    y = np.subtract(u, u1)
    for a in range(n):
        for b in range(r):
            if(y[a][b]<0):
                y[a][b] = -y[a][b]
    # y = np.argsort(y)
    print(y)
    while(np.array_equal(c,c1)==False):
        c = c1
        d = dis(X, c1, n)
        u = fuz(d, 2, n)
        c1 = cen(u, X, n)

    if(np.array_equal(c,c1)==True):
        print(u)
        print( "convergence achived ")
    return c, u


col = pc(ol)
_, ax = p.subplots()
ax.scatter(X[:, 0], X[:, 1], s=4, c=col[:])
c, u = Fuzzy(3) #Calling the Fuzzy C-means funcion
#Displaying the output
_, a1 = p.subplots()
a1.scatter(X[:, 0], X[:, 1], c = u[0], cmap="Greens")
_, a2 = p.subplots()
a2.scatter(X[:, 0], X[:, 1], c = u[1], cmap="Blues")
_, a3 = p.subplots()
a3.scatter(X[:, 0], X[:, 1], c = u[2], cmap="Reds")
p.show()
