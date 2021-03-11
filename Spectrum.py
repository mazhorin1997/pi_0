#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import math as math
import pickle
import scipy
from scipy.sparse.linalg import eigs
from numpy.linalg import inv,norm,eig
from numpy import linspace,cos,shape,tensordot,einsum,diag,kron
from math import pi
from copy import copy


# In[1]:


# Заполнение матрицы ёмкостей или индуктивностей
def filling(number):
    C=np.zeros((degrees,degrees))
    for i in range (0,number):
        if(end[i,0]!=0):
            C[int(end[i,0]-1),int(end[i,0]-1)]+=element[i]
        if(end[i,1]!=0):
            C[int(end[i,1]-1),int(end[i,1]-1)]+=element[i]
        if(end[i,0]*end[i,1]!=0):
            C[int(end[i,0]-1),int(end[i,1]-1)]+=-element[i]
            C[int(end[i,1]-1),int(end[i,0]-1)]+=-element[i]
    #print(C)
    return C


# In[2]:


# Замена базиса в матрице
def change(C):
    C1=tensordot(tensordot(pot.transpose(),C,axes=1),pot,axes=1)
    return C1


# In[4]:


# Замена базиса в обратной матрице
def changeinv(C):
    C1=tensordot(tensordot(inv(pot),C,axes=1),inv(pot.transpose()),axes=1)
    return C1


# In[5]:


# Определение джозефсоновской энерии
def Jsph(number):
    fj1=np.zeros((number,degrees))
    for i in range(0,number):
        if(end[i,0]>0):
            fj1[i,int(end[i,0])-1]=-1
        if(end[i,1]==0):
            print('ERROR')
            return 0
        else:
            fj1[i,int(end[i,1])-1]=1
    return fj1


# In[6]:


# Привидение двух квадратичных форм к диаональному виду
# Схема такая:
# C-diag-E-E
# L-L'-L''-diag
# E-S-S-S''
def bidiag(C,L):
    global pot
    S1=diag(linspace(1,1,degrees))
    (C1,pot)=eig(C)
    C1=diag(C1)
    L1=change(L)
    S1=tensordot(S1,pot,axes=1)
    pot=np.zeros((degrees,degrees))
    for i in range(0,degrees):
        pot[i,i]=C1[i,i]**(-0.5)
    C1=change(C1)
    L1=change(L1)
    S1=tensordot(S1,pot,axes=1)
    (L1,pot)=eig(L1)
    L1=diag(L1)
    C1=change(C1)
    S1=tensordot(S1,pot,axes=1)
    return (C1,L1,S1)






def dividing(degrees,a,b,period,dim):
    """Построение разбиения
    """
    b1=copy(b)
    h=np.zeros(degrees)
    for i in range(0,degrees):
        if(period[i]==0):
            h[i]=(b1[i]-a[i])/(dim[i]-1)
        else:
            h[i]=(b1[i]-a[i])/dim[i]
            b1[i]=b1[i]-h[i]
    return (h,b1)




# Нахождение осцилляторов в цепи и перестановка их в конец(надо находить до каких-либо смен базиса) Требуется знать количество джконтактов и их окончания
def findoscillator():
    global fj
    global C
    global L
    global pot
    osc=0  
    pot=np.zeros((degrees,degrees))
    pot=diag(linspace(1,1,degrees))
    for i in range(0,degrees):
        j=degrees-1-i#так удобней ибо осцилляторы переставляются в конец
        if(all(fj[:,j]==0)):
            osc+=1
            #Перестановка в конец осцилляторов(требует проверки и исправления)
            for i1 in range(0,degrees):
                for i2 in range(0,degrees):
                    if(pot[j,i1]==1 and pot[degrees-osc,i2]==1):
                        pot[j,i1]=0
                        pot[j,i2]=1
                        pot[degrees-osc,i1]=1
                        pot[degrees-osc,i2]=0            
            C=change(C)
            L=change(L)
            #(необходимо проверить)
            fj[:,j]=fj[:,degrees-osc]
            fj[:,degrees-osc]=0
    return osc


# In[125]:


# Построение элемента базиса всей цепочки(используется для построения базиса всей цепочки)
def bvector(i,degrees,dim,a,b):
    fi=1
    for j in range (0,degrees):
            if(i==j):
                fi=kron(fi,linspace(a[j],b[j],int(dim[j])))
            else:
                fi=kron(fi,linspace(1,1,int(dim[j])))
    return fi


# In[126]:


# Построение базиса всей цепочки 
def basis(degrees,dim,a,b):
    N=1
    for i in range (0,degrees):
        N=int(N*dim[i])
    fi=np.zeros((N,degrees))
    for i in range (0,degrees):
        fi[:,i]=bvector(i,degrees,dim,a,b)
    return fi,N


# In[372]:


def fillingA(amp,A,dimension,dx,mode='n**2',bound='not',axes=0,axesr=0):
    """
    calculate matrix in flux basis for n^2, n*nr (where nr - charge for resonator), n, periodic and nonperiodic conditions, 
    for different axeses and axes for resonator(axesr),
    """
    if bound!='not' and bound!='periodic' :
        print("wrong bounds, bounds should be not or periodic")
        return None
    if(amp!=abs(amp) and amp!=-abs(amp)):
        print('For amp should be real to make the hamiltonian hermitian')
        return None
    if(mode!='n**2'and mode!='n' and mode!='n*nr'):
        print('Wrong mode: mode should be n**2 or n or n*nr')
        return None
    A1=copy(A)
    # step to describe displacemant of axes's coordinate
    step=1
    # up - upper bound of my block in matrix
    up=int(dimension[dimension.shape[0]-1])
    if axes>dimension.shape[0]-1:
        print("axes is out of range")
    if axesr>dimension.shape[0]-1:
        print("axesr is out of range")
    # n is -1j*d/dfi 
    if(mode=='n'):
        for i in range(-(dimension.shape[0]-1),-axes):
            step=int(step*dimension[-i])
            up=int(up*dimension[-i-1])
        # nbl - number of these blocks
        nbl=A1.shape[0]//up
        for j in range(0,nbl):
            for i in range(0,up-step):
                A1[i+j*up,i+j*up+step]+=-1j*amp/dx[axes]
                A1[i+j*up+step,i+j*up]+=1j*amp/dx[axes]
        if(bound=='periodic'):
            for j in range(0,nbl):
                for i in range(up-step,up):
                    A1[i+j*up,i+j*up+step-up]+=-1j*amp/dx[axes]
                    A1[i+j*up+step-up,i+j*up]+=1j*amp/dx[axes]
    # n**2 is -d2/dfi^2
    if(mode=='n**2'):
        for i in range(0,A1.shape[0]):
            A1[i,i]+=2*amp/dx[axes]**2
        for i in range(-(dimension.shape[0]-1),-axes):
            step=int(step*dimension[-i])
            up=int(up*dimension[-i-1])
        # nbl - number of these blocks
        nbl=A1.shape[0]//up
        for j in range(0,nbl):
            for i in range(0,up-step):
                A1[i+j*up,i+j*up+step]+=-amp/dx[axes]**2
                A1[i+j*up+step,i+j*up]+=-amp/dx[axes]**2
        if(bound=='periodic'):
            for j in range(0,nbl):
                for i in range(up-step,up):
                    A1[i+j*up,i+j*up+step-up]+=-amp/dx[axes]**2
                    A1[i+j*up+step-up,i+j*up]+=-amp/dx[axes]**2
    # nr here is 1j(a^+-a)
    if(mode=='n*nr'):
        for i in range(-(dimension.shape[0]-1),-axes):
            step=int(step*dimension[-i])
            up=int(up*dimension[-i-1])
        # stepr to describe displacemant of axesr's coordinate
        stepr=1
        # upr - upper bound of my block in matrix for resonator
        upr=int(dimension[dimension.shape[0]-1])
        for i in range(-(dimension.shape[0]-1),-axesr):
            stepr=int(stepr*dimension[-i])
            upr=int(upr*dimension[-i-1])
        if (axes==axesr):
            print("axes and axesr must be different")
            return None
        if (axes<axesr):
            # nbl - number of these blocks
            nbl=A1.shape[0]//up
            nblr=up//upr
            for j in range(0,nbl):
                for jr in range(0,nblr-step//upr):
                    for ir in range(0,upr-stepr):
                        A1[j*up+ir+jr*upr,j*up+step+ir+jr*upr+stepr]+=-amp/dx[axes]*(ir//stepr+1)**0.5
                        A1[j*up+step+ir+jr*upr+stepr,j*up+ir+jr*upr]+=-amp/dx[axes]*(ir//stepr+1)**0.5
                        A1[j*up+step+ir+jr*upr,j*up+ir+jr*upr+stepr]+=amp/dx[axes]*(ir//stepr+1)**0.5
                        A1[j*up+ir+jr*upr+stepr,j*up+step+ir+jr*upr]+=amp/dx[axes]*(ir//stepr+1)**0.5
        if (axesr<axes):
            # nbl - number of these blocks
            nblr=A1.shape[0]//upr
            nbl=upr//up
            for jr in range(0,nblr):
                for j in range(0,nbl-stepr//up):
                    for i in range(0,up-step):
                        A1[i+j*up+jr*upr,i+j*up+step+jr*upr+stepr]+=-amp/dx[axes]*((j*nbl)//stepr+1)**0.5
                        A1[i+j*up+step+jr*upr+stepr,i+j*up+jr*upr]+=-amp/dx[axes]*((j*nbl)//stepr+1)**0.5
                        A1[i+j*up+step+jr*upr,i+j*up+jr*upr+stepr]+=amp/dx[axes]*((j*nbl)//stepr+1)**0.5
                        A1[i+j*up+jr*upr+stepr,i+j*up+step+jr*upr]+=amp/dx[axes]*((j*nbl)//stepr+1)**0.5
    return A1

