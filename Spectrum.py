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



# Замена базиса в матрице
def change(C):
    C1=tensordot(tensordot(pot.transpose(),C,axes=1),pot,axes=1)
    return C1



# Замена базиса в обратной матрице
def changeinv(C):
    C1=tensordot(tensordot(inv(pot),C,axes=1),inv(pot.transpose()),axes=1)
    return C1

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



def bidiag(C,L):
    """Привидение двух квадратичных форм к диаональному виду
    Схема такая:
    C-diag-E-E
    L-L'-L''-diag
    E-S-S-S''"""
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
    period = 0 для не периодического потенциала, 1 для периодического 
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




def bvector(i,degrees,dim,a,b):
    """
    Построение элемента базиса всей цепочки(используется для построения базиса всей цепочки)
    """
    fi=1
    for j in range (0,degrees):
            if(i==j):
                fi=kron(fi,linspace(a[j],b[j],int(dim[j])))
            else:
                fi=kron(fi,linspace(1,1,int(dim[j])))
    return fi



def basis(degrees,dim,a,b):
    """
    Построение базиса всей цепочки
    """
    N=1
    for i in range (0,degrees):
        N=int(N*dim[i])
    fi=np.zeros((N,degrees))
    for i in range (0,degrees):
        fi[:,i]=bvector(i,degrees,dim,a,b)
    return fi,N

def Runge_Kutta_coefs(pr = 1, M=1):
    """
    pr - какая производная приближается,
    M - до какого порядка (порядок по дефолту выбирается минимальным),
    
    Идея такая:нечетные производные расскладываются по разности функций +дельта и -дельта,
    четные - по сумме и не сдвинутой функции"""
    M = max(M, pr//2+1)
    b = np.asarray([0]*(pr//2)+[math.factorial(pr)/2]+[0]*(M-pr//2-1))
    n = np.linspace(1,M,M)
    A = []
    for i in range(0,M):
        if pr%2 == 1:
            A.append(n**(2*i+1))
        else:
            A.append((n-1)**(2*i))
    A=np.asarray(A)
    # так как расскладываются по сумме смещенных функций, то не смещенную надо поделить на два
    if pr%2 == 0:
        A[0,0] = 0.5
    return (np.linalg.solve(A,b),M)


def fillingA(amp,A,dimension,dx,mode='n**2',bound='not',axes=0,axesr=0, Runge_Kutta = None):
    """
    calculate matrix in flux basis for n^2, n*nr (where nr - charge for resonator), n or cos(phi) and cos(phi_1-phi_2) in charge basis, periodic and nonperiodic conditions,
    for different axeses and axes for resonator(axesr),
    n is -1j*d/dfi
    n**2 is -d2/dfi^2
    n*nr here is d/dfi*(a^+-a)
    n1*n2 here is -d2/dfi1dfi2
    cos(phi)psi(i)=(psi(i+1)+psi(i-1))/2
    cos(phi_1-phi_2)psi(i,j)=(psi(i+1,j-1)+psi(i-1,j+1))/2
    Runge_Kutta - какой порядок метода Рунге-Кутта используется, если None, то минимальный
    """
    if bound!='not' and bound!='periodic' :
        print("wrong bounds, bounds should be not or periodic")
        return None
    if(amp!=abs(amp) and amp!=-abs(amp) and mode!='cos(phi)' and mode!='cos(phi_1-phi_2)'):
        print('Warning: For flux basis amp should be real to make the hamiltonian hermitian')
    if(mode!='n**2'and mode!='n' and mode!='n*nr' and mode!='cos(phi)' and mode!='cos(phi_1-phi_2)' and mode!='n1*n2'):
        print('Wrong mode: mode should be n**2 or n or n*nr or cos(phi) or cos(phi_1-phi_2)')
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
        if Runge_Kutta is None:
            for j in range(0,nbl):
                for i in range(0,up-step):
                    A1[i+j*up,i+j*up+step]+=-1j*amp/dx[axes]/2
                    A1[i+j*up+step,i+j*up]+=1j*amp/dx[axes]/2
            if(bound=='periodic'):
                for j in range(0,nbl):
                    for i in range(up-step,up):
                        A1[i+j*up,i+j*up+step-up]+=-1j*amp/dx[axes]/2
                        A1[i+j*up+step-up,i+j*up]+=1j*amp/dx[axes]/2
        else:
            (popravka,M) = Runge_Kutta_coefs(M = Runge_Kutta)
            for R in range(1,M+1):
                for j in range(0,nbl):
                    for i in range(0,up-step*R):
                        A1[i+j*up,i+j*up+step*R]+=-1j*amp/dx[axes]/2*popravka[R-1]
                        A1[i+j*up+step*R,i+j*up]+=1j*amp/dx[axes]/2*popravka[R-1]
                if(bound=='periodic'):
                    for j in range(0,nbl):
                        for i in range(up-step*R,up):
                            A1[i+j*up,i+j*up+step*R-up]+=-1j*amp/dx[axes]/2*popravka[R-1]
                            A1[i+j*up+step*R-up,i+j*up]+=1j*amp/dx[axes]/2*popravka[R-1]
                        
    # n**2 is -d2/dfi^2
    if(mode=='n**2'):
        for i in range(-(dimension.shape[0]-1),-axes):
            step=int(step*dimension[-i])
            up=int(up*dimension[-i-1])
        # nbl - number of these blocks
        nbl=A1.shape[0]//up
        if Runge_Kutta is None:
            for i in range(0,A1.shape[0]):
                A1[i,i]+=2*amp/dx[axes]**2
            for j in range(0,nbl):
                for i in range(0,up-step):
                    A1[i+j*up,i+j*up+step]+=-amp/dx[axes]**2
                    A1[i+j*up+step,i+j*up]+=-amp/dx[axes]**2
            if(bound=='periodic'):
                for j in range(0,nbl):
                    for i in range(up-step,up):
                        A1[i+j*up,i+j*up+step-up]+=-amp/dx[axes]**2
                        A1[i+j*up+step-up,i+j*up]+=-amp/dx[axes]**2
        else:
            (popravka,M) = Runge_Kutta_coefs(pr = 2, M = Runge_Kutta)
            for i in range(0,A1.shape[0]):
                A1[i,i]+=amp/dx[axes]**2*popravka[0]
            for R in range(1,M):
                for j in range(0,nbl):
                    for i in range(0,up-step*R):
                        A1[i+j*up,i+j*up+step*R]+=-amp/dx[axes]**2*popravka[R]
                        A1[i+j*up+step*R,i+j*up]+=-amp/dx[axes]**2*popravka[R]
                if(bound=='periodic'):
                    for j in range(0,nbl):
                        for i in range(up-step*R,up):
                            A1[i+j*up,i+j*up+step*R-up]+=-amp/dx[axes]**2*popravka[R]
                            A1[i+j*up+step*R-up,i+j*up]+=-amp/dx[axes]**2*popravka[R]
                    
    # n*nr here is d/dfi*(a^+-a)
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
                        A1[j*up+ir+jr*upr,j*up+step+ir+jr*upr+stepr]+=-amp/dx[axes]*(ir//stepr+1)**0.5/2
                        A1[j*up+step+ir+jr*upr+stepr,j*up+ir+jr*upr]+=-amp/dx[axes]*(ir//stepr+1)**0.5/2
                        A1[j*up+step+ir+jr*upr,j*up+ir+jr*upr+stepr]+=amp/dx[axes]*(ir//stepr+1)**0.5/2
                        A1[j*up+ir+jr*upr+stepr,j*up+step+ir+jr*upr]+=amp/dx[axes]*(ir//stepr+1)**0.5/2
        if (axesr<axes):
            # nbl - number of these blocks
            nblr=A1.shape[0]//upr
            nbl=upr//up
            for jr in range(0,nblr):
                for j in range(0,nbl-stepr//up):
                    for i in range(0,up-step):
                        A1[i+j*up+jr*upr,i+j*up+step+jr*upr+stepr]+=-amp/dx[axes]*((j*nbl)//stepr+1)**0.5/2
                        A1[i+j*up+step+jr*upr+stepr,i+j*up+jr*upr]+=-amp/dx[axes]*((j*nbl)//stepr+1)**0.5/2
                        A1[i+j*up+step+jr*upr,i+j*up+jr*upr+stepr]+=amp/dx[axes]*((j*nbl)//stepr+1)**0.5/2
                        A1[i+j*up+jr*upr+stepr,i+j*up+step+jr*upr]+=amp/dx[axes]*((j*nbl)//stepr+1)**0.5/2
    # n1*n2 here is -d2/dfi1dfi2
    if (mode == 'n1*n2'):
        if (axes == axesr):
            print("axes and axesr must be different")
            return None
        if (axesr < axes):
            axes1 = axes
            axes = axesr
            axesr = axes1
        for i in range(-(dimension.shape[0] - 1), -axes):
            step = int(step * dimension[-i])
            up = int(up * dimension[-i - 1])
        # stepr to describe displacemant of axesr's coordinate
        stepr = 1
        # upr - upper bound of my block in matrix for axesr's mode
        upr = int(dimension[dimension.shape[0] - 1])
        for i in range(-(dimension.shape[0] - 1), -axesr):
            stepr = int(stepr * dimension[-i])
            upr = int(upr * dimension[-i - 1])
        if (axes < axesr):
            # nbl - number of these blocks
            nbl = A1.shape[0] // up
            nblr = up // upr
            for j in range(0, nbl):
                for jr in range(0, nblr - step // upr):
                    for ir in range(0, upr - stepr):
                        A1[j * up + ir + jr * upr, j * up + step + ir + jr * upr + stepr] += -amp / dx[axes] / dx[axesr] / 4
                        A1[j * up + step + ir + jr * upr + stepr, j * up + ir + jr * upr] += -amp / dx[axes] / dx[axesr] / 4
                        A1[j * up + step + ir + jr * upr, j * up + ir + jr * upr + stepr] += amp / dx[axes] / dx[axesr] / 4
                        A1[j * up + ir + jr * upr + stepr, j * up + step + ir + jr * upr] += amp / dx[axes] / dx[axesr] / 4
            if(bound=='periodic'):
                for j in range(0, nbl):
                    for jr in range(nblr - step // upr, nblr):
                        for ir in range(0, upr - stepr):
                            A1[j * up + ir + jr * upr, j * up + step - up + ir + jr * upr + stepr] += -amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + step - up + ir + jr * upr + stepr, j * up + ir + jr * upr] += -amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + step - up + ir + jr * upr, j * up + ir + jr * upr + stepr] += amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + ir + jr * upr + stepr, j * up + step - up + ir + jr * upr] += amp / dx[axes] / dx[
                                axesr] / 4
                        for ir in range(upr - stepr, upr):
                            A1[j * up + ir + jr * upr, j * up + step - up + ir + jr * upr + stepr - upr] += -amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + step - up + ir + jr * upr + stepr - upr, j * up + ir + jr * upr] += -amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + step - up + ir + jr * upr, j * up + ir + jr * upr + stepr - upr] += amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + ir + jr * upr + stepr - upr, j * up + step - up + ir + jr * upr] += amp / dx[axes] / dx[
                                axesr] / 4
                    for jr in range(0, nblr - step // upr):
                        for ir in range(upr - stepr, upr):
                            A1[j * up + ir + jr * upr, j * up + step + ir + jr * upr + stepr - upr] += -amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + step + ir + jr * upr + stepr - upr, j * up + ir + jr * upr] += -amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + step + ir + jr * upr, j * up + ir + jr * upr + stepr - upr] += amp / dx[axes] / dx[
                                axesr] / 4
                            A1[j * up + ir + jr * upr + stepr - upr, j * up + step + ir + jr * upr] += amp / dx[axes] / dx[
                                axesr] / 4
    # cos(phi) psi(i)=(psi(i+1)+psi(i-1))/2
    if (mode == 'cos(phi)'):
        if dx is not None:
            print("Warning: It's tunelling operator in charge basis. There is no dx")
        for i in range(-(dimension.shape[0] - 1), -axes):
            step = int(step * dimension[-i])
            up = int(up * dimension[-i - 1])
        # nbl - number of these blocks
        nbl = A1.shape[0] // up
        for j in range(0, nbl):
            for i in range(0, up - step):
                A1[i + j * up, i + j * up + step] += 0.5 * amp
                A1[i + j * up + step, i + j * up] += 0.5 * amp.conjugate()
    # cos(phi_1-phi_2) psi(i,j)=(psi(i+1,j-1)+psi(i-1,j+1))/2
    if (mode == 'cos(phi_1-phi_2)'):
        if dx is not None:
            print("Warning: It's tunelling operator in charge basis. There is no dx")
        if (axesr < axes):
            axes1 = axes
            axes = axesr
            axesr = axes1
        for i in range(-(dimension.shape[0] - 1), -axes):
            step = int(step * dimension[-i])
            up = int(up * dimension[-i - 1])
        # stepr to describe displacemant of axesr's coordinate
        stepr = 1
        # upr - upper bound of my block in matrix for resonator
        upr = int(dimension[dimension.shape[0] - 1])
        for i in range(-(dimension.shape[0] - 1), -axesr):
            stepr = int(stepr * dimension[-i])
            upr = int(upr * dimension[-i - 1])
        if (axes == axesr):
            print("axes and axesr must be different")
            return None
        if (axes < axesr):
            # nbl - number of these blocks
            nbl = A1.shape[0] // up
            nblr = up // upr
            for j in range(0, nbl):
                for jr in range(0, nblr - step // upr):
                    for ir in range(0, upr - stepr):
                        A1[j * up + ir + jr * upr + step, j * up + ir + jr * upr + stepr] += amp / 2
                        A1[j * up + ir + jr * upr + stepr, j * up + ir + jr * upr + step] += amp.conjugate() / 2
    return A1

