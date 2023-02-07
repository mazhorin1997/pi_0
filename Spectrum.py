import matplotlib.pyplot as plt
import numpy as np
import math as math
import scipy
from scipy.sparse.linalg import eigs
from numpy.linalg import inv,norm,eig
from numpy import linspace,cos,shape,tensordot,einsum,diag,kron
from math import pi
from copy import copy
import itertools


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
    #if(amp!=abs(amp) and amp!=-abs(amp) and mode!='cos(phi)' and mode!='cos(phi_1-phi_2)'):
        #print('Warning: For flux basis amp should be real to make the hamiltonian hermitian')
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
                        A1[i+j*up,i+j*up+step*R]+=-1j*amp/dx[axes]*popravka[R-1]
                        A1[i+j*up+step*R,i+j*up]+=1j*amp/dx[axes]*popravka[R-1]
                if(bound=='periodic'):
                    for j in range(0,nbl):
                        for i in range(up-step*R,up):
                            A1[i+j*up,i+j*up+step*R-up]+=-1j*amp/dx[axes]*popravka[R-1]
                            A1[i+j*up+step*R-up,i+j*up]+=1j*amp/dx[axes]*popravka[R-1]
                        
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
                A1[i,i]+=-amp/dx[axes]**2*popravka[0]
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

def make_state(desired,shapes):
    """
    Делает желаемое (desired) состояние общей системы из собственных состояний подсистем c размерностями shapes.
    desired - список из номеров состояний каждой подсистем,
    shapes - размерности каждой из подсистем.
    """
    f=1
    for i in range(0,len(shapes)):
        f1=np.linspace(0,0,shapes[i])
        f1[desired[i]]=1
        f=np.kron(f,f1)
    return f

def find_proper_levels(f_all,f):
    """
    Находит нужный собственный уровень системы по наиболее похожей собственной функции на заданную в аргумента f.
    f_all - все собственные уровни системы,
    f - желаемая функция.
    """
    a=0
    j=0
    for i in range(0,f_all.shape[1]):
        b=abs(f_all[:,i]@f)
        if b>a:
            a=b
            j=i
    return j

def transmon(Ec, Ej1, Ej2, N_max=30, z = np.linspace(-np.pi,0,2), r=7):
    """
        Расчет спектра трансмона в зарядовом базисе в зависимости от внешнего потока,
        Ec, Ej1, Ej2 - энергетические параметры системы (Гамильтониан системы надо понятно записать: H=Ec*n**2/2-Ej1*cos(phi)-Ej2*sin(phi))
        N_max - максимальный заряд
        z - внешний поток
        r - номер максимального уровня в итоговом спектре
        Выходные данные : (B,fm2,fi[:,0],h)
        B - энергии в зависимости от внешнего потока (последний индекс)
        fm2 - волновые функции в зависимости от внешнего потока (первый индекс - координата, второй - номер уровня, третий - внешний поток),
        fi[:,0] - базисный вектор
        h - шаг по сетке
        Зависимость от внешнего потока в последнем индексе"""
    degrees=1
    period=[0]
    dim=np.asarray([N_max*2+1])
    b=np.asarray([-N_max])
    a=np.asarray([N_max])
    (h,b)=dividing(degrees,a,b,period,dim)
    (fi,N)=basis(degrees,dim,a,b)
    m=z.shape[0]
    B=np.zeros((r,m))
    fm2=np.tensordot(linspace(0j,0,N),tensordot(linspace(0,0,r),linspace(0,0,m),axes=0),axes=0)
    A0=np.tensordot(linspace(0j,0,N),linspace(0,0,N),axes=0)
    for i in range(0,N):
        A0[i,i]+=Ec*fi[i,0]**2/2
    for k in range(0,m):
        F=z[k]
        A=copy(A0)
        A=fillingA(-(Ej1+Ej2)*np.cos(F/2),A,dim,1,axes=0,mode='cos(phi)',bound='not')
        A=fillingA(-1j*(Ej1-Ej2)*np.sin(F/2),A,dim,1,axes=0,mode='cos(phi)',bound='not')
        (B2,f)= scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(A),k=r,which='SA',maxiter=4000)
        l_order=np.argsort(np.real(B2))
        B2=B2[l_order]
        f=f[:,l_order]
        B[:,k]=B2
        fm2[:,:,k]=f
    return (B,fm2,fi[:,0],h)

def fluxonium(El, Ec, Ej,dim=100,z=np.linspace(-np.pi,-np.pi,1),a=-6*np.pi,r=3,Runge_Kutta = 5):
    """
    Расчет спектра флаксониума в зависимости от внешнего потока.
    El, Ec, Ej - энергетические параметры системы (Гамильтониан системы: H = 0.5*El*(phi-phi0)**2+0.5*Ec*n**2+Ej*(1-cos(phi)))
    dim - размерность
    z - внешний поток
    а - граница области
    r - номер максимального уровня
    Runge_Kutta - порядок аппроксимации производных
    Выходные данные : (B,fm2,fi[:,0],h,n,phi)
    B - энергии в зависимости от внешнего потока (последний индекс)
    fm2 - волновые функции в зависимости от внешнего потока (первый индекс - координата, второй - номер уровня, третий - внешний поток),
    fi[:,0] - базисный вектор
    h - шаг по сетке
    n - матричные элементы заряда в зависимости от внешнего потока (последний индекс)
    phi - матричные элементы потока в зависимости от внешнего потока (последний индекс)
    """
    # количество степеней свободы системы
    degrees=1
    # периодический ли потенциал (0 - нет, 1 - да)
    period=[0]
    # преобразование размерностей и границ в массив
    dim=np.asarray([dim])
    b=np.asarray([-a])
    a=np.asarray([a])
    # определение шага сетки и правой границы (для периодических граничных условий она немного сдвигается)
    (h,b)=dividing(degrees,a,b,period,dim)
    # Построение базиса
    (fi,N)=basis(degrees,dim,a,b)
    # Определение размерностей энергий и собственных функций и гамильтониана
    m=z.shape[0]
    B=np.zeros((r,m))
    fm2=np.tensordot(linspace(0,0,N),tensordot(linspace(0,0,r),linspace(0,0,m),axes=0),axes=0)
    # Построение неизменной для внешнего потока части гамильтониана
    A0=np.tensordot(linspace(0,0,N),linspace(0,0,N),axes=0)
    A0=fillingA(Ec/2,A0,dim,h,axes=0,Runge_Kutta = Runge_Kutta)
    for i in range(0,N):
        A0[i,i]+=Ej*(1-np.cos(fi[i,0]))
    # Цикл по внешнему потоку
    for k in range(0,m):
        F=z[k]
        A=copy(A0)
        # Построение зависимой от внешнего потока части гамильтониана
        for i in range(0,N):
            A[i,i]+=El*(fi[i,0]-F)**2/2
        # Диагонализация Гамильтониана
        (B2,f)= scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(A),k=r,which='SA',maxiter=4000)
        # Его сортировка
        l_order=np.argsort(np.real(B2))
        B2=B2[l_order]
        f=f[:,l_order]
        B[:,k]=B2
        fm2[:,:,k]=f
    n = charge_elements(fm2,dim[0],h,Runge_Kutta = Runge_Kutta)
    phi = flux_elements(fm2,fi[:,0])
    return (B,fm2,fi[:,0],h,n,phi)



def charge_elements(f,dim,h,Runge_Kutta = 5):
    """
    Расчет матричных элементов заряда в потоковом базисе для одномерной задачи (если базис зарядовый, то будут рассчитаны потоковые элементы),
    f - волновые функции,
    dim - размерность системы,
    h - шаг сетки,
    Runge_Kutta - порядок аппроксимации производной"""
    A0=np.diag(np.linspace(0j,0,dim))
    A0=fillingA(1,A0,np.asarray([dim]),h,mode='n',axes=0,Runge_Kutta = Runge_Kutta)
    return np.einsum('imk,il,ljk->mjk',f,A0,f.conjugate())

def flux_elements(f,fi):
    """
    Расчет матричных элементов потока в потоковом базисе для одномерной задачи (если базис зарядовый, то будут рассчитаны зарядовые элементы),
    f - волновые функции,
    fi - базисный вектор"""
    return np.einsum('imk,il,ljk->mjk',f,np.diag(fi),f.conjugate())

def delete_global_phase(fm2):
    """
    Делает так, чтобы не было скачков глобальной фазы при изменении внешнего потока у собственных функций
     fm2 - собственные функции, зависящие от внешнего потока.
     Первый индекс - номер точки разбиения по обобщенной координате, второй индекс - номер уровня, третий - номер потока"""
    f=copy(fm2)
    for lvl in range(0,fm2.shape[1]):
        for fl in range(1,fm2.shape[2]):
            if f[:,lvl,fl]@f[:,lvl,fl-1]<0:
                f[:,lvl,fl]=-f[:,lvl,fl]
    return f

def dispersive_shift(B,n,n_r,B_r,Ec2,r=5):
    """
    Рассчитывает общий спектр двух систем в зависимости от одного параметра. От параметра зависит только первая система
    B - спектр первой системы, зависимость от параметра - его вторая ось
    n - матричные элементы первой системы,
    n_r - матричные элементы второй системы,
    B_r - спектр второй системы,
    Ec2 - коэффициент связи (входит в гамильтониан как коэффициент перед произведением матричных элементов двух систем)
    """
    B_disp=[]
    f_disp=[]
    for i in range(B.shape[1]):
        E=np.diag(np.linspace(1,1,B.shape[0]))
        E_r=np.diag(np.linspace(1,1,B_r.shape[0]))
        Energies=[]
        f_all=[]
        H_q=np.diag(B[:,i]-B[0,i])
        H_r=np.diag(B_r[:]-B_r[0])
        H1=np.kron(H_q,E_r)
        H2=np.kron(E,H_r)
        n1=np.kron(n[:,:,i],E_r)
        n2=np.kron(E,n_r)
        H=H1+H2+Ec2*n1@n2
        (B2,f)= scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(H),k=r,which='SA',maxiter=4000)
        l_order=np.argsort(np.real(B2))
        f=f[:,l_order]
        B_disp.append(B2[l_order])
        f_disp.append(f)
    return (np.asarray(B_disp),np.asarray(f_disp))

def disp_shift_plot(w,C,Cc,B,n,r_r,z,r=10):
    """
    Рассчитывает и строит дисперсионный сдвиг резонатора,
    Параметры:
    w - частота резонатора,
    С - емкость кубита
    Сс - емкость связи
    B -  спектр кубита в зависимости от внешнего потока
    n - оператор числа куперовских пар (матричнй вид)
    r_r - количество уровней учитываемых в резонаторе
    z - массив внешних потоков
    r -  число уровней кубита, учитываемых в расчете
    """
    g=make_g(w,Cc,C-Cc)
    B_r=np.linspace(0,(r_r-1)*w,r_r)
    n_r=-1j*(np.diag(np.linspace(1,(r_r-1),(r_r-1))**0.5,k=1)-np.diag(np.linspace(1,(r_r-1),(r_r-1))**0.5,k=-1))
    B_disp,f_disp=dispersive_shift(B[:r,:],n[:r,:r,:],n_r,B_r,g,r=10)
    fr=[]
    fr_1=[]
    for i in range(0,f_disp.shape[0]):
        fr.append(B_disp[i,find_proper_levels(f_disp[i],make_state([0,1],[r,r_r]))]-B_disp[i,0])
        fr_1.append(B_disp[i,find_proper_levels(f_disp[i],make_state([1,1],[r,r_r]))]-
                    B_disp[i,find_proper_levels(f_disp[i],make_state([1,0],[r,r_r]))])
    fr=np.asarray(fr)
    fr_1=np.asarray(fr_1)
    print((fr_1-fr)[0]*1e3)
    fig=plt.figure(figsize=(8,4))
    size=13
    plt.tick_params(labelsize = size)
    plt.plot(z/2/np.pi,(fr_1-fr)*1e3)
    plt.xlabel(r'$\Phi^{ext}/\Phi_0$',fontsize=size)
    plt.ylabel(r'$\chi_{01}$, МГЦ',fontsize=size)
    plt.title(r'Дисперсионный сдвиг')
    plt.tight_layout()

def multi_kron(j, B, dtype = 'complex128'):
    """
    Вычисляет тензорное произведение целевой двумерной матрицы с единичными.
    Единичные матрицы берутся той же размерности, что и из в списке, а целевая ставится на место j
    B - список матриц
    j - номер целевой матрицы в списке
    dtype - тип данных
    Выход:
    Полученная тензорным произведением матрица с нужным типом данных
    """
    result = np.diag([1])
    for i in range(len(B)):
        if i != j:
            E = np.diag(np.linspace(1,1,len(B[i])))
        else:
            if len(B[0].shape) == 1:
                E = np.diag(B[i]-B[i][0])
            else:
                E = copy(B[i])
        result = np.kron(result,E)
    return np.asarray(result,dtype = dtype)

def multikron_multimatrix(B):
    """
    Переводит все матрицы в одной тензорный базис (с помощью функции multi_kron)
    B - список матриц
    Выход:
    Список матриц в тензорном базисе
    """
    result=[]
    for i in range(len(B)):
        result.append(multi_kron(i,B))
    return result


def Unite_systems(B, n, g, t=None, r=20, mode = None):
    """
    Рассчитывает общий Гамильтониан системы
    B - список энергий отдельных систем
    n - список матричных элементов отдельных систем
    g - связи между системами (пример заполнения: [g01,g02,g12])
    t - список перестраиваемых элементов
    r -  номер последнего вычисляемого уровня энергии
    mode - влияет на выдачу, ниже написан стандартный выход,
        если mode == 'Hamiltonian', то выход: Гамильтониан общей системы
    Выход: (Energies, wavefunctions)
    Energies - собственные энергии
    wavefunctions - собственные функции
    """
    ### Проверка размерностей перестраиваемых элементов
    if t != None:
        check = B[t[0]].shape[1]
        for i in range(1, len(t)):
            if B[t[i]].shape[1] != check:
                print("ERROR: Не совпадают сетки по внешнему потоку для перестраиваемых элементов")
                return None
    else:
        check = 1
        t = [0]
        B[0] = np.tensordot(B[0], [1], axes=0)
        n[0] = np.tensordot(n[0], [1], axes=0)

    Energies = []
    wave_functions = []
    for i in range(B[t[0]].shape[1]):
        A = copy(B)
        M = copy(n)
        for j in range(len(t)):
            A[t[j]] = A[t[j]][:, i]
            M[t[j]] = M[t[j]][:, :, i]
        H_e = multikron_multimatrix(A)
        M_e = multikron_multimatrix(M)
        H = 0
        m = 0
        for j in range(len(H_e)):
            H += H_e[j]
            for k in range(j + 1, len(H_e)):
                H += g[m] * M_e[j] @ M_e[k]
                m += 1
        if mode == 'Hamiltonian':
            return H
        if r < H.shape[0] / 2:
            (C, f) = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(H), k=r, which='SA', maxiter=4000)
        else:
            (C, f) = np.linalg.eig(H)
        l_order = np.argsort(np.real(C))
        f = f[:, l_order]
        wave_functions.append(f)
        Energies.append(C[l_order])
    return (np.asarray(Energies), np.asarray(wave_functions))


def find_energy(Energies, f_all, f):
    """
    Выдает энергию состояния наиболее похожего на целевое состояние для различных значений параметра
        (например внешнего потока).
    Используется для определения состояний в заранее диагонализованных, а затем связанных систем.

    Energies - массив энергий (первый индекс - зависимость от параметра, второй - номер уровня)
    f_all - массив волновых функций (первый индекс - зависимость от параметра,
        второй - положение на сетке,третий - номер уровня)
    f - волновая функция целевого состояния
    Вывод: Energie, index
    Energie - массив с нужным уровенем в зависимости от параметра
    index - список индексов, соответствующих нужному уровню в Energies, для различных значений параметра
    """
    Energie = []
    index = []
    for i in range(0, Energies.shape[0]):
        j = find_proper_levels(f_all[i], f)
        Energie.append(Energies[i, j])
        index.append(j)
    return np.asarray(Energie), index


def find_all_energies(Energies, f_all, N_sys, state_max, n_lvls):
    """
    Выдает энергии первых состояний в зависимости от параметра (например, внешнего потока).
    Используется для определения состояний в заранее диагонализованных, а затем связанных систем.

    Energies - массив энергий (первый индекс - зависимость от параметра, второй - номер уровня),
        если нет зависимости от параметра, то массив одномерный.
    f_all - массив волновых функций (первый индекс - зависимость от параметра,
        второй - положение на сетке,третий - номер уровня),
        если нет зависимости от параметра, массив двумерный
    N_sys - количество связываемых систем
    state_max - список максимальных номеров уровней для различных подсистем,
        если число - то максимальные уровни одинаковы для каждой подсистемы
    n_lvls - список размерностей каждой подсистемы, если число, то одинаково для всех подсистем
    Вывод: (Energie,indexes)
    Energie - словарь с энергиями, зависящими от параметра. Ключи - номера уровней
        (например, для основного состояния трех связанных подсистем ключ такой :'0_0_0')
    indexes - словарь с индексами, соответствующим положению уровня в массиве Energies
        (формат ключей такой же, как для Energie)
    """
    if len(Energies.shape) == 2 and len(f_all.shape) == 3:
        New_Energies = copy(Energies)
        f_all_new = np.asarray(copy(f_all))
    elif len(Energies.shape) == 1 and len(f_all.shape) == 2:
        New_Energies = np.asarray([copy(Energies)])
        f_all_new = np.asarray([copy(f_all)])
    else:
        print("ERROR неправильные размерности Energies и f_all")
    if type(state_max) is int:
        state_max = [state_max] * N_sys
    if type(n_lvls) is int:
        n_lvls = [n_lvls] * N_sys
    if len(state_max) != N_sys or len(n_lvls) != N_sys:
        print("ERROR: Не совпадает количество подсистем и размернось state_max или n_lvls")
    states = list(
        map(lambda x: list(map(int, x)), list(itertools.product(*list(map(lambda x: list(np.linspace(0, x, x + 1)),
                                                                          state_max))))))
    desired_states = list(map(lambda x: make_state(x, n_lvls), states))
    Energie = {}
    indexes = {}
    keys = []
    for state in states:
        key = ''
        for i in range(len(state)):
            if i != 0:
                key += '_'
            key += str(state[i])
        keys.append(key)
        Energie[key] = []
        indexes[key] = []
    for i in range(0, New_Energies.shape[0]):
        for j in range(0, len(desired_states)):
            k = find_proper_levels(f_all_new[i], desired_states[j])
            f_all_new[i, :, k] *= 0
            indexes[keys[j]].append(k)
            Energie[keys[j]].append(New_Energies[i, k])
    for key in keys:
        Energie[key] = np.asarray(Energie[key])
    return (Energie, indexes)