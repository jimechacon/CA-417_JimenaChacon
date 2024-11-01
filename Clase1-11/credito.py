
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import  scipy.optimize as spopt
from scipy.optimize import least_squares
from scipy.stats import norm

def probs(dh):
    # Cálculo de probabilidad de impago
    p1 = dh['Defaults1'].sum()/dh['Total1'].sum()
    p2 = dh['Defaults2'].sum()/dh['Total2'].sum()
    return {'p1':p1,'p2':p2}

# Objetivo: Calcular ECL, VaR y ES para esta cartera
# Cálculo de umbrales

def umbrales(prob):
    c = {key:norm.ppf(p) for key,p in prob.items()}

    return c

def momentos(dh):

    mm = dict()
    # Momentos para numero de impagos
    mm['meanK1'] = np.mean(dh['Defaults1'])
    mm['meanK2'] = np.mean(dh['Defaults2'])
    mm['mK21'] = np.mean(dh['Defaults1']**2)
    mm['mK22'] = np.mean(dh['Defaults2']**2)
    mm['mKK'] = np.mean(dh['Defaults1']*dh['Defaults2'])

    # Momentos para número total
    mm['meanN1'] = np.mean(dh['Total1'])
    mm['meanN2'] = np.mean(dh['Total2'])
    mm['mN21'] = np.mean(dh['Total1']**2)
    mm['mN22'] = np.mean(dh['Total2']**2)
    mm['mNN'] = np.mean(dh['Total1']*dh['Total2'])
    return mm

def FuncionCov1(r,p,mK2,mN2,meanN,c):
    rv = mvn(mean = [0,0],cov = [[1,r],[r,1]])
    return (mK2-meanN*p)/(mN2-meanN) - rv.cdf([c,c])

def FuncionCov2(r,mKK,mNN,c1,c2):
    rv = mvn(mean = [0,0],cov = [[1,r],[r,1]])
    return mKK/mNN - rv.cdf([c1,c2])

def calibra1(p,mK2,mN2,meanN,c):
    min = float(-.9)
    max = float(.9)
    r = spopt.bisect(lambda x: FuncionCov1(x,p,mK2,mN2,meanN,c),min,max)
    return(r)

def calibra2(mKK,mNN,c):
    min = float(-.9)
    max = float(.9)
    r = spopt.bisect(lambda x: FuncionCov2(x,mKK,mNN,c['p1'],c['p2']),min,max)
    return(r)

def covar_un_factor(x,r11,r22,r12):
    return np.array([x[0]**2-r11,x[1]**2-r22,x[0]*x[1]-r12])

def calibra_un_factor(r11,r22,r12):
    a10 = np.sqrt(r11)
    a20 = np.sqrt(r22)
    x0 = [a10,a20]
    print(x0)
    res = least_squares(lambda x: covar_un_factor(x,r11,r22,r12), x0)
    print(res.x)
    return(res.x)

def simula_credito(dc,a,c,nesc):

    ncreds = len(dc)
    Z = norm.rvs(size=nesc)
    xi = norm.rvs(size=(ncreds,nesc))

    data_cartera['a'] = 0
    data_cartera.loc[data_cartera['categorias']==1,'a'] = a[0]
    data_cartera.loc[data_cartera['categorias']==2,'a'] = a[1]
    data_cartera['b'] = np.sqrt(1-data_cartera['a']**2)
    data_cartera['c'] = 0
    data_cartera.loc[data_cartera['categorias']==1,'c'] = c['p1']
    data_cartera.loc[data_cartera['categorias']==2,'c'] = c['p2']

    data_cartera['X'] = 0
    data_cartera['Y'] = 0

    perd = list()

    for iesc in range(0,nesc):
        data_cartera['X'] = data_cartera['a']*Z[iesc]+data_cartera['b']*xi[:,iesc]
        data_cartera['Y'] = (data_cartera['X']<data_cartera['c'])
        perd.append((data_cartera['Y']*data_cartera['saldos']).sum())

    return perd

## TEST
if __name__ == "__main__":
    data_hist = pd.read_csv('/workspaces/CA-417_JimenaChacon/Clase1-11/data_hist.csv')
    data_cartera = pd.read_csv('/workspaces/CA-417_JimenaChacon/Clase1-11/data_cartera.csv')

    dp = probs(data_hist)
    dc = umbrales(dp)
    dm = momentos(data_hist)

    r11 = calibra1(dp['p1'],dm['mK21'],dm['mN21'],dm['meanN1'],dc['p1'])
    r22 = calibra1(dp['p2'],dm['mK22'],dm['mN22'],dm['meanN2'],dc['p2'])
    r12 = calibra2(dm['mKK'],dm['mNN'],dc)
    print(r11)
    print(r22)
    print(r12)

    a2 = np.sqrt(r22)
    a1 = r12/a2
    b1 = np.sqrt(r11-a1**2)

    a1,a2 = calibra_un_factor(r11,r22,r12)

    nesc = 100
    esc_perd = simula_credito(data_cartera,[a1,a2],dc,nesc)

    print(np.mean(esc_perd))


    print('Fin')