################################ Imports ########################################

import numpy as np
import matplotlib.pyplot as plt
import math
import time

start = time.perf_counter()

#################################################################################


######################### Variable deffinition ##################################

N = 500
ni = 1000
buffer = 2000
Ncor = 1
sampleSize = 3


mu = 1
m = 1
g = 0
ga = 0
l = 7.5
eps = 3
epsa = 10

Elist = list()
Elista = list()
llist = list()

dt = 0.5
ti = 0
tf = N * dt + ti

x = np.zeros(N)

xold = np.zeros(N)
xnew = np.zeros(N)
x2 = list()
x2a = list()
x4a = list()
X = list()
Xa = list()
xlen = len(x)

xi = 0
xf = 0
x[0] = xi
x[N - 1] = xf

S = 0
Sold = 0
Snew = 0
dS = 0
dSa = 0
sign = 0
w = (mu/math.sqrt(m))*(1-(mu**2*dt**2/(8*m)))

e0 = 1/2*w
e1 = 3/2*w

e0a = ((1/2)+(3*l/4)-(21*l**2/8))*w 
e1a = ((1/2)+(3*l/4)-(21*l**2/8))*w + w

la = np.linspace(-0,0.5,100)
y1 = ((1/2)+(3*la/4))*w
y2 = ((1/2)+(3*la/4)-(21*la**2/8))*w

e0array = [0, 1000]
e0plot = [0.484, 0.484]


ss = 0
sign = 0
signa = 0

#################################################################################


####################### Main function deffinition ###############################

def V(x):
    return mu**2 * x**2 / 2

def Va(x):
    return (mu**2 * x**2 / 2) + (l * x**4)

def S(j, x):  #S for harmonic osc.
    return (dt * V(x[j]) + x[j] * (x[j] - x[(j+1) % N] - x[(j-1) % N] ) / dt)

def Sa(j, x):  #S for anharmonic osc.
    return (dt * Va(x[j]) + x[j] * (x[j] - x[(j+1) % N] - x[(j-1) % N] ) / dt)

def update(x): #update 1 chain
    global g
    global dS
    global ss
    for j in range(1,N-2):
        old_x = x[j] #save original value
        old_Sj = S(j, x)
        # print(old_x, old_Sj)  #dbug
        x[j] = x[j] + np.random.uniform(-eps,eps) # update x[j]
        dS = S(j, x) - old_Sj # change in action
        p = np.e ** (-dS)
        r = np.random.uniform(0,1)
        if dS>0: #restore old value
            if p < r:
                x[j] = old_x
                X.append(x[j])
                # Elist.append(Etot(j, x))
            elif p > r and sign == 1:
                g = g + 1
                X.append(x[j])
                # Elist.append(Etot(j, x))
        elif dS < 0 and sign == 1:
            g = g + 1
            X.append(x[j])
            # Elist.append(Etot(j, x))

def updatea(x): #update 1 chain anharmonic
    global ga
    global dSa
    for j in range(1,N-1):
        old_x = x[j] #save original value
        old_Sj = Sa(j, x)
        # print(old_x, old_Sj)  #dbug
        x[j] = x[j] + np.random.uniform(-epsa,epsa) # update x[j]
        dSa = Sa(j, x) - old_Sj # change in action
        p = np.e ** (-dSa)
        r = np.random.uniform(0,1)
        if dSa>0: #restore old value
            if p < r:
                x[j] = old_x
                Xa.append(x[j])
                # Elista.append(Etot(j, x))
            elif p > r and signa == 1:
                ga = ga + 1
                Xa.append(x[j])
                # Elista.append(Etot(j, x))
        elif dSa < 0 and signa == 1:
            ga = ga + 1
            Xa.append(x[j])
            # Elista.append(Etot(j, x))

def Etot(j, x):
    for i in range(1, N-1):
        Et = 0
        Ei = m/(2*dt**2) * (x[j+1]-x[i])**2 + V(x[j])
        Et += Ei
    return Et

def Etota(j, x):
    for i in range(1, N-1):
        Et = 0
        Ei = m/(2*dt**2) * (x[j+1]-x[i])**2 + V(x[j])
        Et += Ei
    return Et

def PI(x):
    global g
    global sign
    global ss
    print('Initialising')
    for i in range(1, N-1): # Initialize
        x[i] = 0
    print('Thermalising')
    for i in range(0, buffer): # Thermalize
        update(x)
    sign = 1
    print('Main')
    g = 0
    for i in range(0, ni): # Run
        # print(i)	#dbug
        # update(x)
        for i in range(0, Ncor):
            update(x)
        x2.append(np.average([v ** 2 for v in x]))

def PIa(x):
    global ga
    global signa
    print('Initialising')
    for i in range(1, N-1): # Initialize
        x[i] = 0
    print('Thermalising')
    for i in range(0, buffer): # Thermalize
        updatea(x)
    signa = 1
    ga = 0
    print('Main')
    for i in range(0, ni): # Run
        #updatea(x)
        for _ in (0,Ncor):
            updatea(x)
        x2a.append(np.average([v ** 2 for v in x]))
        x4a.append(np.average([v ** 4 for v in x]))
        # print(i)  # dbug

def AHOgraph(n):
    global l
    for j in range(n):
        print(j+1,'/', n)
        l = l + (0.25/n)*j
        PIa(x)
        e0a_PI = (m * w ** 2 * np.average(x2a)) + (3 * l * np.average(x4a))
        Elist.append(e0a_PI)
        llist.append(l)

#################################################################################


######################### Data & Error Generators ###############################

def mean(x, n):
    return sum(x)/n


def stDevi(x, n):
    newSig2 = 0
    for i in range(n):
        oldSig2 = newSig2
        sig2 = (x[i]-mean(x, sampleSize))**2 / (n-1)
        newSig2 = oldSig2 + sig2
    stDev=np.sqrt(newSig2)
    return stDev

error=[]
def StError(sig, n):
    global error
    er = sig / np.sqrt(n)
    error.append(er) 


data = []
def dataGen(x, n):
    global data
    for i in range(n):
        print(i+1, '/', sampleSize)
        PI(x)
        # PIa(x)
        # e0a_PI = (m * w ** 2 * np.average(x2a)) + (3 * l * np.average(x4a))
        e0_PI = m * w ** 2 * np.average(x2)
        data.append(e0_PI)

dataGen(x, sampleSize)
dev = stDevi(data, sampleSize)
StError(dev, sampleSize)

###############################################################################


# ########################### Histagram Plotters ################################

# binNo = 500

# xa = np.linspace(-2.5, 2.5, binNo) #HO prob. linspace
# xb = np.linspace(0, 10, binNo) #HO energy linspace
# xc = np.linspace(-2.5, 2.5, binNo) #AHO prob. linspace
# xd = np.linspace(0, 10, binNo) #AHO energy linspace

# a = np.array(X)
# b = np.array(Elist)
# c = np.array(Xa)
# d = np.array(Elista)

# weights  = np.ones_like(a) / len(a)
# weightsb = np.ones_like(b) / len(b)
# weightsc = np.ones_like(c) / len(c)
# weightsd = np.ones_like(d) / len(d)

# hist, bins = np.histogram(a, bins=xa)
# histb, binsb = np.histogram(b, bins=xb)
# histc, binsc = np.histogram(c, bins=xc)
# histd, binsd = np.histogram(d, bins=xd)

# normHist = hist / max(hist)

# center = 0.5*(bins[1:]+bins[:-1])
# centerb = 0.5*(binsb[1:]+binsb[:-1])
# centerc = 0.5*(binsc[1:]+binsc[:-1])

# std = np.sqrt(a)
# stdb = np.sqrt(histb)
# stdc = np.sqrt(histc)
# plt.errorbar(center, normHist, yerr=sE, ecolor='k', ls='none', capsize=2, elinewidth=0.5, label='Calculated Prob. Dens.')
# #plt.scatter(center, hist, color='r', marker='h', s=5)
# k = np.linspace(-2.5, 2.5, 100)
# k2 = np.linspace(0, 10, 100)
# psi2 = np.exp(-(k**2))/(np.sqrt(np.pi))

# plt.plot(k, psi2, 'r--', label='Analytical Prob. Dens.')

# plt.errorbar(, , yerr=, ecolor='k', marker='.', ls='none', capsize=2, elinewidth=0.5, label='Calculated Prob. Dens.')
# plt.hist(a, bins=xa, weights=weights , density=True, label='Calculated Prob. Dens.', histtype='barstacked')
# plt.hist(b, bins=xb, weights=weightsb, density=True, label='Calculated', histtype='barstacked')
# plt.hist(c, bins=xc, weights=weightsc, density=True, label='lambda=0 ', histtype='step')
# plt.hist(d, bins=xd, weights=weightsd, density=True, label='Calculated', histtype='barstacked')

# def fitfunc(x):
#     return np.e**(-l*x)

# AHO psi^2 graph generator
# it = 0
# for i in range(5):
#     Xa = list()
#     PIa(x)
#     c = np.array(Xa)
#     weightsc = np.ones_like(c) / len(c)
#     plt.hist(c, bins=xc,weights=weightsc, density=True, label='lambda=' + str(l), histtype='stepfilled', alpha=0.5, edgecolor='none')
#     #plt.errorbar(centerc, histc, yerr=stdc, ecolor='k', marker='.', ls='none', capsize=2, elinewidth=0.5, label='Calculated Prob. Dens.')
#     plt.show()
#     it += 1 
#     if it == 1:
#         l = 2
#     if it == 2:
#         l = 10
#     if it == 3:
#         l = 50
#     if it == 4:
#         l = 1000
  
# AHO Edens graph generator
# ita = 0
# for i in range(5):
#     Xa = list()
#     PIa(x)
#     d = np.array(Xa)
#     weightsd = np.ones_like(d) / len(d)
#     plt.hist(d, bins=xd,weights=weightsd, density=True, label='lambda=' + str(l), histtype='stepfilled', alpha=0.5, edgecolor='black')
#     plt.show()
#     ita += 1 
#     if ita == 1:
#         l = 2
#     if ita == 2:
#         l = 10
#     if ita == 3:
#         l = 50
#     if ita == 4:
#         l = 1000

#################################################################################


################################# Data presentation ###############################

xbar = mean(data, sampleSize)
finish = time.perf_counter()

print('Finished in', round(finish-start, 2), 'seconds')
if sign == 1:
    e0_PI = m * w ** 2 * np.average(x2)
    gp = g / (ni * (N - 1)) * 100
    print('======Harmonic Oscillator========')
    print('Acceptance number/rate(%):', g,'/',round(gp, 2),'%')
    print('Vacuum energy using PI:',round(e0_PI, 3))
    print('Vacuum energy mean using PI:',round(xbar, 3))
    print('1st Excitation Energy using PI:', round(xbar+w, 3))
    print('Theoretical vacuum energy:',round(e0, 3))
    print('Theoretical 1st excitation energy:',round(e1, 3))
    print('Percentage error:', round(abs((e0 - e0_PI))/e0 * 100, 2), '%')
    print('Standar Deviation:', dev)
    print('Standrd Error:', error)
    print('% std error:', error/xbar * 100 )

if signa == 1:
    e0a_PI = (m * w ** 2 * np.average(x2a)) + (3 * l * np.average(x4a))
    gpa = ga / (ni * (N - 1)) * 100
    print('======Anharmonic Oscillator========')
    print('Acceptance number/rate(%):', ga,'/',round(gpa, 2),'%')
    #print('Vacuum energy using PI:',round(e0a_PI, 3))
    print('Vacuum energy mean using PI:',round(xbar, 3))
    print('1st Excitation Energy using PI:', round(xbar+w, 3))
    print('Theoretical vacuum energy:',round(e0a, 3))
    print('Theoretical 1st excitation energy:',round(e1a, 3))
    print('Percentage error:', round(abs((e0a - e0a_PI))/e0 * 100, 2), '%')
    print('Standar Deviation:', dev)
    print('Standrd Error:', error)
    print('% std error:', error/xbar * 100)


# ar1 = [0, 0.2, 0.5, 1, 2.5, 5, 7.5, 10] #l
# ar2 = [1.434, 1.573, 1.709, 1.709, ] #E1
# ar3 = [0.465,] #E0
# aeYerr = [0.00117, ] #N Vary error
# e1Yerr = [] #N Vary error

# plt.plot(e0array, e0plot, '--r', label='Analytical result')
# plt.plot(la, y1, '--g', label='1st order purturbation sol.')
# plt.plot(la, y2, '--b', label='2nd order purturbation sol.')
# plt.errorbar(ar1, ar2, yerr=aeYerr, capsize=2, ecolor='black', barsabove=True, label='Calculated E_0', elinewidth=1)
# plt.plot(ar1, ar3, label='Analytical result')
# plt.gca().invert_xaxis()
# plt.xlabel('dt')
# plt.ylabel('E_0')

# plt.legend()
# plt.show()

################################################################################