import sobol as sb
import numpy as np
from scipy.special import erfinv as inverrorfn
from scipy.misc import logsumexp

#defines constants
beta = 1.0
hslash = 1.0
m = 1.0

#defines constants for ff harm
omega = 1.0
omega2 = np.power(omega,2)
k = m*omega2

#defines constants for ff morse
De = 10.0
a = 3.0

#defines constants for ff dw
dw = 1.0
bb = 2.0

#defines ff functions

def ffpoly(x):
  return x**2 - x**3 + x**4, -2.0 * x + 3.0 * x**2 -4.0 * x**3

def ffdw(x):
  return  dw * (x**4 - bb * x**2),  dw * (2.0 * bb * x - 4.0 * x**3)

def ffharm(q):
  return 0.5 * k * np.power(q,2), -k * q

def ffmorse(q):
  return De * (1 - np.exp(-0.5 * a * q))**2, -a * De * np.exp(-0.5 * a * q)*(1 - np.exp(-0.5 * a * q))

def Amorse(beta, tol=100):
  w = a * np.sqrt(2.0 * De / m)
  e_j = lambda j : hslash * w * (float(j) + 0.5) - (hslash * w * (float(j) + 0.5))**2 / 4.0 / De
  jmax  =  int(np.sqrt(8.0 * m * De  / a**2) - 1) / 2 + 1
  return -1.0 * beta**-1 * logsumexp(np.asarray([(-beta * e_j(j)) for j in range(jmax)]))

def Ascp(beta, q0, K, avgV):
  w = np.sqrt(K/m)
  x = hslash * w * beta / 2.0 
  return avgV + beta**-1 * np.log(beta * w) - 0.50 * beta**-1 
  return avgV + beta**-1 * np.log(2.0 * np.sinh(x)) - hslash * w / 4.0 * np.cosh(x) / np.sinh(x)

def Aharm(beta, q0, K):
  w = np.sqrt(K/m)
  x = hslash * w * beta / 2.0 
  return beta**-1 * np.log(beta * w)
  return beta**-1 * np.log(2.0 * np.sinh(x))

#defines functions for the scp method
def fD(K):
  if(K > 0):
    dw = np.sqrt(K/m)
    return 1.0 / dw**2 / beta
    return np.nan_to_num(hslash / 2.0 / dw / np.tanh(beta * hslash * dw / 2.0))
  else:
    dw = np.sqrt(-K/m)
    return 1.0 / dw**2 / beta
    return np.nan_to_num(hslash / 2.0 / dw / np.tan(beta * hslash * dw / 2.0))

#harmonic guess
qh = -0.9
Kh = 5.72


dqct = np.sqrt(2.0 / beta / Kh)
v0, f0 = ffdw(qh)
q = qh
c = 0
i = 0

while True:
  q = qh + i * dqct
  v , f = ffdw(q)
  c += 1 
  if (v - v0 > beta**-1): 
    v_plus = v - v0
    q_plus = q
    break
  i += 1


i = 0
while True:
  q = qh - i * dqct
  v , f = ffdw(q)
  c += 1
  if (v - v0 > beta**-1): 
    v_minus = v - v0
    q_minus = q
    break
  i += 1

qf = (q_plus + q_minus * np.sqrt(v_plus / v_minus)) / (1 +  np.sqrt(v_plus / v_minus))  
Kf = 2.0 * v_plus / (q - q_plus)**2

print "# ", c, " points were used."
print (v_plus) / beta**-1, q_plus
print (v_minus) / beta**-1, q_minus 
print "q, ", qf
print "K, ", Kf
