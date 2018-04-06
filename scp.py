import sobol as sb
import numpy as np
from scipy.special import erfinv as inverrorfn
from scipy.misc import logsumexp

#defines constants
beta = 1.0
hbar = 1.0
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

#def Aharm(beta, tol=100):
#  w = omega
#  e_j = lambda j : hbar * w * (float(j) + 0.5)
#  jmax  =  int(tol / beta / hbar / w) + 1
#  return -1.0 * beta**-1 * np.log(np.asarray([np.exp(-beta * e_j(j)) for j in range(jmax)]).sum())

def Amorse(beta, tol=100):
  w = a * np.sqrt(2.0 * De / m)
  e_j = lambda j : hbar * w * (float(j) + 0.5) - (hbar * w * (float(j) + 0.5))**2 / 4.0 / De
  jmax  =  int(np.sqrt(8.0 * m * De  / a**2) - 1) / 2 + 1
  return -1.0 * beta**-1 * logsumexp(np.asarray([(-beta * e_j(j)) for j in range(jmax)]))

def Ascp(beta, q0, K, avgV):
  w = np.sqrt(K/m)
  x = hbar * w * beta / 2.0 
  #return avgV + beta**-1 * np.log(beta * w) - 0.50 * beta**-1 
  return avgV + beta**-1 * np.log(2.0 * np.sinh(x)) - hbar * w / 4.0 * np.cosh(x) / np.sinh(x)

def Aharm(beta, q0, K):
  w = np.sqrt(K/m)
  x = hbar * w * beta / 2.0 
  #return beta**-1 * np.log(beta * w)
  return beta**-1 * np.log(2.0 * np.sinh(x))

#defines functions for the scp method
def fD(K):
  if(K > 0):
    dw = np.sqrt(K/m)
    #return 1.0 / dw**2 / beta
    return np.nan_to_num(hbar / 2.0 / dw / np.tanh(beta * hbar * dw / 2.0))
  else:
    dw = np.sqrt(-K/m)
    #return 1.0 / dw**2 / beta
    return np.nan_to_num(hbar / 2.0 / dw / np.tan(beta * hbar * dw / 2.0))

#######################################################################
def rwt_avg(ref_par, scp_x, scp_v, scp_f, scp_par):
  if (Kh  >= 0):
    w = np.sqrt(Kh / m)
    x = beta * hbar * w / 2.0
    avgVharm = hbar * w / 4.0 / np.tanh(x)
  else:
    w = np.sqrt(-Kh / m)
    x = beta * hbar * w / 2.0
    avgVharm = hbar * w / 4.0 / np.tan(x)

  scp_bw = np.zeros((scp_maxiter, scp_maxmc), float)
  scp_avgv = np.zeros(scp_maxiter, float)
  scp_avgf = np.zeros(scp_maxiter, float)
  scp_avgK = np.zeros(scp_maxiter, float)
  scp_avgD = np.zeros(scp_maxiter, float)
  scp_w = np.zeros(scp_maxiter, float)

  #computes the weights of the samples.
  for k in range(j+1):
    scp_bw[k] = np.exp( -0.50 * ref_par[1] * (scp_x[k] - ref_par[0])**2 + 0.50 * scp_par[k,1] * (scp_x[k] -  scp_par[k,0])**2)
    scp_avgv[k] =  avgVharm + np.average(scp_v[k,:] - 0.50 * ref_par[2] * (scp_x[k] - ref_par[0])**2, weights = scp_bw[k])
    scp_avgf[k] =  0 + np.average(scp_f[k,:] - ref_par[2] * (scp_x[k] - ref_par[0]), weights = scp_bw[k])
    scp_avgK[k] =  ref_par[0] -np.average((scp_x[k,:] - ref_par[0]) * (scp_f[k,:]  + ref_par[2] * (scp_x[k] - ref_par[0])), weights = scp_bw[k]) * ref_par[1]
    scp_avgD[k] =  np.average((scp_x[k,:] - ref_par[0])**2, weights = scp_bw[k])
    scp_w[k] = np.nan_to_num(np.exp(-1.0*np.var(np.log(scp_bw[k]))))
  #computes the weighted average.
  scp_w = scp_w / scp_w.sum()
  #print scp_w
  scp_av = np.average(scp_avgv, weights = scp_w)
  scp_af = np.average(scp_avgf, weights = scp_w)
  scp_aK = np.average(scp_avgK, weights = scp_w)
  scp_aD = np.average(scp_avgD, weights = scp_w)

  return scp_av, scp_af, scp_aK, scp_aD

#Hyperparams
fatol = 1e-4
fdamp_qh = 0.7
#fpush_qh < 1.0 / (1 - fdamp_qh)
fpush_qh = 1.0
fdamp_Kh = 0.7
fpush_Kh = 0.7
fthresh = 1e-5

# Constants
Kh = 1.0
delta_Kh = Kh
Kh_old = Kh
Dh = fD(Kh)
qh = -0.0
dqh = 0.0
dqh_old = 0.0
delta_qh = 0.0
qh_old = qh
tau = 1.0
scp_maxiter = 200
scp_maxmc = 1000
amode = "vk"
dmode = "vk"
rmode = "sobol"
fmode = "harm"
aharm = Aharm(beta, qh, Kh)
atol = aharm * fatol
ascp = aharm
ascp_old = aharm
print "# aharm, tol = ", aharm, atol

scp_x = np.zeros((scp_maxiter, scp_maxmc), float) 
scp_v = np.zeros((scp_maxiter, scp_maxmc), float) 
scp_f = np.zeros((scp_maxiter, scp_maxmc), float) 
scp_K = np.zeros((scp_maxiter, scp_maxmc), float)
scp_par = np.zeros((scp_maxiter, 3), float)

sb.i4_sobol(1,0)
for j in range(scp_maxiter):

  # generates samples for the iteration and computes the potential and force at each point.
  scp_par[j] = np.asarray([qh,1.0/Dh, Kh])
  for i in range(0, scp_maxmc, 2):
    if(rmode == "pseudo"):
      scp_x[j,i] = qh + np.sqrt(Dh) * np.random.normal()
    elif(rmode == "sobol"):
      delta = np.sqrt(Dh) * np.sqrt(2.0) * inverrorfn(2.0 * sb.i4_sobol(1,j*scp_maxmc/2 + i/2 + 1) - 1.0)
      scp_x[j,i] = qh + delta
      scp_x[j,i+1] = qh - delta
    if(fmode == "harm"):
      scp_v[j,i], scp_f[j,i] = ffharm(scp_x[j,i])
      scp_v[j,i+1], scp_f[j,i+1] = ffharm(scp_x[j,i+1])
    elif(fmode == "morse"):
      scp_v[j,i], scp_f[j,i] = ffmorse(scp_x[j,i])
      scp_v[j,i+1], scp_f[j,i+1] = ffmorse(scp_x[j,i+1])
    elif(fmode == "dw"):
      scp_v[j,i], scp_f[j,i] = ffdw(scp_x[j,i])
      scp_v[j,i+1], scp_f[j,i+1] = ffdw(scp_x[j,i+1])
    elif(fmode == "poly"):
      scp_v[j,i], scp_f[j,i] = ffpoly(scp_x[j,i])
      scp_v[j,i+1], scp_f[j,i+1] = ffpoly(scp_x[j,i+1])

  #computes the avg value of the potential, force and the Hessian.
  if(amode == "vk"):
      scp_av, scp_af, scp_aK, dum = rwt_avg(scp_par[j], scp_x, scp_v, scp_f, scp_par)
  elif(amode == "sb"):
      scp_av, scp_af, scp_aK = np.mean(scp_v[j]), np.mean(scp_f[j]), -np.mean(scp_x[j] * scp_f[j])

  #prints the averages from the previous step
  ascp_old = ascp
  ascp = Ascp(beta, qh, Kh, scp_av)
  print j, qh, Kh, Ascp(beta, qh, Kh, scp_av), abs(ascp - ascp_old) / abs(ascp_old), fatol
  if (abs(ascp - ascp_old) / abs(ascp_old) < fatol):
    break

  #computes the avg value of the potential, force and the hessian.
  if(dmode == "nr"):
      vh, fh, Kh, Dh = rwt_avg([qh, 1.0/Dh], scp_x, scp_v, scp_f, scp_par)
      print Kh, 1.0/Dh/beta, 1.0 / np.mean((scp_x[j])**2) / beta
      qh +=  fh * Dh * beta
  if(dmode == "sb"):
      vh, fh, Kh, dum = rwt_avg([qh, 1.0/Dh], scp_x, scp_v, scp_f, scp_par)
      Dh = fD(Kh)
      qh +=  fh / Kh
      print j, qh, Kh
  elif(dmode == "vk"):
      alpha = 0.005
      delta_qh_old = qh - qh_old
      qh_old = qh
      #for x in range(1000):
      #  vh, fh, dummKh, dum  = rwt_avg([qh, 1.0/Dh], scp_x, scp_v, scp_f, scp_par)
      #  #bail out condition
      #  dqh = np.sign(fh) * min(alpha,np.abs(fh))
      #  qh = qh + dqh 
      while True:
        vh, fh, dummKh, dum  = rwt_avg([qh, 1.0/Dh], scp_x, scp_v, scp_f, scp_par)
        #bail out condition
        if(abs(fh) < fthresh): break
        dqh_old = dqh
        dqh = np.sign(fh) * min(alpha,np.abs(fh))
        qh = qh + dqh #* np.sign(Kh)
       #print qh, fh, dqh, dqh_old
        if( dqh * dqh_old < 0.0):
             qh = qh - dqh_old * fdamp_qh
             alpha = alpha * fdamp_qh
      delta_qh = qh - qh_old
      #Pushes the displacement if it doesn't change sign and dampens it if it does.
      if (delta_qh * delta_qh_old < 0.0):
        qh = qh_old - delta_qh_old * fdamp_qh
      else:
        qh = qh_old + delta_qh * fpush_qh
      #print "deltas," , delta_qh, delta_qh_old
      delta_Kh_old = Kh - Kh_old
      Kh_old = Kh
      vh, fh, Kh, dum  = rwt_avg([qh, 1.0/Dh], scp_x, scp_v, scp_f, scp_par)
      delta_Kh = Kh - Kh_old
      #Pushes the change in the curvature if it doesn't change sign and dampens it if it does.
      if (delta_Kh * delta_Kh_old < 0.0):
        Kh = Kh_old - delta_Kh_old * fdamp_Kh
      else:
        Kh = Kh_old + delta_Kh * fpush_Kh
      Dh = fD(Kh)
  elif(dmode == "tdep"):
    f_all = scp_f[j].flatten()
    x_all = scp_x[j].flatten()
    for x in range(1):
        Kh = -1.0 * np.dot(f_all, x_all - qh) / np.dot(x_all - qh, x_all - qh)
        qh = np.sum(f_all + Kh * x_all) / f_all.size / Kh
        Dh = fD(Kh)
    print j, qh, Kh, f_all.size

#prints the final average
#print qh, Kh, scp_av, scp_aK, Ascp(beta, qh, Kh, scp_av), Amorse(beta)
