import sobol as sb
import numpy as np
from scipy.special import erfinv as inverrorfn
from scipy.special import hermite
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from sympy.core import S, pi, Rational

#================================================
# defines model constants
beta = 1 # inverse temperature
hbar = 1.0 # Plancks constant
m = 1.0 # particle mass
fmode = "dw" # potential
#================================================

#================================================
# FUNCTIONS
#================================================

#------------------------------------------------
# DEFINE POTENTIALS
#------------------------------------------------

# harmonic
omega = 1.0
omega2 = np.power(omega,2)
k = m*omega2
def ffharm(q):
  return 0.5 * k * np.power(q,2), -k * q

# morse
De = 10.0
a = 3.0
def ffmorse(q):
  return De * (1 - np.exp(-0.5 * a * q))**2, -a * De * np.exp(-0.5 * a * q)*(1 - np.exp(-0.5 * a * q))

# double well
dw = 1.0
bb = 2.0
def ffdw(x):
  return  dw * (x**4 - bb * x**2),  dw * (2.0 * bb * x - 4.0 * x**3)

# polynomial
def ffpoly(x):
  return x**2 - x**3 + x**4, -2.0 * x + 3.0 * x**2 -4.0 * x**3

#------------------------------------------------
# CALCULATE FREE ENERGIES
#------------------------------------------------

#def Aharm(beta, tol=100):
#  w = omega
#  e_j = lambda j : hbar * w * (float(j) + 0.5)
#  jmax  =  int(tol / beta / hbar / w) + 1
#  return -1.0 * beta**-1 * np.log(np.asarray([np.exp(-beta * e_j(j)) for j in range(jmax)]).sum())

#def Amorse(beta, tol=100):
#  w = a * np.sqrt(2.0 * De / m)
#  e_j = lambda j : hbar * w * (float(j) + 0.5) - (hbar * w * (float(j) + 0.5))**2 / 4.0 / De
#  jmax  =  int(np.sqrt(8.0 * m * De  / a**2) - 1) / 2 + 1
#  return -1.0 * beta**-1 * logsumexp(np.asarray([(-beta * e_j(j)) for j in range(jmax)]))

def Ascp(beta, q0, K, avgV):
  w = np.sqrt(K/m)
  x = hbar * w * beta / 2.0 
  # CLASSICAL
  #return avgV + beta**-1 * np.log(beta * w) - 0.50 * beta**-1
  # QUANTUM
  return avgV + beta**-1 * np.log(2.0 * np.sinh(x)) - hbar * w / 4.0 * np.cosh(x) / np.sinh(x) 

def Aharm(beta, q0, K):
  w = np.sqrt(K/m)
  x = hbar * w * beta / 2.0 
  # CLASSICAL
  #return beta**-1 * np.log(beta * w)
  # QUANTUM
  return beta**-1 * np.log(2.0 * np.sinh(x))

#------------------------------------------------
# 
#------------------------------------------------

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

#------------------------------------------------
# 
#------------------------------------------------

def rwt_avg(ref_par, scp_x, scp_v, scp_f, scp_par):
  scp_bw = np.zeros((scp_maxiter, scp_maxmc), float)
  scp_avgv = np.zeros(scp_maxiter, float)
  scp_avgf = np.zeros(scp_maxiter, float)
  scp_avgK = np.zeros(scp_maxiter, float)
  scp_avgD = np.zeros(scp_maxiter, float)
  scp_w = np.zeros(scp_maxiter, float)

  #computes the weights of the samples.
  for k in range(j+1):
    scp_bw[k] = np.exp( -0.50 * ref_par[1] * (scp_x[k] - ref_par[0])**2 + 0.50 * scp_par[k,1] * (scp_x[k] -  scp_par[k,0])**2)
    scp_avgv[k] =  np.average(scp_v[k,:], weights = scp_bw[k])
    scp_avgf[k] =  np.average(scp_f[k,:], weights = scp_bw[k])
    scp_avgK[k] = -np.average((scp_x[k,:] - ref_par[0]) * scp_f[k,:], weights = scp_bw[k]) * ref_par[1]
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

#------------------------------------------------
# SCP
#------------------------------------------------

def scp():

  # Hyperparameters
  fatol = 1e-4
  fdamp_qh = 0.7
  # fpush_qh < 1.0 / (1 - fdamp_qh)
  fpush_qh = 1.0
  fdamp_Kh = 0.7
  fpush_Kh = 0.7
  fthresh = 1e-5
  
  # Internal constants
  tau = 1.0
  scp_maxiter = 20
  scp_maxmc = 500
  amode = "vk"
  dmode = "vk"
  rmode = "sobol"
  
  # Initialise internal variables
  Kh = 8.0
  delta_Kh = Kh
  Kh_old = Kh
  Dh = fD(Kh)
  qh = -1.0
  dqh = 0.0
  dqh_old = 0.0
  delta_qh = 0.0
  qh_old = qh
  aharm = Aharm(beta, qh, Kh)
  atol = aharm * fatol
  ascp = aharm
  ascp_old = aharm
  
  scp_x = np.zeros((scp_maxiter, scp_maxmc), float) 
  scp_v = np.zeros((scp_maxiter, scp_maxmc), float) 
  scp_f = np.zeros((scp_maxiter, scp_maxmc), float) 
  scp_K = np.zeros((scp_maxiter, scp_maxmc), float)
  scp_par = np.zeros((scp_maxiter, 2), float)
  
  for j in range(scp_maxiter):
  
    # generates samples for the iteration and computes the potential and force at each point.
    sb.i4_sobol(1,0)
    scp_par[j] = np.asarray([qh,1.0/Dh])
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
    print j, qh, Kh, Ascp(beta, qh, Kh, scp_av), Ascp(beta, qh, Kh, scp_av) - ascp_old, atol
    if (abs(ascp - ascp_old) < atol):
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
        while True:
          vh, fh, dummKh, dum  = rwt_avg([qh, 1.0/Dh], scp_x, scp_v, scp_f, scp_par)
          #bail out condition
          if(abs(fh) < fthresh): break
          dqh_old = dqh
          dqh = np.sign(fh) * min(alpha,np.abs(fh))
          qh = qh + dqh #* np.sign(Kh)
          if( dqh * dqh_old < 0.0):
               qh = qh - dqh_old * fdamp_qh
               alpha = alpha * fdamp_qh
        delta_qh = qh - qh_old
        # Pushes the displacement if it doesn't change sign and dampens it if it does.
        if (delta_qh * delta_qh_old < 0.0):
          qh = qh_old - delta_qh_old * fdamp_qh
        else:
          qh = qh_old + delta_qh * fpush_qh
        delta_Kh_old = Kh - Kh_old
        Kh_old = Kh
        vh, fh, Kh, dum  = rwt_avg([qh, 1.0/Dh], scp_x, scp_v, scp_f, scp_par)
        delta_Kh = Kh - Kh_old
        # Pushes the change in the curvature if it doesn't change sign and dampens it if it does.
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
  
  return 0  
  
#------------------------------------------------
# VSCF
#------------------------------------------------
ndata = 41
nrms = 16.0
nint = 4001
nbasis = 20
hf = []
for i in xrange(nbasis):
  hf.append(hermite(i))

def psi(n,mass,freq,x):
    nu = mass * freq / hbar
    norm = (nu/np.pi)**0.25 * np.sqrt(1.0/(2.0**(n)*np.math.factorial(n)))
#    herm = np.polynomial.hermite.hermval(np.sqrt(nu)*x, n+1)
#    herm = hermite(n)(np.sqrt(nu)*x)
    psival = norm * np.exp(-nu * x**2 /2.0) * hf[n](np.sqrt(nu)*x)
    return psival
 
def vscf():

  ff = ffdw
  # ffdw 
  K = -2.0
  qeq = 0.0
  #K = 8.0
  #qeq = -1.0

  #ff = ffharm
  # ffharm
  #K = 1.0
  #qeq = 0.0

  #
  if(K > 0):
    whar = np.sqrt(K/m)
  else:
    whar = np.sqrt(-K/m)
  
  # 
  betathresh = 1e6
  if (beta > betathresh):
    qrms = np.sqrt(0.5/whar**2)
  else:
    qrms=np.sqrt(1/(np.exp(beta*whar**2)-1)+0.5)/whar
  qmax=nrms*qrms
  dq = qmax/((ndata-1)/2)

  # 

  # Map the potential
  qs = []
  vs = []
  vtots = []
  fs = []
  for i in xrange(-(ndata-1)/2,(ndata+1)/2):
    q = qeq + i * dq
    v,f = ff(q)
    qs.append(i*dq)
    vs.append(v - 0.5*m*whar**2*(i*dq)**2 - ff(qeq)[0])
    vtots.append(v - ff(qeq)[0])
    fs.append(f)

  np.savetxt('pot_anh.map.dat',np.c_[qs,vs])
  np.savetxt('pot_tot.map.dat',np.c_[qs,vtots])

  # Fit the potential with a cubic spline
  vspline = interp1d(np.asarray(qs), np.asarray(vs), kind='cubic', bounds_error=False)
  np.savetxt('pot.fit.dat',np.c_[np.linspace(-qmax,qmax,110),vspline(np.linspace(-qmax,qmax,110))])

  # Set up the wavefunction basis
  h = []
  for i in xrange(nbasis):
    hrow = []
    for j in xrange(nbasis):
      ddq = np.linspace(-qmax,qmax,nint)
      dv = np.sum(psi(i,m,whar,ddq) * vspline(ddq) * psi(j,m,whar,ddq)) * (ddq[1] - ddq[0])

      if (i == j):
        hrow.append( (i + 0.5) * hbar * whar + dv )
      else:
        hrow.append( dv )
    h.append(hrow)


  # Diagonalise Hamiltonian matrix
  evals, evecs = np.linalg.eigh(h)

  #order = np.argsort(evals)
  #evals = evals[order]
  #evecs = []
  #for i in xrange(nbasis):
  #  row = []
  #  for j in xrange(nbasis):
  #    row.append(evecstmp[i][order[j]])
  #  evecs.append(row)

  print evals  

  print evecs[0]
  print evecs.T[0]


  np.savetxt('psi.dat',np.c_[np.linspace(-qmax,qmax,1000), np.sum(np.asarray([psi(jj,m,whar,np.linspace(-qmax,qmax,1000)) * evecs.T[0][jj] for jj in xrange(nbasis)]),axis=0) ])

  np.savetxt('psi_gs.dat',np.c_[np.linspace(-qmax,qmax,1000),psi(0,m,whar,np.linspace(-qmax,qmax,1000))])

  A = -np.log(np.sum(np.exp(-beta*evals)))/beta
  print "free energy = ",A

  return 0
  
#================================================
# MAIN
#================================================

vscf()

scp()
