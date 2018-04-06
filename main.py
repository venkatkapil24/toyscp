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
beta = 1e+1 # inverse temperature
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
# VSCF
#------------------------------------------------
#ndata = 41 # number of sampling points
#nrms = 16.0 # number of RMS displacements for max sampling displacement
fqrms = 4.0 # spacing of sampling points in terms of RMS displacements
nint = 4001 # integration points for numerical integration of Hamiltonian matrix elements
nbasis = 20 # number of SHO states used as basis for anharmonic wvfn
nenergy = 10.0 # potential mapped until v - v_eq > nenergy * ehar
wthresh = 1e-3
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

  # ---------------------
  ff = ffdw
  # ffdw 
  #K = -2.0
  #qeq = 0.0
  K = 8.0
  qeq = -1.0
  # ---------------------
  #ff = ffharm
  # ffharm
  #K = 1.0
  #qeq = 0.0
  # ---------------------

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
    qrms = np.sqrt(1/(np.exp(beta*whar**2)-1)+0.5)/whar
  dq = fqrms * qrms


  # Map the potential in a smarter fashion -- precondition
  # calculate finite temperature harmonic oscillator energy
  ehar = 0.5*hbar*whar/np.tanh(beta*0.5*hbar*whar)
  # initialise displacements and potentials (and fores) arrays
  qs = []
  qtots = []
  vs = []
  vtots = []
  fs = []

  v,f = ff(qeq)
  qs.append(0.0)
  qtots.append(qeq)
  vs.append(v - ff(qeq)[0])
  vtots.append(v - ff(qeq)[0])
  fs.append(f)
  # sample outwards from static equilibrium position until v > ne * ehar
  i = 0
  while True:
    i += 1
    q = qeq + i * dq
    v,f = ff(q)
    qs.append(i*dq)
    qtots.append(qeq+i*dq)
    vs.append(v - 0.5*m*whar**2*(i*dq)**2 - ff(qeq)[0])
    vtots.append(v - ff(qeq)[0])
    fs.append(f)
    if((v - ff(qeq)[0]) > nenergy*ehar): 
      qmax = i*dq
      print "nplus = ",i
      break
  i = 0
  while True:
    i += 1
    q = qeq - i * dq
    v,f = ff(q)
    qs.append(-i*dq)
    qtots.append(qeq-i*dq)
    vs.append(v - 0.5*m*whar**2*(i*dq)**2 - ff(qeq)[0])
    vtots.append(v - ff(qeq)[0])
    fs.append(f)
    if ((v - ff(qeq)[0]) > nenergy*ehar): 
      qmin = -i*dq
      print "nminus = ",i
      break
  # write out potentials
  np.savetxt('pot_anh.map.dat',np.c_[qtots,vs])
  np.savetxt('pot_tot.map.dat',np.c_[qtots,vtots])

  wanh = []
  wanh.append(whar)

  # Converge anharmonic vibrational energies w.r.t. sampling range
  iter = 0
  while True:
    iter += 1

    qmin -= dq
    qmax += dq

    q = qeq + qmin
    v,f = ff(q)
    qs.append(qmin)
    qtots.append(qeq+qmin)
    vs.append(v - 0.5*m*whar**2*(qmin)**2 - ff(qeq)[0])
    vtots.append(v - ff(qeq)[0])
    fs.append(f)

    q = qeq + qmax
    v,f = ff(q)
    qs.append(qmax)
    qtots.append(qeq+qmax)
    vs.append(v - 0.5*m*whar**2*(qmax)**2 - ff(qeq)[0])
    vtots.append(v - ff(qeq)[0])
    fs.append(f)
    
    # Fit the potential with a cubic spline
    vspline = interp1d(np.asarray(qs), np.asarray(vs), kind='cubic', bounds_error=False)
    np.savetxt('pot.fit.dat',np.c_[np.linspace(qmin+qeq,qmax+qeq,110),vspline(np.linspace(qmin,qmax,110))])
    vtotspline = interp1d(np.asarray(qs), np.asarray(vtots), kind='cubic', bounds_error=False)
    np.savetxt('pot_tot.fit.dat',np.c_[np.linspace(qmin+qeq,qmax+qeq,110),vtotspline(np.linspace(qmin,qmax,110))])
    # find new best guess equilibrium position -- SHO basis expected to 
    # be a better basis when centered well -- need fewer basis functions
    #print "qmin,qmax = ",qmin,qmax
    ddq = np.linspace(qmin,qmax,nint)
    #print "min potential ",np.min(vtotspline(ddq) + ff(qeq)[0])," at ",qeq + ddq[np.argmin(vtotspline(ddq))]
  
    qeqshift = np.sum( ddq * np.exp(vtotspline(ddq) - np.min(vtotspline(ddq)) - ehar) ) / np.sum( np.exp(vtotspline(ddq) - np.min(vtotspline(ddq)) - ehar) )
    qeqnew = qeq + qeqshift
  
    # Set up the wavefunction basis
    h = []
    for i in xrange(nbasis):
      hrow = []
      for j in xrange(nbasis):
        dv = np.sum(psi(i,m,whar,ddq-qeqshift) * (vtotspline(ddq) - 0.5*m*whar**2*(ddq-qeqshift)**2) * psi(j,m,whar,ddq-qeqshift)) * (ddq[1] - ddq[0])
  
        if (i == j):
          hrow.append( (i + 0.5) * hbar * whar + dv )
        else:
          hrow.append( dv )
      h.append(hrow)
  
  
    # Diagonalise Hamiltonian matrix
    evals, evecs = np.linalg.eigh(h)
  
    A = -np.log(np.sum(np.exp(-beta*evals)))/beta
  
    wanh.append(2.0*A/hbar)

    print "Range Iteration : ",iter," Free energy = ",A

    if ( (np.abs(wanh[-1]-wanh[-2])/np.abs(wanh[-2])) < wthresh ): break

  # Converge anharmonic vibrational energies w.r.t. density of sampling points
  iter = 0
  ffqrms = fqrms
  while True:
    iter += 1
    ffqrms *= 0.5
    dq = ffqrms * qrms

    i = 1
    while True:
      q = qeq + i * dq
      v,f = ff(q)
      qs.append(i*dq)
      qtots.append(qeq+i*dq)
      vs.append(v - 0.5*m*whar**2*(i*dq)**2 - ff(qeq)[0])
      vtots.append(v - ff(qeq)[0])
      fs.append(f)
      if((i*dq) > qmax): break
      i += 2
    i = 1
    while True:
      q = qeq - i * dq
      v,f = ff(q)
      qs.append(-i*dq)
      qtots.append(qeq-i*dq)
      vs.append(v - 0.5*m*whar**2*(i*dq)**2 - ff(qeq)[0])
      vtots.append(v - ff(qeq)[0])
      fs.append(f)
      if ((-i*dq) < qmin): break
      i += 2

    # write out potentials
    np.savetxt('pot_anh.map2.dat',np.c_[qtots,vs])
    np.savetxt('pot_tot.map2.dat',np.c_[qtots,vtots])

    # Fit the potential with a cubic spline
    vspline = interp1d(np.asarray(qs), np.asarray(vs), kind='cubic', bounds_error=False)
    np.savetxt('pot.fit2.dat',np.c_[np.linspace(qmin+qeq,qmax+qeq,110),vspline(np.linspace(qmin,qmax,110))])
    vtotspline = interp1d(np.asarray(qs), np.asarray(vtots), kind='cubic', bounds_error=False)
    np.savetxt('pot_tot.fit2.dat',np.c_[np.linspace(qmin+qeq,qmax+qeq,110),vtotspline(np.linspace(qmin,qmax,110))])
    # find new best guess equilibrium position -- SHO basis expected to 
    # be a better basis when centered well -- need fewer basis functions
    #print "qmin,qmax = ",qmin,qmax
    ddq = np.linspace(qmin,qmax,nint)
    #print "min potential ",np.min(vtotspline(ddq) + ff(qeq)[0])," at ",qeq + ddq[np.argmin(vtotspline(ddq))]

    # Set up the wavefunction basis
    h = []
    for i in xrange(nbasis):
      hrow = []
      for j in xrange(nbasis):
        dv = np.sum(psi(i,m,whar,ddq-qeqshift) * (vtotspline(ddq) - 0.5*m*whar**2*(ddq-qeqshift)**2) * psi(j,m,whar,ddq-qeqshift)) * (ddq[1] - ddq[0])

        if (i == j):
          hrow.append( (i + 0.5) * hbar * whar + dv )
        else:
          hrow.append( dv )
      h.append(hrow)

    # Diagonalise Hamiltonian matrix
    evals, evecs = np.linalg.eigh(h)

    A = -np.log(np.sum(np.exp(-beta*evals)))/beta

    wanh.append(2.0*A/hbar)

    print "Density Iteration : ",iter," Free energy = ",A

    if ( (np.abs(wanh[-1]-wanh[-2])/np.abs(wanh[-2])) < wthresh ): break

    #if (beta > betathresh):
    #  qrmsanh = np.sqrt(0.5/whar**2)
    #else:
    #  qrmsanh = np.sqrt(1/(np.exp(beta*whar**2)-1)+0.5)/whar

  return 0
  
#================================================
# MAIN
#================================================

vscf()
