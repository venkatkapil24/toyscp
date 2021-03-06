import sobol as sb
import numpy as np
from scipy.special import erfinv as inverrorfn
from scipy.special import hermite
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from sympy.core import S, pi, Rational
import argparse

#================================================
# defines model constants
hbar = 1.0 # Plancks constant
m = 1.0 # particle mass
#================================================

#================================================
# FUNCTIONS
#================================================

#------------------------------------------------
# DEFINE FREE ENERGIES
#------------------------------------------------

def Aharm(K,beta):
  w = np.sqrt(K/m)
  x = hbar * w * beta / 2.0
  # classical
  #return beta**-1 * np.log(beta * w)
  # quantum
  return beta**-1 * np.log(2.0 * np.sinh(x))

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
#def ffpoly(x, p):
#  return p[0] * x**2 + p[1] * x**3 + p[2] * x**4, -1.0 * (2.0 * p[0] * x + 3.0 * x**2 * p[1] -4.0 * x**3 * p[2])
def ffpoly(x, p):
  return p[0] * x**2 + p[1] * x**4 + p[2] * x**6 + p[3] * x**3, -1.0 * (2.0 * p[0] * x + 4.0 * x**3 * p[1] + 6.0 * x**5 * p[2] + p[3] * x**2 * 3.0)
  
#------------------------------------------------
# VSCF
#------------------------------------------------
fqrms = 4.0 # spacing of sampling points in terms of RMS displacements
nint = 4001 # integration points for numerical integration of Hamiltonian matrix elements
nbasis = 25 # number of SHO states used as basis for anharmonic wvfn
nenergy = 10.0 # potential mapped until v - v_eq > nenergy * ehar
wthresh = 1e-3

# 
hf = []
for i in xrange(4*nbasis):
  hf.append(hermite(i))

def psi(n,mass,freq,x):
    nu = mass * freq / hbar
    norm = (nu/np.pi)**0.25 * np.sqrt(1.0/(2.0**(n)*np.math.factorial(n)))
    psival = norm * np.exp(-nu * x**2 /2.0) * hf[n](np.sqrt(nu)*x)
    return psival
 
def imf(param, qeq, K, beta):

  fmode = 'poly'
  # set potential
#  if (fmode == 'dw'):
#    ff = ffdw
#  elif (fmode == 'harm'):
#    ff = ffharm
#  elif (fmode == 'poly'):
#    ff = ffpoly
  def ff(x): return ffpoly(x, param)

  # evaluate harmonic free energy as reference
  Ahar = Aharm(K, beta)

  # evaluate initial harmonic vibrational energy
  if(K > 0):
    whar = np.sqrt(K/m)
  else:
    whar = np.sqrt(-K/m)
  
  # evaluate harmonic RMS displacement
  betathresh = 1e6
  if (beta > betathresh):
    qrms = np.sqrt(0.5/whar**2)
  else:
    qrms = np.sqrt(np.exp(-beta*whar**2)/(1 - np.exp(-beta*whar**2))+0.5)/whar
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
      break

  wanh = []
  wanh.append(whar)

  # Initialise arrays for storing type of iteration, range, points, and free energy
  itypes = []
  qmins = []
  qmaxs = []
  npoints = []
  Ahars = []
  Aanhs = []
  Adiffs = []
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
    vtotspline = interp1d(np.asarray(qs), np.asarray(vtots), kind='cubic', bounds_error=False)
    # find new best guess equilibrium position -- SHO basis expected to 
    # be a better basis when centered well -- need fewer basis functions
    #print "qmin,qmax = ",qmin,qmax
    ddq = np.linspace(qmin,qmax,nint)
    #print "min potential ",np.min(vtotspline(ddq) + ff(qeq)[0])," at ",qeq + ddq[np.argmin(vtotspline(ddq))]
 
    qeqshift = np.sum( ddq * np.exp(-vtotspline(ddq)+np.min(vtotspline(ddq)) ) ) / np.sum( np.exp(-vtotspline(ddq)+np.min(vtotspline(ddq)) ) )
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
  
  
    # Diagonalise Hamiltonian matrix and evaluate anharmonic free energy and vibrational freq
    evals, evecs = np.linalg.eigh(h)
    A = -np.log(np.sum(np.exp(-beta*evals)))/beta 
    wanh.append(2.0*A/hbar)

    # Print free energy to terminal
    # print "Range Iteration : ",iter," Free energy = ",A+ff(qeq)[0]

    # Store range, points, and free energy in arrays
    itypes.append('range')
    qmins.append(qmin)
    qmaxs.append(qmax)
    npoints.append(len(qs))
    Ahars.append(Ahar+ff(qeq)[0])
    Aanhs.append(A+ff(qeq)[0])
    Adiffs.append(A-Ahar)
    
    # Check whether anharmonic frequency is converged
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

    # Fit the potential with a cubic spline
    vspline = interp1d(np.asarray(qs), np.asarray(vs), kind='cubic', bounds_error=False)
    vtotspline = interp1d(np.asarray(qs), np.asarray(vtots), kind='cubic', bounds_error=False)
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

    # Diagonalise Hamiltonian matrix and evaluate anharmonic free energy and vibrational freq
    evals, evecs = np.linalg.eigh(h)
    A = -np.log(np.sum(np.exp(-beta*evals)))/beta
    wanh.append(2.0*A/hbar)

    # Print free energy to terminal
    #print "Density Iteration : ",iter," Free energy = ",A+ff(qeq)[0]

    # Store range, points, and free energy in arrays
    itypes.append('density')
    qmins.append(qmin)
    qmaxs.append(qmax)
    npoints.append(len(qs))
    Ahars.append(Ahar+ff(qeq)[0])
    Aanhs.append(A+ff(qeq)[0])
    Adiffs.append(A-Ahar)

    # Check whether anharmonic frequency is converged
    if ( (np.abs(wanh[-1]-wanh[-2])/np.abs(wanh[-2])) < wthresh ): break

    #if (beta > betathresh):
    #  qrmsanh = np.sqrt(0.5/whar**2)
    #else:
    #  qrmsanh = np.sqrt(1/(np.exp(beta*whar**2)-1)+0.5)/whar

  iter = 0
  nnbasis = nbasis
  while True:
    iter += 1
    nnbasis += 5

    # Set up the wavefunction basis
    h = []
    for i in xrange(nnbasis):
      hrow = []
      for j in xrange(nnbasis):
        dv = np.sum(psi(i,m,whar,ddq-qeqshift) * (vtotspline(ddq) - 0.5*m*whar**2*(ddq-qeqshift)**2) * psi(j,m,whar,ddq-qeqshift)) * (ddq[1] - ddq[0])

        if (i == j):
          hrow.append( (i + 0.5) * hbar * whar + dv )
        else:
          hrow.append( dv )
      h.append(hrow)

    # Diagonalise Hamiltonian matrix and evaluate anharmonic free energy and vibrational freq
    evals, evecs = np.linalg.eigh(h)
    A = -np.log(np.sum(np.exp(-beta*evals)))/beta
    wanh.append(2.0*A/hbar)

    # Print free energy to terminal
    #print "Basis Iteration : ",iter,"Har/Anh/Diff Free energy = ",Ahar+ff(qeq)[0],A+ff(qeq)[0],A-Ahar

    # Store range, points, and free energy in arrays
    itypes.append('basis')
    qmins.append(qmin)
    qmaxs.append(qmax)
    npoints.append(len(qs))
    Ahars.append(Ahar+ff(qeq)[0])
    Aanhs.append(A+ff(qeq)[0])
    Adiffs.append(A-Ahar)

    # Check whether anharmonic frequency is converged
    if ( (np.abs(wanh[-1]-wanh[-2])/np.abs(wanh[-2])) < wthresh ): break


  # write out potentials
  dddq = np.linspace(qmin,qmax,1000)
  np.savetxt('pot_anh.map.dat',np.c_[qtots,vs])
  np.savetxt('pot_tot.map.dat',np.c_[qtots,vtots])
  np.savetxt('pot_anh.fit.dat',np.c_[dddq,vspline(dddq)])
  np.savetxt('pot_tot.fit.dat',np.c_[dddq,vtotspline(dddq)])
  # write out wvfn
  np.savetxt('psi_gs.dat',np.c_[dddq-qeqshift,psi(0,m,whar,dddq-qeqshift)])
  np.savetxt('psi.dat',np.c_[dddq-qeqshift, np.sum(np.asarray([psi(jj,m,whar,dddq-qeqshift) * evecs.T[0][jj] for jj in xrange(nbasis)]),axis=0) ])
  # write convergence to logfile
  np.savetxt('log.imf.'+fmode+'.'+str(qeq)+'.'+str(K)+'.'+str(beta)+'.dat',np.c_[qmins,qmaxs,npoints,Ahars,Aanhs,Adiffs], fmt='%6.3f %6.3f %3i % 12.6f % 12.6f % 12.6f')

  # done
  return Ahars[-1],Aanhs[-1],Adiffs[-1]
  
#================================================
# MAIN
#================================================

#parser = argparse.ArgumentParser()
#parser.add_argument("pot", help="type of potential [harm, dw, morse]", type=str)
#parser.add_argument("qeq", help="equilibrium position (harm approx)", type=float)
#parser.add_argument("Keq", help="equilibrium Hessian (harm approx)", type=float)
#parser.add_argument("invT", help="inverse temperature", type=float)
#args = parser.parse_args()

#print vscf(args.pot,args.qeq,args.Keq,args.invT)

parser = argparse.ArgumentParser()
parser.add_argument("p1", help="parameter p1 of the polynomial potential [p1, p2, p3, p4]", type=float)
parser.add_argument("p2", help="parameter p2 of the polynomial potential [p1, p2, p3, p4]", type=float)
parser.add_argument("p3", help="parameter p3 of the polynomial potential [p1, p2, p3, p4]", type=float)
parser.add_argument("p4", help="parameter p4 of the polynomial potential [p1, p2, p3, p4]", type=float)
parser.add_argument("invT", help="inverse temperature", type=float)
args = parser.parse_args()

p = np.asarray([args.p1, args.p2, args.p3, args.p4])
xtrial = np.linspace(-10,10,1e4)
qeq = xtrial[np.argmin(ffpoly(xtrial, p)[0])]
Keq = 2 * p[0]  + 6 * p[3] * qeq + 12 * p[1] * qeq**2 + 30 * p[2] * qeq**4

print imf(p,qeq,Keq,args.invT)
