#import sobol as sb
import numpy as np
from scipy.special import erfinv as inverrorfn
from scipy.special import hermite
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import scipy.integrate as integrate
#from sympy.core import S, pi, Rational
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
# 1D potentials 
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
w = -np.sqrt(2.0)
v = 1.0
k = -m*np.power(w,2)
l = m*np.power(v,2)
def ffdw(q):
  return  k * q**2 + l * q**4, -(2.0 * k * q + 4.0 * l * q**3)
# polynomial
def ffpoly(x, px):
  return (px[0] * x**2 + px[1] * x**3 + px[2] * x**4 + px[3] * x**6), -(2.0 * px[0] * x + 3.0 * px[1] * x**2 + 4.0 * px[2] * x**3 + 6.0 * px[3] * x**5)

#------------------------------------------------
# 2D potentials 
#------------------------------------------------
# harmonic
w1 = -np.sqrt(2.0)
w2 = -np.sqrt(2.0)
v1 = 1.0
v2 = 1.0
k1 = -m*np.power(w1,2)
k2 = -m*np.power(w2,2)
l1 = m*np.power(v1,2)
l2 = m*np.power(v2,2)
def ffharm2(q):
  return 0.5 * (k1 * np.power(q[0],2) + k2 * np.power(q[1],2)), -k1 * q[0], -k2 * q[1]
# double double well
def ffdw2(q):
  return k1 * q[0]**2 + k2 * q[1]**2 + l1 * q[0]**4 + l2 * q[1]**4, -(2.0 * k1 * q[0] + 4.0 * l1 * q[0]**3), -(2.0 * k2 * q[1] + 4.0 * l2 * q[1]**3)
# double polynomial
def ffpoly2(q, px, py):
  return (px[0] * q[0]**2 + px[1] * q[0]**3 + px[2] * q[0]**4 + px[3] * q[0]**6) + (py[0] * q[1]**2 + py[1] * q[1]**3 + py[2] * q[1]**4 + py[3] * q[1]**6), -1.0 * (2.0 * px[0] * q[0] + 3.0 * px[1] * q[0]**2 + 4.0 * px[2] * q[0]**3 + 6.0 * px[3] * q[0]**5), -1.0 * (2.0 * py[0] * q[1] + 3.0 * py[1] * q[1]**2 + 4.0 * py[2] * q[1]**3 + 6.0 * py[3] * q[1]**5)


#================================================================================================
#================================================================================================
#================================================================================================
# IMF (1D)
#================================================================================================
#================================================================================================
#================================================================================================

fqrms = 4.0 # spacing of sampling points in terms of RMS displacements
nint = 4001 # integration points for numerical integration of Hamiltonian matrix elements
nbasis = 25 # number of SHO states used as basis for anharmonic wvfn
nenergy = 10.0 # potential mapped until v - v_eq > nenergy * ehar
wthresh = 1e-3

# define basis functions
hf = []
for i in xrange(4*nbasis):
  hf.append(hermite(i))

def psi(n,mass,freq,x):
    nu = mass * freq / hbar
    norm = (nu/np.pi)**0.25 * np.sqrt(1.0/(2.0**(n)*np.math.factorial(n)))
    psival = norm * np.exp(-nu * x**2 /2.0) * hf[n](np.sqrt(nu)*x)
    return psival
 
def imf1d(qeq, K, beta, param):

  # set potential
  #if (fmode == 'dw'):
  #  ff = ffdw
  #elif (fmode == 'harm'):
  #  ff = ffharm
  def ff(x): return ffpoly(x, param)

  # evaluate harmonic free energy as reference
  Ahar = Aharm(np.abs(K), beta)

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


    qfs = []
    vfs = []
    vtotfs = []
    for i in range(len(qs)):
      qfs.append(qs[i])
      vfs.append(vs[i])
      vtotfs.append(vtots[i])
      qfs.append(qs[i] + 0.05*dq)
      vfs.append(vs[i] - 0.05*dq*fs[i])
      vtotfs.append(vtots[i] - 0.05*dq*fs[i])
      qfs.append(qs[i] - 0.05*dq)
      vfs.append(vs[i] + 0.05*dq*fs[i])
      vtotfs.append(vtots[i] + 0.05*dq*fs[i])

    # Fit the potential with a cubic spline
    vspline = interp1d(np.asarray(qfs), np.asarray(vfs), kind='cubic', bounds_error=False)
    vtotspline = interp1d(np.asarray(qfs), np.asarray(vtotfs), kind='cubic', bounds_error=False)
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


    qfs = []
    vfs = []
    vtotfs = []

    for i in range(len(qs)):
      qfs.append(qs[i])
      vfs.append(vs[i])
      vtotfs.append(vtots[i])
      qfs.append(qs[i] + 0.05*dq)
      vfs.append(vs[i] - 0.05*dq*fs[i])
      vtotfs.append(vtots[i] - 0.05*dq*fs[i])
      qfs.append(qs[i] - 0.05*dq)
      vfs.append(vs[i] + 0.05*dq*fs[i])
      vtotfs.append(vtots[i] + 0.05*dq*fs[i])

    # Fit the potential with a cubic spline
    vspline = interp1d(np.asarray(qfs), np.asarray(vfs), kind='cubic', bounds_error=False)
    vtotspline = interp1d(np.asarray(qfs), np.asarray(vtotfs), kind='cubic', bounds_error=False)
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
  #np.savetxt('pot_anh.map.dat',np.c_[qtots,vs])
  #np.savetxt('pot_tot.map.dat',np.c_[qtots,vtots])
  #np.savetxt('pot_anh.fit.dat',np.c_[dddq,vspline(dddq)])
  #np.savetxt('pot_tot.fit.dat',np.c_[dddq,vtotspline(dddq)])
  # write out wvfn
  #np.savetxt('psi_gs.dat',np.c_[dddq-qeqshift,psi(0,m,whar,dddq-qeqshift)])
  #np.savetxt('psi.dat',np.c_[dddq-qeqshift, np.sum(np.asarray([psi(jj,m,whar,dddq-qeqshift) * evecs.T[0][jj] for jj in xrange(nbasis)]),axis=0) ])
  # write convergence to logfile
  #np.savetxt('log.imff.'+fmode+'.'+str(qeq)+'.'+str(K)+'.'+str(beta)+'.dat',np.c_[qmins,qmaxs,npoints,Ahars,Aanhs,Adiffs], fmt='%6.3f %6.3f %3i % 12.6f % 12.6f % 12.6f')


  # Print 2-body energies E for debugging
  #print E
  #for i in range(10):
  #  print evals[i]

  #E = np.zeros((10,10))
  #for s0 in xrange(10):
  #  for s1 in xrange(10):
  #    E[s0][s1] = evals[s0] + evals[s1]
  #print E[0][0]

  # Calculate total partition function and free energy for current iteration iiter
  #Z = np.sum(np.exp(-beta*E))
  #A = -np.log(Z)/beta

  # done
  return Ahars[-1],Aanhs[-1],Adiffs[-1]
  


#================================================================================================
#================================================================================================
#================================================================================================
# IMF (2D)
#================================================================================================
#================================================================================================
#================================================================================================

fqrms = 0.5 # spacing of sampling points in terms of RMS displacements
nint = 101 # integration points for numerical integration of Hamiltonian matrix elements
nbasis = 10 # number of SHO states used as basis for anharmonic wvfn
nenergy = 10.0 # potential mapped until v - v_eq > nenergy * ehar
wthresh = 1e-3

def imf2d(qeq, K, beta, parx, pary):

  # initialise harmonic equilibrium positions and Hessian
  #qeq = np.zeros(2)
  #K = np.ones(2)

  # set potential
  def ff(q): return ffpoly2(q, parx, pary)

  # evaluate harmonic free energy as reference
  Ahar = Aharm(np.abs(K[0]), beta) + Aharm(np.abs(K[1]), beta)

  # evaluate initial harmonic vibrational energy
  whar = np.zeros(len(qeq))
  for i in range(len(K)):
    if(K[i] > 0):
      whar[i] = np.sqrt(K[i]/m)
    else:
      whar[i] = np.sqrt(-K[i]/m)

  # evaluate harmonic RMS displacement
  qrms = np.zeros(len(qeq))
  dq = np.zeros(len(qeq))
  betathresh = 1e6
  for i in range(len(qeq)):
    if (beta > betathresh):
      qrms[i] = np.sqrt(0.5/whar[i]**2)
    else:
      qrms[i] = np.sqrt(1/(np.exp(beta*whar[i]**2)-1)+0.5)/whar[i]
    dq[i] = fqrms * qrms[i]

  # map the potential
  ehar = np.zeros(len(qeq))
  for i in range(len(qeq)):
    ehar[i] = 0.5*hbar*whar[i]/np.tanh(beta*0.5*hbar*whar[i])
  # map along 1D axes to find min and max displacements (until v > ne * ehar)
  npts = np.ones(len(qeq), dtype=int)
  f = np.zeros(len(qeq))
  qmax = np.zeros(len(qeq))
  qmin = np.zeros(len(qeq))
  for i in range(len(qeq)):
    j = 0
    while True:
      j += 1
      npts[i] += 1
      q = np.zeros(len(qeq))
      q[i] = j*dq[i]
      v,f[0],f[1] = ff(qeq+q)
      if((v - ff(qeq)[0]) > nenergy*ehar[i]): 
        qmax[i] = j*dq[i]
        break
    j = 0
    while True:
      j += 1
      npts[i] += 1
      q = np.zeros(len(qeq))
      q[i] = -j*dq[i]
      v,f[0],f[1] = ff(qeq+q)
      if ((v - ff(qeq)[0]) > nenergy*ehar[i]): 
        qmin[i] = -j*dq[i]
        break

  if (min(npts) <= 3):
    print "Fewer than three sampling points along one axis -- cannot fit cubic spline"
    print edgar

  wanh = []
  wanh.append(whar)

  # actually map full 2D surface
  qs = np.zeros((npts[0]*npts[1],len(qeq)))
  qtots = np.zeros((npts[0]*npts[1],len(qeq)))
  vs = np.zeros(npts[0]*npts[1])
  vtots = np.zeros(npts[0]*npts[1])
  fs = np.zeros((npts[0]*npts[1],len(qeq)))

  k=-1
  q[0] = qmin[0]
  for i in range(npts[0]):
    q[1] = qmin[1]
    for j in range(npts[1]):
      k += 1
      v,f[0],f[1] = ff(qeq+q)
      # store positions, potentials, and forces in arrays
      qs[k] = q
      qtots[k] = qeq+q
      vs[k] = v - 0.5*m*((whar[0]*q[0])**2 + (whar[1]*q[1])**2) - ff(qeq)[0]
      vtots[k] = v - ff(qeq)[0]
      fs[k] = f
      # done storing
      q[1] += dq[1]
    q[0] += dq[0]

  # fit 2D cubic spline to sampled potential
  vtotspline = interp2d(np.asarray(qs.T[0]), np.asarray(qs.T[1]), np.asarray(vtots), kind='cubic', bounds_error=False)

#  # print fitted potential
#  for q0 in np.linspace(qmin[0],qmax[0],21):
#    for q1 in np.linspace(qmin[1],qmax[1],21):
#      print q0,q1,vtotspline(q0,q1)
#  # print sampled potential
#  iv=0
#  for i0 in range(npts[0]):
#    for i1 in range(npts[1]):
#      print np.asarray(qs.T[0])[iv], np.asarray(qs.T[1])[iv], np.asarray(vtots)[iv]
#      iv+=1
#  # bail out
#  print edgar

  #-------------------------------------------------------
  # solve the VSCF problem
  ########################### Assuming mean field generated by remnant modes according to their finite temperature mixed states
#  evecs = np.zeros((len(qeq),nbasis,nbasis))
#  evals = np.zeros((len(qeq),nbasis))
#  A = np.zeros(len(qeq))
#  # initially use harmonic solution 
#  for mode in range(len(qeq)):
#    for state in range(nbasis):
#      evecs[mode,state,state] = 1.0
#      evals[mode,state] = whar[mode] * (0.5 + state)
#    #A[mode] = -np.log(np.sum(np.exp(-beta*evals[mode])))/beta
#    #Atot = np.sum(A)
#    #Atotold = Atot
#    wanh = np.sum(whar)
#    wanhold = wanh
#  # SCF loop
#  iiter = 0
#  ddq = []
#  psicurr = []
#  rhocurr = []
#  for mode in range(len(qeq)):
#    ddq.append(np.linspace(qmin[mode],qmax[mode],nint))
#    Z = np.sum(np.asarray([np.exp(-beta*evals[mode,state]) for state in xrange(nbasis)]))
#    for state in range(nbasis):
#      psicurr.append(np.sum([psi(basis,m,whar[mode],ddq[mode])*evecs[mode][basis][state] for basis in xrange(nbasis)],axis=0))
#    rhocurr.append( np.sum(np.asarray( [ (psicurr[-1])**2*np.exp(-beta*evals[mode,state])/Z for state in xrange(nbasis)] ),axis=0) )
#
#  while True:
#    iiter += 1
#    # Calculate Hamiltonian matrix for each mode given current guess of independent mode eigenstates
#    hT = np.zeros((len(qeq),nbasis,nbasis))
#    AT = np.zeros(len(qeq))
#    wmft = np.zeros(len(qeq))
#
#    hT = np.zeros((len(qeq),nbasis,nbasis))
#    AT = np.zeros(len(qeq))
#    vmodeT = np.zeros((len(qeq),nint))
#    for k in range(nint):
#      dddq = ddq[0][k] 
#      vmodeT[0][k] = np.sum(np.asarray([ ((vtotspline(dddq,ddq[1][jj]) - 0.5 * (whar[1]*ddq[1][jj])**2) * rhocurr[1][jj] * (ddq[1][1]-ddq[1][0])) for jj in xrange(nint)]),axis=0)
#    for k in range(nint):
#      dddq = ddq[1][k]
#      vmodeT[1][k] = np.sum(np.asarray([ ((vtotspline(ddq[0][jj],dddq) - 0.5 * (whar[0]*ddq[0][jj])**2) * rhocurr[0][jj] * (ddq[0][1]-ddq[0][0])) for jj in xrange(nint)]),axis=0)
#
#    for mode in range(len(qeq)):
#      for i in xrange(nbasis):
#        for j in xrange(nbasis):
#          k = np.linspace(0,nint-1,nint,dtype=int)
#          dv = np.sum(psi(i,m,whar[mode],ddq[mode][k]) * (vmode[mode][k] - 0.5*m*(whar[mode]*ddq[mode][k])**2) * psi(j,m,whar[mode],ddq[mode][k])) * (ddq[mode][1] - ddq[mode][0])
#  
#          if (i == j):
#            hT[mode][i][j] = (i + 0.5) * hbar * whar[mode] + dv
#          else:
#            hT[mode][i][j] = dv
#
#      # Diagonalise Hamiltonian matrix and evaluate anharmonic free energy and vibrational freq
#      evals[mode], evecs[mode] = np.linalg.eigh(h[mode])
#      wmft[mode] = 2.0*evals[mode][0]/hbar

  ########################### Independent MFT calculation for every possible set of independent mode eigenstates to generate correct partition function
  evecs = np.zeros((len(qeq),nbasis,nbasis,nbasis))
  evals = np.zeros((len(qeq),nbasis,nbasis))
  E = np.zeros((nbasis,nbasis))
  # initially use harmonic solution 
  for s0 in range(nbasis):
    for s1 in range(nbasis):
      evecs[0,s1,s0,s0] = 1.0
      evals[0,s1,s0] = whar[0] * (0.5 + s0)
      evecs[1,s0,s1,s1] = 1.0
      evals[1,s0,s1] = whar[1] * (0.5 + s1)
      E[s0][s1] = whar[0] * (0.5 + s0) + whar[1] * (0.5 + s1)

  Z = np.sum(np.exp(-beta*E)) # harmonic partition function
  A = -np.log(Z)/beta # harmonic free energy
  A += ff(qeq)[0]
  Aold = A

  ddq = []
  for mode in range(len(qeq)):
    ddq.append(np.linspace(qmin[mode],qmax[mode],nint))


  vtotgrid = np.asarray([np.asarray([vtotspline(x1,x2) for x1 in ddq[0]]) for x2 in ddq[1]]).reshape((nint,nint))

  iiter = 0
  while True:
    iiter += 1

    # Construct Hamiltonian matrices
    h = np.zeros((len(qeq),nbasis,nbasis))
    vmode = np.zeros((len(qeq),nbasis,nbasis,nint))
    psicurr = np.zeros((len(qeq),nbasis,nint))
    for s0 in xrange(nbasis):
      for s1 in xrange(nbasis):

        #print "Calc. ",s0,"/",nbasis," ",s1,"/",nbasis

        # Wvfn of mode 1 in state s1 given that mode 0 is in state s0
        psitmp = np.sum([psi(basis,m,whar[1],ddq[1])*evecs[1][s0][s1][basis] for basis in xrange(nbasis)],axis=0)
        #normtmp = np.sum(np.asarray([ psitmp[jj]**2 for jj in xrange(nint)]),axis=0)
        normtmp = np.dot(psitmp,psitmp)
        # MF potential that mode 0 lives in given that mode 1 is in state s1 with wvfn psitmp
        vmode[0][s1][s0] = np.dot(vtotgrid.T,psitmp**2)/normtmp
        kqeq = np.argmin(ddq[0]**2)
        vmode[0][s1][s0] -= np.ones(nint) * vmode[0][s1][s0][kqeq]

        # Wvfn of mode 0 in state s0 given that mode 1 is in state s1
        psitmp = np.sum([psi(basis,m,whar[0],ddq[0])*evecs[0][s1][s0][basis] for basis in xrange(nbasis)],axis=0)
        #normtmp = np.sum(np.asarray([ psitmp[jj]**2 for jj in xrange(nint)]),axis=0)
        normtmp = np.dot(psitmp,psitmp)
        # MF potential that mode 1 lives in given that mode 0 is in state s0 with wvfn psitmp
        vmode[1][s0][s1] = np.dot(vtotgrid,psitmp**2)/normtmp
        kqeq = np.argmin(ddq[1]**2)
        vmode[1][s0][s1] -= np.ones(nint) * vmode[1][s0][s1][kqeq]

        # mode 0 in state s0 and mode 1 in state s1
        for i in xrange(nbasis):
          for j in xrange(nbasis):

            k = np.linspace(0,nint-1,nint,dtype=int)
            # mode 1 MFT Hamiltonian
            dv0 = np.sum(psi(i,m,whar[0],ddq[0][k]) * (vmode[0][s1][s0][k] - 0.5*m*(whar[0]*ddq[0][k])**2) * psi(j,m,whar[0],ddq[0][k])) * (ddq[0][1] - ddq[0][0])
            dv1 = np.sum(psi(i,m,whar[1],ddq[1][k]) * (vmode[1][s0][s1][k] - 0.5*m*(whar[1]*ddq[1][k])**2) * psi(j,m,whar[1],ddq[1][k])) * (ddq[1][1] - ddq[1][0])
            if (i == j):
              h[0][i][j] = (i + 0.5) * hbar * whar[0] + dv0
              h[1][i][j] = (i + 0.5) * hbar * whar[1] + dv1
            else:
              h[0][i][j] = dv0
              h[1][i][j] = dv1

        # Diagonalise Hamiltonian matrix given mode 0 in state s0 and mode 1 in state s1
        evals[0][s1], evecs[0][s1] = np.linalg.eigh(h[0])
        evals[1][s0], evecs[1][s0] = np.linalg.eigh(h[1])

        psicurr[0][s1] = np.sum([psi(basis,m,whar[0],ddq[0])*evecs[0][s1][s0][basis] for basis in xrange(nbasis)],axis=0)
        psicurr[1][s0] = np.sum([psi(basis,m,whar[1],ddq[1])*evecs[1][s0][s1][basis] for basis in xrange(nbasis)],axis=0)
        norm = np.dot(psicurr[0][s1],psicurr[0][s1]) * np.dot(psicurr[1][s0],psicurr[1][s0])

        dE = np.dot(psicurr[0][s1]**2,np.dot((vtotgrid.T - np.tile(vmode[0][s1][s0],(nint,1)).T - np.tile(vmode[1][s0][s1],(nint,1))),psicurr[1][s0]**2))
        dE /= norm

        E[s0][s1] = evals[0][s1][s0] + evals[1][s0][s1] + dE

        #if (s0==1):
        #  if (s1==1):
        #    for k in [45,46,47,48,49,50,51,52,53,54,55]:
        #      for l in [45,46,47,48,49,50,51,52,53,54,55]:
        #        print ddq[0][k],ddq[1][l],(vtotspline(ddq[0][k],ddq[1][l]) - vmode[0][s1][s0][k] - vmode[1][s0][s1][l])
        #    print edgar


        # Print 2-body energies E for debugging
        #print s0,s1,evals[0][s1][s0],evals[1][s0][s1],dE,E[s0][s1]

    # Print 2-body energies E for debugging
    #print E
    s0 = 0
    s1 = 0
    basis = nbasis - 1
    #print E[s0][s1]
    print evecs[0][s1][s0][basis],evecs[1][s0][s1][basis]

    # Calculate total partition function and free energy for current iteration iiter
    Z = np.sum(np.exp(-beta*E))
    A = -np.log(Z)/beta
    A += ff(qeq)[0]

    print "Iteration ",iiter, "Anh. free energy: ",A
    if (abs(A - Aold)/abs(Aold) < 1e-3): break
    Aold = A

    ######################################################


  return

  

#================================================================================================
#================================================================================================
#================================================================================================
# MAIN
#================================================================================================
#================================================================================================
#================================================================================================

#parser = argparse.ArgumentParser()
#parser.add_argument("pot", help="type of potential [harm, dw, morse]", type=str)
#parser.add_argument("qeq", help="equilibrium position (harm approx)", type=float)
#parser.add_argument("Keq", help="equilibrium Hessian (harm approx)", type=float)
#parser.add_argument("invT", help="inverse temperature", type=float)
#args = parser.parse_args()
#print imf1d(args.pot,args.qeq,args.Keq,args.invT)

parser = argparse.ArgumentParser()

parser.add_argument("px1", help="par px1 of the ffpoly2", type=float)
parser.add_argument("px2", help="par px2 of the ffpoly2", type=float)
parser.add_argument("px3", help="par px3 of the ffpoly2", type=float)
parser.add_argument("px4", help="par px4 of the ffpoly2", type=float)
parser.add_argument("py1", help="par py1 of the ffpoly2", type=float)
parser.add_argument("py2", help="par py2 of the ffpoly2", type=float)
parser.add_argument("py3", help="par py3 of the ffpoly2", type=float)
parser.add_argument("py4", help="par py4 of the ffpoly2", type=float)
parser.add_argument("invT", help="inverse temperature", type=float)
args = parser.parse_args()

px = np.asarray([args.px1, args.px2, args.px3, args.px4])
py = np.asarray([args.py1, args.py2, args.py3, args.py4])
invT = args.invT

qxtr = np.linspace(-10,10,1e4)
qxeq = qxtr[np.argmin(ffpoly(qxtr, px)[0])]
Kxeq = 2 * px[0] + 6 * px[1] * qxeq + 12 * px[2] * qxeq**2 + 30 * px[3] * qxeq**4

qytr = np.linspace(-10,10,1e4)
qyeq = qytr[np.argmin(ffpoly(qytr, py)[0])]
Kyeq = 2 * py[0] + 6 * py[1] * qyeq + 12 * py[2] * qyeq**2 + 30 * py[3] * qyeq**4

qxeq = 0.0
qyeq = 0.0
Kxeq = 2 * px[0] + 6 * px[1] * qxeq + 12 * px[2] * qxeq**2 + 30 * px[3] * qxeq**4
Kyeq = 2 * py[0] + 6 * py[1] * qyeq + 12 * py[2] * qyeq**2 + 30 * py[3] * qyeq**4

print "1D results"
print imf1d(qxeq, Kxeq, invT, px)
print imf1d(qyeq, Kyeq, invT, py)

print "2D result"
print imf2d([qxeq, qyeq], [Kxeq, Kyeq], invT, px, py)

