"""
This script contains functions to obtain samples of an fBm, approximate solutions to quasilinear SDEs driven by an fBM using four different methods and evaluating the rate of convergence of these methods
"""

import numpy as np
import matplotlib.pyplot as plt

"""
The three functions below generate two samples of an fBm according to circulant embedding. See Section 6.5 of 'An introduction to computational stochastic PDEs' by G.J. Lord, C.E. Powell and T. Shardlow.

H: Hurst parameter
n: amount increments
dt: time step of a single increment
"""

# Autocovariance function of fBm increments of length dt
def fBm_autocov(n,dt,H):
    return(0.5*dt**(2*H)*(abs(n+1)**(2*H)+abs(n-1)**(2*H)-2*abs(n)**(2*H)))

# Generate circulant embedding vector of v and obtain corresponding diagonal d
def circ_embedding(v):
    v_tilde=np.concatenate([v,v[len(v)-2:0:-1]])
    d=len(v_tilde)*np.fft.ifft(v_tilde)
    return(d)

# Generate sample of fBm by either
def fBm_sample(n,dt=None,H=None,d=None):
    if type(d)==type(None):
        d=circ_embedding(fBm_autocov(np.arange(n),dt,H))

    K=len(d)
    xi=np.random.normal(size=K)+np.random.normal(size=K)*1j
    Z=(np.fft.fft(np.sqrt(d)*xi)/np.sqrt(K))[0:n]

    I1,I2=np.real(Z),np.imag(Z)
    B1,B2=np.cumsum(I1),np.cumsum(I2)
    return(np.append(0,B1),np.append(0,B2))

"""
The four functions below encode the four numerical methods discussed in the paper.


H: Hurst parameter
T: final time
n: amount of approximation steps
B: fBm sample of size n+1 on a uniform grid of [0,T] with mesh size T/n
alpha: drift parameter (see paper)
beta: diffusion parameter (see paper)
a: drift function (see paper) with input (t,x)
x0: initial value
"""

def GBMEM(B,H,T,n,alpha,beta,a,x0):
    dt=T/n
    C=0.5*beta**2*dt**(2*H)
    Z=x0
    for i in range (0,n):
        J=np.exp(-alpha*i*dt+C*(n**(2*H)-(n-i)**(2*H))-beta*B[i])
        Z+=dt*J*(a(i*dt,Z/J))
    J=np.exp(-alpha*T+C*n**(2*H)-beta*B[n])
    X=Z/J
    return(X)

def MishuraEM(B,H,T,n,alpha,beta,a,x0,*args):
    # If a function a_tilde of (t,x) is given as as optional argument, that one is used as drift function a
    if args!=():
        a=args[0]

    dt=T/n
    C=0.5*beta**2*dt**(2*H)
    Z=x0
    for i in range (0,n):
        J=np.exp(C*(n**(2*H)-(n-i)**(2*H))-beta*B[i])
        Z+=dt*J*(a(i*dt,Z/J))
    J=np.exp(C*n**(2*H)-beta*B[n])
    X=Z/J
    return(X)

def ExpFreeze(B,H,T,n,alpha,beta,a,x0):
    dt=T/n
    C=0.5*beta**2*dt**(2*H)
    Z=x0
    for i in range (0,n):
        J=np.exp(-alpha*i*dt+C*(n**(2*H)-(n-i)**(2*H))-beta*B[i])
        Z=np.exp(dt*a(i*dt,Z/J)/(Z/J))*Z
    J=np.exp(-alpha*T+C*n**(2*H)-beta*B[n])
    X=Z/J
    return(X)

def Rosenbrock(B,H,T,n,alpha,beta,a,x0,a_prime):
    dt=T/n
    C=0.5*beta**2*dt**(2*H)
    Z=x0
    for i in range (0,n):
        J=np.exp(-alpha*i*dt+C*(n**(2*H)-(n-i)**(2*H))-beta*B[i])
        Z=Z*np.exp(dt*a_prime(i*dt,Z/J))+dt*(J*a(i*dt,Z/J)-Z*a_prime(i*dt,Z/J))
    J=np.exp(-alpha*T+C*n**(2*H)-beta*B[n])
    X=Z/J
    return(X)

"""
The following function approximates the path of an instance of the solution to the quasilinear SDE.

method: numerical method to be used from [GBMEM, MishuraEM, ExpFreeze, Rosenbrock]
Other arguments are as above
"""

def SDE_sample_path(B,H,T,n,alpha,beta,x0,a,method,*args):
    dt=T/n
    X=np.zeros(n+1)
    X[0]=x0
    for i in range (1,n+1):
        X[i]=method(B,H,i*dt,i,alpha,beta,a,x0,*args)
    return(X)

"""
The following function estimates the RMSE of one of the numerical methods by using a Monte Carlo method for several values of time step size.
GBMEM is used to obtain a reference solution.

M: Monte Carlo sample size. Need to be even.
dts: Array of values of time step sizes for which the RMSE needs to be determined. Needs to be chosen such that T/dts is integer valued
dt_ref: time step size of reference solution. Needs to be chosen such that dts/dt_ref is integer valued.

"""

def RMSE(H,T,alpha,beta,x0,a,method,M,dts,dt_ref,*args):
    n_ref=int(T/dt_ref)
    d=circ_embedding(fBm_autocov(np.arange(n_ref),dt_ref,H))
    S=np.zeros(len(dts))   # Variable that tracks the sum of square errors

    for i in range (0,int(M/2)):
        # Generating reference fBm
        B_ref1,B_ref2=fBm_sample(n_ref,d=d)

        # Determine reference solution
        X_ref1=GBMEM(B_ref1,H,T,n_ref,alpha,beta,a,x0)
        X_ref2=GBMEM(B_ref2,H,T,n_ref,alpha,beta,a,x0)

        for j in range(len(dts)):
            dt=dts[j]
            n=int(T/dt)
            B1=B_ref1[0::int((n_ref/n))]
            B2=B_ref2[0::int((n_ref/n))]
            X1=method(B1,H,T,n,alpha,beta,a,x0,*args)
            X2=method(B2,H,T,n,alpha,beta,a,x0,*args)
            S[j]+=(X_ref1-X1)**2+(X_ref2-X2)**2
    return(np.sqrt(S/M))


## Plot a sample of the numerical approximation to the quasilinear SDE for different numerical methods.
# Set nonlinear drift funtion a(t,x)
def a(t,x):
    return(4*x/(1+x**2))
# Set function for Rosenbrock approximation. Usually the derivative with respect to x of a(t,x).
def a_prime(t,x):
    return(4*(1-x**2)/(1+x**2)**2)

# Set parameters
H=0.25
alpha=1
beta=1
x0=1
T=1
n=1000

# Toggles for which methods to use
use_GBMEM=True
use_MishuraEM=False
use_ExpFreeze=False
use_Rosenbrock=False

t=np.linspace(0,T,n+1)
np.random.seed(0)
B=fBm_sample(n,T/n,H)[0]

if use_GBMEM:
    X=SDE_sample_path(B,H,T,n,alpha,beta,x0,a,GBMEM)
    plt.plot(t,X,label='GBMEM',linestyle='-')
if use_MishuraEM:
    a_tilde=lambda t,x: a(t,x)+alpha*x
    X=SDE_sample_path(B,H,T,n,alpha,beta,x0,a_tilde,MishuraEM)
    plt.plot(t,X,label='MishuraEM',linestyle='--')
if use_ExpFreeze:
    X=SDE_sample_path(B,H,T,n,alpha,beta,x0,a,ExpFreeze)
    plt.plot(t,X,label='ExpFreeze',linestyle=':')
if use_Rosenbrock:
    X=SDE_sample_path(B,H,T,n,alpha,beta,x0,a,Rosenbrock,a_prime)
    plt.plot(t,X,label='Rosenbrock',linestyle=':')

plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.legend()
plt.show()

## Create a log-log plot of the estimated RMSE versus different values of time step size
# Set nonlinear drift funtion a(t,x)
def a(t,x):
    return(4*x/(1+x**2))
# Set function for Rosenbrock approximation. Usually the derivative with respect to x of a(t,x).
def a_prime(t,x):
    return(4*(1-x**2)/(1+x**2)**2)

# Set parameters
H=0.25
alpha=1
beta=1
x0=1
T=1

dt_ref=T/2**19
dts=T/2**np.arange(6,16)
M=500

# Toggles for which methods to use
use_GBMEM=True
use_MishuraEM=True
use_ExpFreeze=True
use_Rosenbrock=True

if use_GBMEM:
    np.random.seed(0)
    X=RMSE(H,T,alpha,beta,x0,a,GBMEM,M,dts,dt_ref)
    plt.scatter(np.log(dts),np.log(X),label='GBMEM',marker='s')

    # Add a linear fit to estimate rate of convergence
    b,c=np.polyfit(np.log(dts),np.log(X),1)
    plt.plot(np.log(dts),b*np.log(dts)+c,linestyle='--')
    print('Linear fit, slope={:.3f}'.format(b))

if use_MishuraEM:
    np.random.seed(0)
    a_tilde=lambda t,x: a(t,x)+alpha*x
    X=RMSE(H,T,alpha,beta,x0,a,MishuraEM,M,dts,dt_ref,a_tilde)
    plt.scatter(np.log(dts),np.log(X),label='MishuraEM',marker='o')

if use_ExpFreeze:
    np.random.seed(0)
    X=RMSE(H,T,alpha,beta,x0,a,ExpFreeze,M,dts,dt_ref)
    plt.scatter(np.log(dts),np.log(X),label='ExpFreeze',marker='x')

if use_Rosenbrock:
    np.random.seed(0)
    X=RMSE(H,T,alpha,beta,x0,a,Rosenbrock,M,dts,dt_ref,a_prime)
    plt.scatter(np.log(dts),np.log(X),label='Rosenbrock',marker='*')


plt.xlabel('log $\Delta t$')
plt.ylabel('log RMSE')
plt.legend()
plt.show()


## Create a plot of the estimated rate of convergence and error constant for several values of H
# Set nonlinear drift funtion a(t,x)
def a(t,x):
    return(4*x/(1+x**2))
# Set function for Rosenbrock approximation. Usually the derivative with respect to x of a(t,x).
def a_prime(t,x):
    return(4*(1-x**2)/(1+x**2)**2)

# Set parameters
H=np.append(0.001,np.linspace(0.05,0.95,19))
alpha=1
beta=1
x0=1
T=1

dt_ref=T/2**19
dts=T/2**np.arange(6,16)
M=500

# Divide total sample in batches for error bars
batch_size=50
m=int(M/batch_size)

np.random.seed(0)

b=np.zeros((len(H),m))
c=np.zeros((len(H),m))
for i in range (0,len(H)):
    for j in range (0,m):
        X=RMSE(H[i],T,alpha,beta,x0,a,GBMEM,batch_size,dts,dt_ref)
        b[i,j],c[i,j]=np.polyfit(np.log(dts),np.log(X),1)

slope=np.mean(b,axis=1)
error=1.96*np.std(b,axis=1)
const=np.mean(c,axis=1)
error_const=1.96*np.std(c,axis=1)

plt.errorbar(H, slope, yerr=error,marker='o',linestyle='',capsize=2,label='Estimated rate of convergence')
plt.plot(np.append(H,1),np.append(H,1),linestyle=':',label='$H$')
plt.plot(np.append(H,1),[min(val,1) for val in np.append(H,1)+1/2],linestyle='--',label='$\min(H+1/2,1)$')
paras,cov=np.polyfit(H[H<=0.5],slope[H<=0.5],1,cov=True)
plt.plot(H[H<=1/2],np.polyval(paras,H[H<=1/2]),linestyle='-.',label='Linear fit, slope={:.3f}'.format(paras[0]),color='r')

plt.xticks(np.linspace(0,1,5))
plt.ylim((-0.05,1.25))
plt.xlabel('$H$')
plt.ylabel('Rate')
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,0,1,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.show()

# plt.errorbar(H, const, yerr=error_const,marker='o',linestyle='',capsize=2)
# plt.xlabel('$H$')
# plt.ylabel('Error constant')
# plt.xticks(np.linspace(0,1,5))
# plt.show()
