"""
This is the drive code for the model predictive controller -

Unconstrained Model Predictive Control Implementation in Python 
- This version is without an observer, that is, it assumes that the
- the state vector is perfectly known

Tutorial page that explains how to derive the algorithm is given here:
https://aleksandarhaber.com/model-predictive-control-mpc-tutorial-1-unconstrained-formulation-derivation-and-implementation-in-python-from-scratch/
    


@author: Aleksandar Haber
Date: September 2023


"""
import numpy as np
import matplotlib.pyplot as plt

from functionMPC import systemSimulate
from ModelPredictiveControl import ModelPredictiveControl


###############################################################################
#  Define the MPC algorithm parameters
###############################################################################
# prediction horizon
f= 20
# control horizon 
v=20

###############################################################################
# end of MPC parameter definitions
###############################################################################


###############################################################################
# Define the model
###############################################################################

# masses, spring and damper constants
m1=2  ; m2=2   ; k1=100  ; k2=200 ; d1=1  ; d2=5; 
# define the continuous-time system matrices
Ac=np.matrix([[0, 1, 0, 0],
              [-(k1+k2)/m1 ,  -(d1+d2)/m1 , k2/m1 , d2/m1 ],
              [0 , 0 ,  0 , 1], 
              [k2/m2,  d2/m2, -k2/m2, -d2/m2]])
Bc=np.matrix([[0],[0],[0],[1/m2]])
Cc=np.matrix([[1, 0, 0, 0]])

r=1; m=1 # number of inputs and outputs
n= 4 # state dimension


###############################################################################
# end of model definition
###############################################################################

###############################################################################
# discretize and simulate the system step response
###############################################################################

# discretization constant
sampling=0.05

# model discretization
I=np.identity(Ac.shape[0]) # this is an identity matrix
A=np.linalg.inv(I-sampling*Ac)
B=A*sampling*Bc
C=Cc

# check the eigenvalues
eigen_A=np.linalg.eig(Ac)[0]
eigen_Aid=np.linalg.eig(A)[0]

timeSampleTest=200

# compute the system's step response
inputTest=10*np.ones((1,timeSampleTest))
x0test=np.zeros(shape=(4,1))


# simulate the discrete-time system 
Ytest, Xtest=systemSimulate(A,B,C,inputTest,x0test)


plt.figure(figsize=(8,8))
plt.plot(Ytest[0,:],linewidth=4, label='Step response - output')
plt.xlabel('time steps')
plt.ylabel('output')
plt.legend()
plt.savefig('stepResponse.png',dpi=600)
plt.show()

###############################################################################
# end of step response
###############################################################################

###############################################################################
# form the weighting matrices
###############################################################################

# W1 matrix
W1=np.zeros(shape=(v*m,v*m))

for i in range(v):
    if (i==0):
        W1[i*m:(i+1)*m,i*m:(i+1)*m]=np.eye(m,m)
    else:
        W1[i*m:(i+1)*m,i*m:(i+1)*m]=np.eye(m,m)
        W1[i*m:(i+1)*m,(i-1)*m:(i)*m]=-np.eye(m,m)

# W2 matrix
Q0=0.0000000011
Qother=0.0001

W2=np.zeros(shape=(v*m,v*m))

for i in range(v):
    if (i==0):
        W2[i*m:(i+1)*m,i*m:(i+1)*m]=Q0
    else:
        W2[i*m:(i+1)*m,i*m:(i+1)*m]=Qother

# W3 matrix        
W3=np.matmul(W1.T,np.matmul(W2,W1))

# W4 matrix
W4=np.zeros(shape=(f*r,f*r))

# in the general case, this constant should be a matrix
predWeight=10

for i in range(f):
    W4[i*r:(i+1)*r,i*r:(i+1)*r]=predWeight
###############################################################################
# end of step response
###############################################################################

###############################################################################
# Define the reference trajectory 
###############################################################################

timeSteps=300

# here you need to comment/uncomment the trajectory that you want to use

# exponential trajectory
# timeVector=np.linspace(0,100,timeSteps)
#desiredTrajectory=np.ones(timeSteps)-np.exp(-0.01*timeVector)
#desiredTrajectory=np.reshape(desiredTrajectory,(timeSteps,1))

# pulse trajectory
desiredTrajectory=np.zeros(shape=(timeSteps,1))
desiredTrajectory[0:100,:]=np.ones((100,1))
desiredTrajectory[200:,:]=np.ones((100,1))

# step trajectory

#desiredTrajectory=0.3*np.ones(shape=(timeSteps,1))

###############################################################################
# end of definition of the reference trajectory 
###############################################################################

###############################################################################
# Simulate the MPC algorithm and plot the results
###############################################################################

# set the initial state
x0=x0test

# create the MPC object

mpc=ModelPredictiveControl(A,B,C,f,v,W3,W4,x0,desiredTrajectory)

# simulate the controller

for i in range(timeSteps-f):
    mpc.computeControlInputs()
    
    

# extract the state estimates in order to plot the results
desiredTrajectoryList=[]
controlledTrajectoryList=[]
controlInputList=[]
for j in np.arange(timeSteps-f):
    controlledTrajectoryList.append(mpc.outputs[j][0,0])
    desiredTrajectoryList.append(desiredTrajectory[j,0])
    controlInputList.append(mpc.inputs[j][0,0])

# plot the results
    
plt.figure(figsize=(8,8))
plt.plot(controlledTrajectoryList,linewidth=4, label='Controlled trajectory')
plt.plot(desiredTrajectoryList,'r', linewidth=2, label='Desired trajectory')
plt.xlabel('time steps')
plt.ylabel('Outputs')
plt.legend()
plt.savefig('controlledOutputsPulse.png',dpi=600)
plt.show()


plt.figure(figsize=(8,8))
plt.plot(controlInputList,linewidth=4, label='Computed inputs')
plt.xlabel('time steps')
plt.ylabel('Input')
plt.legend()
plt.savefig('inputsPulse.png',dpi=600)
plt.show()






















