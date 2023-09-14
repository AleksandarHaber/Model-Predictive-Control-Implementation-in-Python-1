###############################################################################
# This function simulates an open loop state-space model:
# x_{k+1} = A x_{k} + B u_{k}
# y_{k}   = C x_{k}
# starting from an initial condition x_{0}

# Input parameters:
# A,B,C - system matrices 
# U - the input matrix, its dimensions are \in \mathbb{R}^{m \times simSteps},  where m is the input vector dimension
# Output parameters:
# Y - simulated output - dimensions \in \mathbb{R}^{r \times simSteps}, where r is the output vector dimension
# X - simulated state - dimensions  \in \mathbb{R}^{n \times simSteps}, where n is the state vector dimension

def systemSimulate(A,B,C,U,x0):
    import numpy as np
    simTime=U.shape[1]
    n=A.shape[0]
    r=C.shape[0]
    X=np.zeros(shape=(n,simTime+1))
    Y=np.zeros(shape=(r,simTime))
    for i in range(0,simTime):
        if i==0:
            X[:,[i]]=x0
            Y[:,[i]]=np.matmul(C,x0)
            X[:,[i+1]]=np.matmul(A,x0)+np.matmul(B,U[:,[i]])
        else:
            Y[:,[i]]=np.matmul(C,X[:,[i]])
            X[:,[i+1]]=np.matmul(A,X[:,[i]])+np.matmul(B,U[:,[i]])
    
    return Y,X

###############################################################################
# end of function
###############################################################################