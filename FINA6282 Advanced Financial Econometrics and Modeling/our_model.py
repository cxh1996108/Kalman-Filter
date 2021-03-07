import numpy as np
from numpy.random import random as rnd 
import pandas as pd
from numpy.linalg import inv
import math
from scipy.optimize import minimize

class parameters:
    def __init__(self,n):
        self.mu = rnd((n,1))
        self.F = rnd((n,n))
        self.r = rnd((1,1))
        self.R = self.r.T@self.r
        self.q = rnd((n,n))
        self.Q = self.q.T@self.q
        self.beta00 = rnd((n,1))
        self.p00 = rnd((n,n))
        self.P00 = self.p00.T@self.p00
        self.parnames = list(self.__dict__.keys())
        self.parnames.sort()
        self.do_not_pack = ['R','Q','P00']
    def pack(self):
        big_column = np.array([[]]).T
        for aname in self.parnames: 
            if aname in self.do_not_pack: continue
            matrix = getattr(self,aname) 
            matrix = matrix.reshape((-1,1),order = "F") 
            big_column = np.concatenate((big_column,matrix),axis = 0)
        return big_column
    def unpack(self, big_column):
        position = 0
        for aname in self.parnames:
            if aname in self.do_not_pack: continue
            matrix = getattr(self,aname) 
            nrow, ncol = matrix.shape
            new_position = position + nrow*ncol
            new_matrix = big_column[position : new_position]
            new_matrix = new_matrix.reshape((nrow, ncol), order = 'F')
            setattr(self, aname, new_matrix)
            position = new_position
        self.R = self.r.T@self.r
        self.Q = self.q.T@self.q
        self.P00 = self.p00.T@self.p00
            
class data:
    def __init__(self,n):
        self.y = rnd((1,1))
        self.x = rnd((1,n))
        
class prediction:
    def __init__(self,n):
        self.beta = rnd((n,1))
        self.P = rnd((n,n))
        self.eta = rnd((1,1))
        self.f = rnd((1,1))
        
class updating:
    def __init__(self,n):
        self.beta = rnd((n,1))
        self.P = rnd((n,n))
        self.K = rnd((n,1))
        
class a_period:
    def __init__(self,t,model):
        self.t = t
        self.mymodel = model
        n = self.mymodel.n
        self.data = data(n)
        self.prd = prediction(n)
        self.upd = updating(n)
    def predict(self):
        par = self.mymodel.pars
        mu = par.mu
        F = par.F
        if self.t == 0: 
            b = par.beta00
            P = par.P00
        else:
            previous_period = self.mymodel.sample[self.t-1]
            b = previous_period.upd.beta
            P = previous_period.upd.P
        self.prd.beta = mu + F @ b
        self.prd.P = (F @ P) @ F.T + par.Q
        x = self.data.x
        eta = self.data.y - x @ self.prd.beta
        f = (x @ self.prd.P) @ x.T + par.R
        self.prd.eta = eta
        self.prd.f = f
        self.ll = float( math.log(2 * math.pi * f) + (eta.T @ inv(f)) @ eta )
        
    def update(self):
        P = self.prd.P
        x = self.data.x
        K = (P @ x.T) @ inv(self.prd.f)
        self.upd.K = K
        self.upd.beta = self.prd.beta + K @ self.prd.eta
        self.upd.P = P - (K @ x) @ P
            
class the_model:
    def __init__(self,datafile):
        self.datafile = datafile
        df = pd.read_csv(datafile)
        T,n = df.shape
        n = n - 1
        self.n = n
        self.T = T
        self.pars = parameters(n)
        self.sample = []
        for t in range(T): 
            new_period = a_period(t,self)
            new_period.data.y = np.array([[ df.iloc[t,0] ]])
            new_period.data.x = df.iloc[t,1:].values.reshape((1,n))
            self.sample.append(new_period)
    def run(self):
        self.ll = 0
        for t in range(self.T):
            period_t = self.sample[t]
            period_t.predict()
            period_t.update()
            self.ll = self.ll + period_t.ll
        self.ll = -0.5 * self.ll
    def fun2min(self,column): # This is the function we will be minimizing.
        self.pars.unpack(column) # Unpack the column into the parameter matrices.
        self.run() # Run the Kalman filter with the new parameters.
        print (self.ll) # Report progress.
        return -self.ll # Return the negative of the lig likelihood.
    def F_constraint(self,column):
        self.pars.unpack(column) # Unpack the column into the parameter matrices.
        F = self.pars.F
        I = np.eye(F.shape[0])
        try: 
            an_inverse = inv(I-F)
            return 1
        except: return -1
    def estimate(self,tol=0.01, maxit = 500):
        response = 'Y'
        while response != 'N':
            starting_column = self.pars.pack()
            constr = {"type": "ineq", "fun": self.F_constraint}
            opt = {"maxiter": maxit}
            solution = minimize(
                fun = self.fun2min,
                x0 = starting_column,
                method = "COBYLA",
                constraints = constr,
                options = opt)
            self.optimum = solution
            self.pars.unpack(solution.x)
            self.run()
            print ('SUCCES: '+ str(solution.success) +': '+solution.message)
            response = input('Should I continue? (Y/N)')

        
        