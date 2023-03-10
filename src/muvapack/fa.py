"""
Performing Factor Analysis
"""

import numpy as np
from scipy.stats import chi2

class FactorAnalysis:

    def __init__(self, Y, alpha=0.05, verbose=False):
        """
        Y: (n,k) n datapoints of dimension k
        alpha: significance level which determines the number of factors given a gaussian model
        The determined factors are centered and standardized.
        """
        self.n = Y.shape[0]
        self.k = Y.shape[1]
        self.Y = Y
        self.alpha = alpha
        # center the data
        self.mu = np.mean(Y,axis=0)
        self.Yc = self.Y - self.mu[None,:]
        # calculate the covariance matrix
        self.Cov = (self.Yc.T @ self.Yc)/self.n
        # we compute decompositions with more and more facotrs until we do not reject the hypothesis
        # that #facotrs explains the data with confidence alpha
        self.H = None
        if(self.k < 3):
            print("[ERROR] Factor analysis with less then 3 covariates does not work")
        if(self.k == 3):
            print("[WARNING] Factor analysis - with 3 covariates one can have only have 1 expl. factor")
        
        for fac in range(1,self.k+1):
            self.fac = fac
            H,Phi = self.compute_decompostion(fac)

            dof = int(((self.k-fac)**2-(self.k+fac))/2)
            if(dof < 0):
                break

            if(dof == 0):
                self.H = H
                self.Phi = Phi
                print("[WARNING] stopped model selection prematurely because of no available d.o.f.")
                break
            else:
                p = self.decompostion_level(H,Phi,fac)
                if(verbose):
                    print(f"{fac} explaining factors - p: {p}")
                if(p>alpha):
                    self.H = H
                    self.Phi = Phi
                    break
        
        if(self.H is None):
            print("[ERROR] Factor analysis - decomposition of covariance matrix failed")


    def compute_decompostion(self, p, epsilon=1e-6):
        """
        p: number of factors to use
        epsilon: smallest change to stop iteration
        return (H,Phi) H: factor loadings matrix and Phi diagonal matrix with the error variances
        """
        max_iter = 1000
        # decompose the covariance matrix
        Phi = np.diag(np.diagonal(self.Cov))
        Hold = self.estimate_H_given_Phi(Phi,p)
        Hnew = Hold
        success = False
        for i in range(max_iter):
            # estimate phi given H
            Phi = np.diag(np.diag(self.Cov - Hold.T @ Hold))
            Hnew = self.estimate_H_given_Phi(Phi,p)
            if(np.max(Hnew-Hold) < epsilon):
                success = True
                break
            else:
                Hold = Hnew
        
        if(not success):
            print("[WARNING] covariance matrix decomposition did not converge. Consider increasing max iterations or tolerance")
        return (Hnew,Phi)

    def estimate_H_given_Phi(self, Phi, p):
        irPhi = np.diag(1.0/np.sqrt(np.diag(Phi)))
        D = irPhi @ self.Cov @ irPhi
        spec, Gamma = np.linalg.eigh(D)
        sb = np.flip(np.sqrt(np.clip(spec-1,0.0,np.inf))[-p:])
        G = np.flip(Gamma[:,-p:],axis=1)
        H = np.diag(sb) @ G.T @ np.diag(np.sqrt(np.diag(Phi)))
        return H

    def decompostion_level(self, H, Phi, fac):
        """
        Given a decompositions Cov = H.T @ H + Phi and H having shape (k,p), calculate (using a gaussian model)
        the approximate p-value of observing Y when having a number fac of explaining factors
        This works better when whe have a large number >100 of datapoints because we use an asymptotic approximation
        """
        # we use wilks and the double negative log likelihood ratio
        aCov = H.T @ H + Phi
        dnllr = self.n * np.log(np.linalg.det(aCov)) + self.n * np.trace(self.Cov @ np.linalg.inv(aCov)) - self.n * np.log(np.linalg.det(self.Cov))-self.n*self.k
        dof = int(((self.k-fac)**2-(self.k+fac))/2)
        pval = chi2.sf(dnllr, df=dof)
        return pval

    def quartimax(self):
        """
        Rotates the loadings matrix to maximize the quartimax functional (sum of 4-th power H entries)
        """
        if(self.fac < 2):
            # no need to perform rotations if there are not at least 2 factors
            return self.H

        #TODO