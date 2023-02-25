"""
A General Linear Model implementation using the BLUE and BLUP.
"""

import numpy as np
from numpy.linalg import pinv, matrix_rank
from scipy.stats import chi2,t,f

class GLM:
    r"""
    This implements the General Linear Model y= X b + e and a covariance
    structure for the error Cov[e] = \sigma^2 W where $W$ is known.
    The parameter vector b is estimated using the BLUE, predictions are made using the
    BLUP. This works in a general framework as long as second error moments exist. The model
    also provides confidence intervals and hypothesis tests, which assume a Gaussian error model.
    """

    def __init__(self,y,X,W=None):
        """
        Creates a new GLM with observations y, design matrix X and Covariance matrix W
        y: [n], X: [n,k], W:[n,n] - all numpy arrays
        """
        self.y = y
        self.X = X
        # dimensional parameters
        self.n = y.shape[0]
        self.k = X.shape[1]
        if(W is None):
            self.W = np.eye(self.n)
        else:
            self.W = W

        if(self.W.shape[0] != self.n or self.W.shape[1] != self.n):
            print("[ERROR] can't create GLM with wrong covariance matrix shape.")
            print("[ERRO] must be of shape "+str(self.n)+"*"+str(self.n)+" to match y")
        
        if(self.X.shape[0] != self.n):
            print("[ERROR] can't create GLM with wrong design matrix shape.")
            print("[ERRO] must be of shape "+str(self.n)+"*"+str(self.n)+" to match y")
        
        # do the fit of the model
        self.fit()

    def fit(self):
        """
        Calculates the BLUE for the parameter vector beta [k] which the model assumes to be estimable
        """
        # orthorgonal projection onto the image of design matrix
        self.PX = self.X @ pinv(self.X.T @ self.X) @ self.X.T
        self.X_ = pinv(self.X) 
        # orthorgonal projection onto the orth. complement of the design matrix image
        self.CX = np.eye(self.n)-self.PX
        self.XBLUE = np.eye(self.n) - self.W @ self.CX @ pinv(self.CX @ self.W @ self.CX) @ self.CX
        # calculate the BLUE estimate
        self.beta = self.X_ @ self.XBLUE @ self.y
        #calculate the explainted part + residual error of beta
        self.yhat = self.X @ self.beta
        self.e = self.y - self.yhat
        # degres of freedom in the model
        self.dof = matrix_rank(self.W @ self.CX)
        # estimate the covariance prefactor and residual sum of squares
        self.rss = self.e @ pinv(self.W) @ self.e
        self.var = self.rss / self.dof
        self.sigma = np.sqrt(self.var)
        # compute the estimated covariance matrix of beta
        self.cov_beta = self.var * (self.X_ @ self.XBLUE @ self.W @ self.XBLUE.T @ self.X_.T)
    
    def get_sigma_ci(self,alpha):
        """
        Gives a confidence interval for the estimated standard deviation sigma
        where Cov[y] = sigma^2 * W i.e. the prefactor of the covariance structure
        This CI uses a chi^2-statistic and is exact under a Gaussian error model
        alpha: Confindence level, i.e. true sigma is in the interval with probability 1-alpha
        """
        CI = (chi2.ppf(alpha/2.0,df=self.dof),chi2.ppf(1.0-alpha/2.0,df=self.dof))
        # take sqrt since we are make a CI for sigma not the variance
        return (np.sqrt(self.rss/CI[1]),np.sqrt(self.rss/CI[0]))
    
    def param_individual_ci(self,alpha):
        """
        Gets confidence intervals for all parameters beta to level 1-alpha.
        Note that these are seperate one variable CIs for the different parameters
        There is no guarante that all parameters are in their respective CIs at level 1-alpha
        This is only guaranteed for each parameter seperately. For a multiparameter CI use the
        confidence cube.
        The CIs assume a Gaussian error model and use a t-statistic
        returns: [k,2] array with CI upper/lower bounds for each parameter 
        """
        CI = np.zeros((self.k,2))
        qlow = t.ppf(alpha/2.0,df=self.dof)
        qhigh = t.ppf(1.0-alpha/2.0,df=self.dof)
        for i in range(self.k):
            CI[i,0] = self.beta[i]+qlow*np.sqrt(self.cov_beta[i,i])
            CI[i,1] = self.beta[i]+qhigh*np.sqrt(self.cov_beta[i,i])
        return CI
    
    def param_confidence_cube(self,alpha):
        """
        Generates a confidence cube for all estimated parameters beta. The true parameter vector is contained
        within this cube with at least probability 1-alpha. For individual cis for the parameters use param_individual_ci.
        This is done using Bonferroni-correction.
        """
        return self.param_individual_ci(alpha/self.k)
    
    def test_linear_restriction(self,A,xi):
        """
        This returns a p-value based on ANOVA for the following hypothesis test.
        H_0: A @ beta = xi meaning the parameters fulfill a linear restriction
        H_1: They dont.
        Test statistic is given by (R_H^2-R_0^2)/(R_0^2) where R_0^2 is the RSS in the unrestricted model
        while R_H^2 is the RSS in the restricted model. Under Gaussian error model, the test statistic follows
        a F-distribution and H_0 should be rejected if its value is high.
        This methods allows to test whether or not any parameters have arbitrary values such as if beta_1 = 0
        A: [s,k] nparray
        xi: [s] nparray
        """
        # first, test if A @ beta = xi can be fulfilled by any beta at all
        tol = 1e-6
        xip = A @ pinv(A) @ xi
        if(np.max(np.abs(xip-xi))>tol):
            print("[ERROR] GLM cannot test a restriction that produces an empty parameter space")
            return None
        
        s = xi.shape[0]
        # make the restricted model
        yr = np.zeros((self.n+s))
        yr[:self.n] = self.y
        yr[self.n:] = xi
        Xr = np.zeros((self.n+s,self.k))
        Xr[:self.n,:] = self.X
        Xr[self.n:,:] = A
        Wr = np.zeros((self.n+s,self.n+s))
        Wr[:self.n,:self.n] = self.W
        # fit it
        rglm = GLM(yr,Xr,Wr)
        # how many degrees of freedom does the additional LZF add?
        m = matrix_rank(A @ self.cov_beta @ A.T)
        T = self.dof/m * (rglm.rss - self.rss)/(self.rss)
        p = 1.0 - f.cdf(T,m,self.dof)

        result = {"p-value": p, "T": T, "d.o.f": self.dof, "constraints:": m}

        return result

    def summary(self,covariate_names=None,alpha=0.05):
        """
        prints all the model results nicely with confidence intervals and p-values for
        H0: parameter #i is 0
        covariate_names: an ordered list of names of each covariate/explanatory variable (optional)
        alpha: confidence level for the CIs
        """
        if( (covariate_names is None) or len(covariate_names)!=self.k):
            covariate_names = ["variable "+str(i) for i in range(self.k)]
        
        # get CIs
        CI = self.param_individual_ci(alpha)

        #get p values for H0: beta[i] = 0
        pvals = []
        for i in range(self.k):
            Ak = np.zeros((1,self.k))
            Ak[0,i] = 1.0
            xi = np.zeros((1))
            pvals += [self.test_linear_restriction(Ak,xi)["p-value"]]

        # headline
        print("---    General Linear Model y=Xb+e    ---------------------------------------")

        # regressor estimates
        max_buf = max(np.max([len(covariate_names[i]) for i in range(self.k)]),20)
        print("explanatory variable"+" "*(max_buf-18) + "BLUE estimate    " + f"{(1-alpha)*100:.2f}% confidence interval   " + "p-Value   ")
        for i in range(self.k):
            ps = f"{self.beta[i]:.3}"
            cs = f"[{CI[i,0]:.3f},{CI[i,1]:.3f}]"
            vs = f"{pvals[i]:.3E}"
            print(covariate_names[i]+" "*(max_buf+2-len(covariate_names[i])) +ps +" "*max(0,17-len(ps)) + cs + " "*max(0,29-len(cs))+ vs)

        # residual error sum of squares and variance
        ps = f"{self.sigma:.3f}"
        CI = self.get_sigma_ci(alpha)
        cs = f"[{CI[0]:.3f},{CI[1]:.3f}]"
        print("Model Error Stdev."+" "*(max_buf-16) + ps + " "*max(0,17-len(ps))+cs+ " "*max(0,29-len(cs))+"  - - -")
        #bottom
        print("*Confidence intervals and p-values assume a Gaussian error model.")
        print(" "*4+f"Residual Sum of Squares: {self.rss:.3F} " + 5*" " + f"  Degrees Of Freedom: {self.dof}" )
        print("-----------------------------------------------------------------------------")

    def predict(self, x0, alpha = 0.05, v0=None):
        """
        This method used the BLUP (Best Linear Unbiased Predictor) to predict the expectation value of a new
        observation y0 given a design vector of x0 and possible covariance v0 to all previous measurements.
        x0: design of the measurement whose true value shall be predicted [k]
        v0: None - no correlation with prev. observations. Otherwise v0: vector of covariance Cov[y0,y] of shape [n]
        alpha: confidence level 1-alpha for the confidence interval for y0
        Note: The confidence interval and predicted value for y0 contain the true expected value for y (i.e. without the error) with
        probability 1-alpha under a Gaussian error model. Notice that they must not contain the observed value y0 with probablity y0, as
        the observation of y0 comes with an extra model error.
        returns: dictionary with {"y": prediction value for y0, "ci": 1-alpha confidence interval for y (tuple)}
        """
        if(v0 is None):
            v0 = np.zeros(self.n)
        # calculate the BLUP
        y_blup = x0 @ self.beta + v0 @ pinv(self.W) @ self.e / self.var
        # and the confidence interval width
        c = t.ppf(1.0-alpha/2.0,df = self.dof) * np.sqrt(x0 @ self.cov_beta @ x0)
        result = {"y": y_blup, "ci": (y_blup-c,y_blup+c)}
        return result
