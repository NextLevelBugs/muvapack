import numpy as np
from numpy.linalg import pinv, matrix_rank
from scipy.stats import chi2,t

class GLM:
    r"""
    This implements the General Linear Model y= X b + e and a covariance
    structure for the error Cov[e] = \sigma^2 W where $W$ is known.
    The parameter vector b is estimated using the BLUE, predictions are made using the
    BLUP. This works in a general framework as long as second error moments exist. The model
    also provides confidence intervals and hypothesis tests, which assume a Gaussian error model.
    """

    def __init__(self,y,X,W):
        """
        Creates a new GLM with observations y, design matrix X and Covariance matrix W
        y: [n], X: [n,k], W:[n,n] - all numpy arrays
        """
        self.y = y
        self.X = X
        self.W = W

        # dimensional parameters
        self.n = y.shape[0]
        self.k = X.shape[1]

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
        self.e = self.yhat - self.y
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
        """
        # TODO