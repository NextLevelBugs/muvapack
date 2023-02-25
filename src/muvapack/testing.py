"""
A collection of hypothesis tests
"""

import numpy as np
from scipy.stats import chi2,f
from numpy.linalg import pinv

def pearson_goft(N,q):
    """
    The Pearson Goodness Of Fit Test (special case of Pearson Chi Square). Suppose we have a random variable which can take
    k different values each with probability q[1],..,q[k]. We observe N[1],..,N[k] i.i.d.
    observations of said values. The null hypothesis is, that observations follow the distributions
    specified by the q probablities. The pearson GOFT considers a test statistic that is asymptotically
    chi^2 distributed with k degrees of freedom.
    N: np.array of shape (k) integer values
    q: np.array of shape (k) values between 0 and 1, sum to 1
    returns: {"p-value": p-value, "chi2": test statistic value, "df": degrees of freedom}
    """
    n = N.shape[0]
    if(q.shape[0] != n):
        print("[ERROR] The number of categories has to correspond to the number of probabilites provided.")
        return None
    df = n-1
    pts = np.sum(N)
    T = np.sum((N-q*pts)**2/(q*pts))
    p = chi2.sf(T,df=df)
    return {"p-value": p, "chi2": T, "df": df}

def pearson_indepence(N):
    """
    The sppecial case of the pearson chi^2 test, for independence of categorical data. Suppose we have two
    properties A and B with Categories A_1,..,A_n and B_1,..,B_m. Then let N[a,b] denote the number of observations
    with A=a and B=b. The null hypothesis to test is that both properties are statistically independent.
    N: np.array of shape (n,m) integer values
    returns: {"p-value": p-value, "chi2": test statistic, "df": degrees of freedom}
    """
    df = (N.shape[0]-1)*(N.shape[1]-1)
    if(df < 1):
        print("[ERROR] No degrees of freedom available. Distribution is already fixed.")
        return None
    pts = np.sum(N)
    m1 = np.sum(N,axis=1)/pts
    m2 = np.sum(N,axis=0)
    w = np.outer(m1,m2)
    T = np.sum((N-w)**2/w)
    p = chi2.sf(T,df=df)
    return {"p-value": p, "chi2": T, "df": df}

def hotelling(X,Y):
    """
    Hotelling T^2 Test for two multivariate normal samples in dimension d
    X: [m1,d]
    Y: [m2,d]
    which follow Y~N(mu_1,V_1) and X~N(mu_2,V_2) with non degenerate covariance matrices.
    We test the null hypothesis mu_1=mu_2 and V_1 = V_2. We use the Mahalonobis distance as
    test statistic which measures the distance of means in terms of units of standard deviation.
    returns {"p-value": p-value, "d": Mahalanobis distance}
    """
    m1 = X.shape[0]
    n = X.shape[1]
    if(n != Y.shape[1]):
        print("[ERROR] dimension mismatch. Samples X and Y must have the same dimension")
        return None
    Sigma1 = X.T @ X
    mu1 = np.mean(X,axis=0)
    m2 = Y.shape[0] 
    Sigma2 = Y.T @ Y
    mu2 = np.mean(Y,axis=0)
    # check degrees of freedom
    if(m1+m2-2 <= n):
        print("[ERROR] not enough samples to invert covariance matrix")
    # compute mahalanobis distance
    D = mu1-mu2
    d2 = (m1+m2-2)*(D @ pinv(Sigma1 + Sigma2) @ D)
    m = m1+m2-2
    T = (m1*m2)/(m1+m2)*d2
    F = (m-n+1)/(n*m)*T
    p = f.sf(F,dfn=n,dfd=m-n+1)
    return {"p-value":p, "d":np.sqrt(d2)}

# TODO Bootstrap hotelling t2