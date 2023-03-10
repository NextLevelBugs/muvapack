from muvapack.fa import FactorAnalysis
import numpy as np

"""
Many of these test are stochastic in nature. They can fail, allthough with very small probabiliy
Simply rerun if one of the test fails. If the same test fails again, there is likely an issue.
"""

def test_factor_analysis_I():
    n = 300
    k = 5
    # one underlying factor in 5d
    fac = np.random.normal(size=(n))*2
    Y = np.zeros((n,k))
    Y[:,0] = 0.5*fac+0.1*np.random.normal(size=(n))
    Y[:,1] = -0.5*fac+0.1*np.random.normal(size=(n))
    Y[:,2] = 0.8*fac+0.2*np.random.normal(size=(n))
    Y[:,3] = -0.2*fac+0.1*np.random.normal(size=(n))
    Y[:,4] = -0.2*fac+0.1*np.random.normal(size=(n))
    fa = FactorAnalysis(Y,verbose=True)
    assert fa.fac == 1, "Incorrect factor dimension recovered."
    # make sure that the phi i.e. noise was correctly recovered
    assert 0.50 < np.sum(np.sqrt(np.diag(fa.Phi))) < 0.70, "Phi was not correctly recovered"

def test_factor_analysis_II():
    # this time we just look at a more high dimensional example with 2 expl factors
    n = 300
    Y = np.random.normal(size=(n,20))
    f1 = np.ones((1,20))
    f2 = np.ones((1,20))
    f1[0,:-5] = 0
    f2[0,5:] = 0
    Y += np.random.normal(size=(n,1)) @ f1
    Y += np.random.normal(size=(n,1)) @ f2
    fa = FactorAnalysis(Y,verbose=True,alpha=0.01)
    assert fa.fac == 2, "Incrorrect factor dimension recovered."