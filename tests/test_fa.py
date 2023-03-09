from muvapack.fa import FactorAnalysis
import numpy as np

"""
Many of these test are stochastic in nature. They can fail, allthough with very small probabiliy
Simply rerun if one of the test fails. If the same test fails again, there is likely an issue.
"""

def test_factor_analysis():
    n = 300
    k = 3
    # one underlying factor
    fac = np.random.normal(size=(n))
    Y = np.zeros((n,k))
    Y[:,0] = 0.5*fac+0.1*np.random.normal(size=(n))
    Y[:,1] = -0.5*fac+0.1*np.random.normal(size=(n))
    Y[:,2] = 0.8*fac+0.2*np.random.normal(size=(n))
    fa = FactorAnalysis(Y)
    assert fa.fac == 1, "Incorrect factor dimension recovered."