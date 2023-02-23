from muvapack.glm import GLM
import numpy as np

def test_glm_fit():
    # do a very simple linear regression test
    y = np.linspace(0,9,num=10)
    X = np.ones((10,2))
    X[:,0] = 0.5*y
    y += 1.0
    W = np.eye(10)
    glm = GLM(y,X,W)
    assert (1.99 <= glm.beta[0] <= 2.01), "GLM regression Test Failed."
    assert (0.99 <= glm.beta[1] <= 1.01), "GLM regression Test Failed."

def test_glm_variance_ci():
    # generate random samples and see how well the sigma confidence interval does
    N = 1000
    count = 0
    for i in range(N):
        y = np.random.normal(scale=2,size=30)
        X = np.ones((30,1))
        W = np.eye(30)
        glm = GLM(y,X,W)
        # now get the CI for sigma (estimate true sigma is 2)
        ci = glm.get_sigma_ci(alpha=0.1)
        if(ci[0] <= 2 <= ci[1]):
            count += 1
        # we would expect count=900 on average. We expect count<850 almost never p<3*10^{-7}
    assert count >= 850

def test_glm_parameter_ci():
    # test the confidence interval for the single parameters + confidence cubes
    N = 1000
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(N):
        y = np.random.normal(loc = 1.0, scale=2,size=30)
        X = np.ones((30,2))
        X[:,1] = np.linspace(0,5,num=30)
        W = np.eye(30)
        glm = GLM(y,X,W)
        # now get the CI for both parameters
        ci = glm.param_individual_ci(alpha=0.1)
        if(ci[0,0] <= 1 <= ci[0,1]):
            count1 += 1
        if(ci[1,0] <= 0 <= ci[1,1]):
            count2 += 1
        # now test confidence cubes
        ci = glm.param_confidence_cube(alpha=0.1)
        if(ci[0,0] <= 1 <= ci[0,1]):
            if(ci[1,0] <= 0 <= ci[1,1]):
                count3 += 1
    assert count1 >= 850
    assert count2 > 850
    assert count3 > 850


