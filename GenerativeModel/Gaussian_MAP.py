import numpy as np
from math import gamma

class Gaussian_NIW_Model: # P(x|theta) ~ Gaussian, P(theta) ~ Normal Inverse Wishart
    def __init__(self):
        pass
    def _multivariate_t_distribution(self, x, mu, Sigma, df):
        '''
        input:
            x = parameter (n-d numpy array; will be forced to 2d)
            mu = mean (d dimensional numpy array)
            Sigma = scale matrix (dxd numpy array)
            df = degrees of freedom
        '''
        x = np.atleast_2d(x) # requires x as 2d
        nD = Sigma.shape[0] # dimensionality
        numerator = np.float64(gamma(1.0 * (nD + df) / 2.0))
        
        denominator = (
                gamma(1.0 * df / 2.0) * 
                np.power(df * np.pi, 1.0 * nD / 2.0) *  
                np.power(np.linalg.det(Sigma), 1.0 / 2.0) * 
                np.power(1.0 + (1.0 / df) * np.diagonal(np.dot( np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)), 
                         1.0 * (nD + df) / 2.0
                         )
                )
        return 1.0 * numerator / denominator
    
    def _multinormpdf(self, x, mu, var):
        mu = np.asarray(mu)
        var = np.asarray(var)
        k = x.shape[-1]
        det = np.linalg.det(var)
        inv = np.linalg.inv(var)
        denominator = np.sqrt(((2*np.pi)**k)*det)
        numerator = np.dot((x - mu), inv)
        numerator = np.sum((x - mu) * numerator, axis=-1)
        numerator = np.exp(-0.5 * numerator)
        return numerator/denominator
    
    def _calhyperparameter(self, n, mean, S, Mu0, k0, v0, P0):
        Mun = (k0 * Mu0 / (k0 + n)) + (n * mean / (k0 + n))
        kn = k0 + n
        vn = v0 + n
        Pn = P0 + S + ((k0*n / (k0 + n))*np.dot(mean-Mu0, mean-Mu0))
        return Mun, kn, vn, Pn
        
    def fit(self, train_X, train_y, prior=0.5):
        self.prior = prior
        c0, c1 = train_X[train_y == 0, :], train_X[train_y == 1, :]
        n0, n1 = c0.shape[0], c1.shape[0]
        mean0, var0 = np.mean(c0, axis=0), np.cov(c0.T)
        mean1, var1 = np.mean(c1, axis=0), np.cov(c1.T)
        Mu, k, v, P = np.array([0,0,0]), 100, 1, np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        self._Mu0, self._k0, self._v0, self._P0 = self._calhyperparameter(n0, mean0, n0*var0, Mu, k, v, P)
        self._Mu1, self._k1, self._v1, self._P1 = self._calhyperparameter(n1, mean1, n1*var1, Mu, k, v, P)
        
    def predict(self, test_X):
        sigma0 = self._P0 * ((self._k0 + 1) / (self._k0*(self._v0 - 3 + 1)))
        sigma1 = self._P1 * ((self._k1 + 1) / (self._k1*(self._v1 - 3 + 1)))
        prob0 = self._multinormpdf(test_X, self._Mu0, sigma0) * self.prior # probability of skin
        prob1 = self._multinormpdf(test_X, self._Mu1, sigma1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 < prob1)
        return self.pred
    
    def test(self, test_X, test_y):
        self.TN, self.TP, self.FN, self.FP = 0, 0, 0, 0
        sigma0 = self._P0 * ((self._k0 + 1) / (self._k0*(self._v0 - 3 + 1)))
        sigma1 = self._P1 * ((self._k1 + 1) / (self._k1*(self._v1 - 3 + 1)))
        prob0 = self._multinormpdf(test_X, self._Mu0, sigma0) * self.prior # probability of skin
        prob1 = self._multinormpdf(test_X, self._Mu1, sigma1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 < prob1)
        good = (self.pred == test_y)
        bad = (self.pred != test_y)
        positive = (self.pred == 0) # predict to skin
        negative = (self.pred == 1) # predict to nonskin
        truepositive = (positive & good)
        truenegative = (negative & good)
        falsepositive = (positive & bad)
        falsenegative = (negative & bad)
        self.TP, self.TN = np.count_nonzero(truepositive), np.count_nonzero(truenegative)
        self.FP, self.FN = np.count_nonzero(falsepositive), np.count_nonzero(falsenegative)
        self.Precision, self.Recall = self.TP/(self.TP+self.FP), self.TP/(self.TP+self.FN)
        self.acc = (self.TP+self.TN)/(self.TN+self.TP+self.FN+self.FP)
        
    
class Gaussian_NIG_Model_for_R: # P(x|theta) ~ Gaussian, P(theta) ~ Normal Inverse Gamma
    def __init__(self):
        pass
         
    def _normpdf(self, x, mu, var):
        denominator = np.sqrt(2*np.pi*var)
        numerator = ((x - mu)**2)/var
        numerator = np.exp(-0.5 * numerator)
        return numerator/denominator
    
    def _calhyperparameter(self, n, mean, var, m0, v0, a0, b0):
        mn = ((v0*m0) + (n*mean))/(v0 + n)
        vn = v0 + n
        an = a0 + (n/2)
        bn = b0 + (0.5*(n*var)) + (((n*v0)/(v0 + n))*((mean - m0)/2))
        return mn, vn, an, bn
        
    def fit(self, train_X, train_y, prior=0.5):
        self.prior = prior
        c0, c1 = train_X[train_y == 0], train_X[train_y == 1]
        n0, n1 = c0.shape[0], c1.shape[0]
        mean0, var0 = np.mean(c0, axis=0), np.cov(c0.T)
        mean1, var1 = np.mean(c1, axis=0), np.cov(c1.T)
        m, v, a, b = 0, 1, 1, 1
        self._Mu0, self._v0, self._a0, self._b0 = self._calhyperparameter(n0, mean0, var0, m, v, a, b)
        self._Mu1, self._v1, self._a1, self._b1 = self._calhyperparameter(n1, mean1, var1, m, v, a, b)
        
    def predict(self, test_X):
        sigma0 = self._b0 * ((self._v0 + 1) / (self._a0 * self._v0))
        sigma1 = self._b1 * ((self._v1 + 1) / (self._a1 * self._v1))
        prob0 = self._normpdf(test_X, self._Mu0, sigma0) * self.prior # probability of skin
        prob1 = self._normpdf(test_X, self._Mu1, sigma1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 < prob1)
        return self.pred
    
    def test(self, test_X, test_y):
        self.TN, self.TP, self.FN, self.FP = 0, 0, 0, 0
        sigma0 = self._b0 * ((self._v0 + 1) / (self._a0 * self._v0))
        sigma1 = self._b1 * ((self._v1 + 1) / (self._a1 * self._v1))
        prob0 = self._normpdf(test_X, self._Mu0, sigma0) * self.prior # probability of skin
        prob1 = self._normpdf(test_X, self._Mu1, sigma1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 < prob1)
        good = (self.pred == test_y)
        bad = (self.pred != test_y)
        positive = (self.pred == 0) # predict to skin
        negative = (self.pred == 1) # predict to nonskin
        truepositive = (positive & good)
        truenegative = (negative & good)
        falsepositive = (positive & bad)
        falsenegative = (negative & bad)
        self.TP, self.TN = np.count_nonzero(truepositive), np.count_nonzero(truenegative)
        self.FP, self.FN = np.count_nonzero(falsepositive), np.count_nonzero(falsenegative)
        self.Precision, self.Recall = self.TP/(self.TP+self.FP), self.TP/(self.TP+self.FN)
        self.acc = (self.TP+self.TN)/(self.TN+self.TP+self.FN+self.FP)