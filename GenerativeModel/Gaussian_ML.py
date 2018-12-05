import numpy as np
from math import gamma

class Gaussian_Model:
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
        if(df > 300):
            df = 300
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
    
    def _normpdf(self, x, mu, var):
        denominator = np.sqrt(2*np.pi*var)
        numerator = ((x - mu)**2)/var
        numerator = np.exp(-0.5 * numerator)
        return numerator/denominator
    
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
    
    def fit(self, train_X, train_y, prior=0.5):
        self.prior = prior
        c0, c1 = train_X[train_y == 0, :], train_X[train_y == 1, :]
        self.mean0 = np.mean(c0, axis=0)
        self.var0 = np.cov(c0.T)
        self.mean1 = np.mean(c1, axis=0)
        self.var1 = np.cov(c1.T)
        
    def predict(self, test_X):
        prob0 = self._multinormpdf(test_X, self.mean0, self.var0) * self.prior # probability of skin
        prob1 = self._multinormpdf(test_X, self.mean1, self.var1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 <= prob1)
        return self.pred
        
    def test(self, test_X, test_y):
        self.TN, self.TP, self.FN, self.FP = 0, 0, 0, 0
        self.Precision, self.Recall = 0, 0
        prob0 = self._multinormpdf(test_X, self.mean0, self.var0) * self.prior # probability of skin
        prob1 = self._multinormpdf(test_X, self.mean1, self.var1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 <= prob1)
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
        
        
class Gaussian_Model_for_R:
    def __init__(self):
        pass
     
    def _normpdf(self, x, mu, var):
        denominator = np.sqrt(2*np.pi*var)
        numerator = ((x - mu)**2)/var
        numerator = np.exp(-0.5 * numerator)
        return numerator/denominator
    
    def fit(self, train_X, train_y, prior=0.5):
        self.prior = prior
        c0, c1 = train_X[train_y == 0], train_X[train_y == 1]
        self.mean0 = np.mean(c0)
        self.var0 = np.var(c0)
        self.mean1 = np.mean(c1)
        self.var1 = np.var(c1)
        
    def predict(self, test_X):
        prob0 = self._normpdf(test_X, self.mean0, self.var0) * self.prior # probability of skin
        prob1 = self._normpdf(test_X, self.mean1, self.var1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 < prob1)
        return self.pred
        
    def test(self, test_X, test_y):
        self.TN, self.TP, self.FN, self.FP = 0, 0, 0, 0
        self.Precision, self.Recall = 0, 0
        prob0 = self._normpdf(test_X, self.mean0, self.var0) * self.prior # probability of skin
        prob1 = self._normpdf(test_X, self.mean1, self.var1) * (1 - self.prior) # probability of nonskin
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