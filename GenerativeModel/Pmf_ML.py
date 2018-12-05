import numpy as np

class Empirical_Model:
    def __init__(self):
        pass
    def _cal_prob(self, test_X, pro_table):
        pR, pG, pB = pro_table[0, test_X[:, 0]], pro_table[1, test_X[:, 1]], pro_table[2, test_X[:, 2]]
        prob = pR * pG * pB
        return prob
        
    def fit(self, train_X, train_y, prior=0.5):
        self.prior = prior
        c0, c1 = train_X[train_y == 0, :], train_X[train_y == 1, :]
        N0, N1 = c0.shape[0], c1.shape[0]
        Bin0_R, Bin0_G, Bin0_B = np.histogram(c0[:, 0], bins=np.arange(257))[0], np.histogram(c0[:, 1], bins=np.arange(257))[0], np.histogram(c0[:, 2], bins=np.arange(257))[0]
        Bin1_R, Bin1_G, Bin1_B = np.histogram(c1[:, 0], bins=np.arange(257))[0], np.histogram(c1[:, 1], bins=np.arange(257))[0], np.histogram(c1[:, 2], bins=np.arange(257))[0]
        self.pro_table0 = np.array([np.true_divide(Bin0_R, N0), np.true_divide(Bin0_G, N0), np.true_divide(Bin0_B, N0)])
        self.pro_table1 = np.array([np.true_divide(Bin1_R, N1), np.true_divide(Bin1_G, N1), np.true_divide(Bin1_B, N1)])
        
    def predict(self, test_X):
        prob0 = self._cal_prob(test_X, self.pro_table0) * self.prior # probability of skin
        prob1 = self._cal_prob(test_X, self.pro_table1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 <= prob1)
        return self.pred
    
    def test(self, test_X, test_y):
        self.TN, self.TP, self.FN, self.FP = 0, 0, 0, 0
        self.Precision, self.Recall = 0, 0
        prob0 = self._cal_prob(test_X, self.pro_table0) * self.prior # probability of skin
        prob1 = self._cal_prob(test_X, self.pro_table1) * (1 - self.prior) # probability of nonskin
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
        
class Empirical_Model_for_R:
    def __init__(self):
        pass
    def _cal_prob(self, test_X, pro_table):
        pR = pro_table[test_X]
        prob = pR
        return prob
        
    def fit(self, train_X, train_y, prior=0.5):
        self.prior = prior
        c0, c1 = train_X[train_y == 0], train_X[train_y == 1]
        N0, N1 = c0.shape[0], c1.shape[0]
        Bin0_R = np.histogram(c0, bins=np.arange(257))[0]
        Bin1_R = np.histogram(c1, bins=np.arange(257))[0]
        self.pro_table0 = np.array(np.true_divide(Bin0_R, N0))
        self.pro_table1 = np.array(np.true_divide(Bin1_R, N1))
        
    def predict(self, test_X):
        prob0 = self._cal_prob(test_X, self.pro_table0) * self.prior # probability of skin
        prob1 = self._cal_prob(test_X, self.pro_table1) * (1 - self.prior) # probability of nonskin
        self.pred = (prob0 <= prob1)
        return self.pred
    
    def test(self, test_X, test_y):
        self.TN, self.TP, self.FN, self.FP = 0, 0, 0, 0
        self.Precision, self.Recall = 0, 0
        prob0 = self._cal_prob(test_X, self.pro_table0) * self.prior # probability of skin
        prob1 = self._cal_prob(test_X, self.pro_table1) * (1 - self.prior) # probability of nonskin
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