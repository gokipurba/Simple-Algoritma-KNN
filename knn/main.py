import numpy as np
from util import get_data
from datetime import datetime
from sortedcontainers import SortedList
import operator
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X #data training 
        self.y = y
    
    def predict(self, X):  #X disini data yang baru dan akan di predict
        y = np.zeros(len(X))

        for j, x in enumerate(X):
            sl = SortedList()
            for i,xt in enumerate(self.X):
                dist = x - xt
                dist = dist.dot(dist)
                if len(sl)< self.k:
                    sl.add((dist, self.y[i]))
                else:
                    if dist < sl[-1][0]:
                        del sl[-1]
                        sl.add((dist,self.y[i]))
             # vote 
            votes = {}
            print(sl)
            for e in sl:
                if e[1] in votes.keys():
                    votes[e[1]] += 1
                else:
                    votes[e[1]] = 1
            
            print(votes)
            # cari nilai max
            max_votes_class, max_votes = max(votes.items(), key=operator.itemgetter(1))
            y[j] = max_votes_class
            print(y[j])
        #print(y)
        return y

if __name__ == '__main__':
    X, y = get_data(100)
    #print(f'ini y sebelum : {y}')
    ntrain = 50
    Xtrain, ytrain = X[:ntrain], y[:ntrain]
    Xtest, ytest = X[ntrain:], y[ntrain:]

    knn = KNN(5)
    knn.fit(Xtest,ytest)
    knn.predict(X)
    #print(f'ini y sesudah : {y}')
