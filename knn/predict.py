import numpy as np
from util import get_data
from datetime import datetime
from sortedcontainers import SortedList
import operator

k = 5
def predict(X):
    y = np.zeros(len(X))
    # loop untuk semua data X baru
    for j, x in enumerate(X):
        sl = SortedList()
        # loop untuk data train
        for i, xt in enumerate(X):
            # hitung jarak
            dist = x - xt
            dist = dist.dot(dist)
            
            if len(sl) < k :
                sl.add(dist,y[i]) # sl untuk mengurutkan value dari kecil ke besar
            else:
                if dist < sl[-1][0]:
                    del sl[-1]
                    sl.add((dist, y[i]))
        
        #vote memilih label dari sortedlist
        votes = {}
        """
        for loop disini berfungsi untuk menghitung banyaknya 
        jumlah dari setiap label pada sl dalam bentuk dict cth {9 : 2, 2: 1} 
        label 9 muncul sebanyak 2 kali dan label 2 cuman 1 kali.
        
        """       
        for e in sl: 
            if e[1] in votes.keys(): 
                votes[e[1]] += 1
            else:
                votes[e[1]] =1
        
        # cari nilai max
        max_votes_class, max_votes = max(votes.items(), key=operator.itemgetter(1))
        # mencari nilai paling banyak muncul dari votes index ke 1, dari contoh diatas max adalah 9 karena 2 kali muncul.
        y[j] = max_votes_class
    return y

X, y = get_data(2000)
print(y)
predict(X)
print(y)