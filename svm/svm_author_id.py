#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# decrease train data,increase train speed

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC
#clf = SVC(kernel="linear")
clf = SVC(10000,kernel="rbf")

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
predicted = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

# print("element 10 predict:%s" %("Sara" if(pred[10]==0) else "Chris"))
# print("element 26 predict:%s" %("Sara" if(pred[26]==0) else "Chris"))
# print("element 50 predict:%s" %("Sara" if(pred[50]==0) else "Chris"))

new_list = [i for i in predicted if i > 0]
print ("Chris:%d" %(len(new_list)))
print ("Sara:%d" %(len(predicted)-len(new_list)))
# predicted.

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,predicted)
print accuracy
#########################################################


