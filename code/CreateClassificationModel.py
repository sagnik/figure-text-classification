import pickle
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    dataDir="../data/"
    dataPickleLoc=dataDir+"wordslabels.nparray.pickle"
    datal=pickle.load(open(dataPickleLoc)) 

    np.random.shuffle(datal)
    data=datal[:,:-1]
    label=datal[:,-1]
    print data.shape,label.shape,np.unique(label)

    clf=OneVsRestClassifier(RandomForestClassifier(n_estimators=10)).fit(data,label)
    modelPickleLoc=dataDir+"wordclassifymodel-rf.pickle"
    pickle.dump(clf,open(modelPickleLoc,"wb"))    

if __name__=="__main__":
    main()   
