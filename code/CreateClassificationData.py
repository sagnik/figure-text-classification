import os,sys
from FeatureExtractionOneFile import FeatureExtractOneFile
import pickle
import numpy as np

def main():
    jsonDir="../data/jsons/"
    jsons=[jsonDir+x for x in os.listdir(jsonDir) if x.endswith('json')]
    a=np.zeros(17).reshape((1,17))
    for json in jsons:
        a=np.vstack((a,np.array(FeatureExtractOneFile(json))))
    datal=a[1:,:]
    pickle.dump(a,open("../data/wordslabels.nparray.pickle","wb"))

if __name__=="__main__":
    main()  
