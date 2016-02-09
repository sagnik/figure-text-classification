import json
import sys
import numpy as np

classDict={
   'xaxislabel':1, 
   'xaxisvalue':2, 
   'yaxislabel':3,
   'yaxisvalue':4,
   'legend':5,
   'figurelabel':6, 
   'notclassified':7, 
   'undefined':7,
   
}

def getRotation(word):
    return word['Rotation']

def FeatureExtractOneFile(loc):
    js=json.load(open(loc))
    if len(js['ImageText'])==0: 
        return None
    else:
        feat=[]
        for word in js['ImageText']:
            featWord=[]
            featWord.append(getRotation(word))
            #Other features to come here
            featWord.append(classDict[word['TextLabelGold']])
            feat.append(featWord) 
        return feat

def main():
    jsonLoc=sys.argv[1]
    feat=np.array(FeatureExtractOneFile(jsonLoc))
    print feat

if __name__=="__main__":
    main()
       
