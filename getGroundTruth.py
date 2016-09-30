from os import path
import pandas as pd
from sys import argv

script,prefix = argv
# example usage python getAccuracy.py imgListTestNewRegression_
# it will generate a single file with name imgListTestNewRegression_.csv containing all the 
# attributes and scores for all the images given in 
prefix = str(prefix)

listSrc = "/home/gautam/deepImageAestheticsAnalysis/imgListFiles_label/"

imgSrc = "/home/gautam/deepImageAestheticsAnalysis/datasetImages"

attr = ["BalacingElements","ColorHarmony","Content","DoF","Light","MotionBlur","Object",
        "Repetition","RuleOfThirds","Symmetry","VividColor","score"]

testFiles = []
for item in attr:
    testfile = path.join(listSrc,prefix+str(item)+".txt")
    assert path.exists(testfile)
    testFiles.append(testfile)
    print testfile


df1 = pd.read_csv(testFiles[0],delimiter = ' ',header=None)
df1.columns = ['ImageFile',attr[0]]
df2 = pd.read_csv(testFiles[1], delimiter = ' ',header=None)
df2.columns = ['ImageFile',attr[1]]
df3 = pd.merge(df1,df2)
olddf = df3

for i in range(2,len(testFiles)):
    print "working on {}".format(testFiles[i])
    df = pd.read_csv(testFiles[i],delimiter = ' ',header=None)
    df.columns = ['ImageFile',attr[i]]

    newdf = pd.merge(olddf,df)
    olddf = newdf.copy()



newdf.to_csv(prefix+'.csv',index=False)

