import pandas as pd
from scipy.stats import spearmanr as spr
from sys import argv

script,prefix = argv
prefix = str(prefix)
# example Usage
# python getCorrelation.py imgListTestNewRegression_

groundTruth = pd.read_csv(prefix+'.csv')
predictAttr = pd.read_csv(prefix+'predict.csv')
assert (groundTruth.columns == predictAttr.columns).all()
assert groundTruth.shape == predictAttr.shape
assert pd.Series.equals(groundTruth.ImageFile,predictAttr.ImageFile)

for col in groundTruth.columns[1:]:
    attrGT = groundTruth.loc[:,col]
    attrP = predictAttr.loc[:,col]
    rho,pval = spr(attrGT,attrP)
    print "For {} rho: {} at p value: {}".format(col,rho,pval)

