import caffe
import pandas as pd
from sys import argv
import cv2
import os 
import numpy as np


def load_image(path):
      try:
          img = cv2.imread(path)
          resized = cv2.resize(img,(256,256),interpolation = cv2.INTER_AREA)
          rgb_img = np.transpose(resized, (2,0,1))
          return rgb_img
      except:
          return None



def saveMean(meanFile):
    a=caffe.io.caffe_pb2.BlobProto()
    file=open(meanFile,'rb')
    data = file.read()
    a.ParseFromString(data)
    means=a.data
    meanImage=np.asarray(means)
    print "shape meanImage",meanImage.shape
    meanImage = meanImage.reshape(3,256,256)
    print "meanImage shape",meanImage.shape
    np.save('aadb_mean.npy',meanImage)


def formatImage(img,meanImg):
    # Substract mean image as we used the mean image substration in training also
    formattedImage = np.float64(img) - np.load(meanImg)
    return formattedImage[:,15:242,15:242] # center crop

#saveMean('./mean_AADB_regression_warp256.binaryproto')
#Run this to generate aadb_mean.npy file

script,prefix= argv
#example usage
# python getAccuracy.py imgListTestNewRegression_
# This will generate imgListTestNewRegression_predict.csv

prefix = str(prefix)
imgSrc = "/home/gautam/deepImageAestheticsAnalysis/datasetImages"
caffe.set_mode_gpu()
print "Initlizaion..."
weightsName = 'initModel'
model = './'+ weightsName +'.prototxt'
weights = './'+ weightsName + '.caffemodel'

net = caffe.Net(model, weights, caffe.TEST)
meanImg = './aadb_mean.npy'

groundTruth = pd.read_csv(prefix+'.csv')
imgList = groundTruth.ImageFile.tolist()
print groundTruth.columns
predAtt = pd.DataFrame(index=groundTruth.index,columns=groundTruth.columns)
for index, row in groundTruth.iterrows():
    imgPath = os.path.join(imgSrc,row['ImageFile'])
    print "{}th Image: processing {}".format(index,imgPath)
    img = formatImage(load_image(imgPath),meanImg)
    net.blobs['imgLow'].data[...] = img
    net.forward()
    predDict = {}
    predDict['BalacingElements'] = net.blobs['fc9_BalancingElement'].data[0][0]
    predDict['ColorHarmony'] = net.blobs['fc9_ColorHarmony'].data[0][0]
    predDict['Content'] = net.blobs['fc9_Content'].data[0][0]
    predDict['DoF'] = net.blobs['fc9_DoF'].data[0][0]
    predDict['Light'] = net.blobs['fc9_Light'].data[0][0]
    predDict['MotionBlur'] = net.blobs['fc9_MotionBlur'].data[0][0]
    predDict['Object'] = net.blobs['fc9_Object'].data[0][0]
    predDict['Repetition']  = net.blobs['fc9_Repetition'].data[0][0]
    predDict['RuleOfThirds'] = net.blobs['fc9_RuleOfThirds'].data[0][0]
    predDict['Symmetry'] = net.blobs['fc9_Symmetry'].data[0][0]
    predDict['VividColor'] = net.blobs['fc9_VividColor'].data[0][0]
    predDict['score'] = net.blobs['fc11_score'].data[0][0]
    predDict['ImageFile'] = row['ImageFile']
    predAtt.loc[index] = pd.Series(predDict)

predAtt.to_csv(prefix+'predict.csv',index=False)
