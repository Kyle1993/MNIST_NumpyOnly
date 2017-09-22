import numpy as np
import data_process
import pickle
from file_path import *


def CrossEntropyLoss(y,tar):
    loss = []
    for i in range(y.shape[0]):
        logit = y[i,:] - max(y[i,:])
        loss.append(-logit[tar[i]]+np.log(np.sum(np.exp(logit))))
    return np.asarray(loss,dtype=np.float32)

def relu(x):
    flag = (x>0).astype(np.int32)
    out = np.multiply(x,flag)
    return out


file_list = os.listdir(inference_path)
if len(file_list)==0:
    print('No files')
else:
    inference_loader = data_process.InferenceLoader(inference_path)
    with open(os.path.join(root_path,'model.mod'),'rb') as f:
        weight1,biase1,weight2,biase2 = pickle.load(f)

    print('Predicting...')

    f = open(os.path.join(root_path,'Predict_result'),'w')
    f.write('{:s}:{:s}\n'.format('FileName','Predict'))
    f.write('--------------------\n')
    for i,(data,names) in enumerate(inference_loader):
        biase1_repeat = np.repeat(biase1, data.shape[0], axis=0)
        biase2_repeat = np.repeat(biase2, data.shape[0], axis=0)
        o1 = np.dot(data, weight1) + biase1_repeat
        o2 = relu(o1)
        o3 = np.dot(o2, weight2) + biase2_repeat
        predict = np.argmax(o3,axis=1)
        for id,name in enumerate(names):
            print('{:8s}:{:d}'.format(name,predict[id]))
            f.write('{:8s}:{:d}\n'.format(name,predict[id]))

    print('Done!')
