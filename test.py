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

data_process.DownloadProcessData(mnist_path)
batch_size = 100

with open(os.path.join(root_path,'model.mod'),'rb') as f:
    weight1,biase1,weight2,biase2 = pickle.load(f)

accuracy = 0
total_loss = 0
test_loader = data_process.DataLoader(os.path.join(mnist_path, 'test_data.npy'),os.path.join(mnist_path, 'test_label.npy'), batch_size)
for i,(test_data,test_label) in enumerate(test_loader):
    biase1_repeat = np.repeat(biase1, batch_size, axis=0)
    biase2_repeat = np.repeat(biase2, batch_size, axis=0)
    o1 = np.dot(test_data, weight1) + biase1_repeat
    o2 = relu(o1)
    o3 = np.dot(o2, weight2) + biase2_repeat
    loss = CrossEntropyLoss(o3, test_label)
    total_loss += np.mean(loss)
    predict = np.argmax(o3,axis=1)
    accuracy += np.sum((predict == test_label).astype(np.int32))/test_label.shape[0]

print('Test Loss:    {}'.format(total_loss/(i+1)))
print('Test Accuracy:{}'.format(accuracy/(i+1)))
with open(os.path.join(root_path,'test.log'),'w') as f:
    f.write('Test Loss    :{}\n'.format(total_loss/(i+1)))
    f.write('Test Accuracy:{}\n'.format(accuracy/(i+1)))