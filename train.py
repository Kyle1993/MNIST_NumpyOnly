import numpy as np
import data_process
import pickle
from file_path import *
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-epochs',type=int,default=10)
parse.add_argument('-batch_size',type=int,default=128)
parse.add_argument('-lr',type=float,default=1e-4)
args = parse.parse_args()


epochs = args.epochs
lr = args.lr
batch_size = args.batch_size

input_size = 28*28
middle_size = input_size*2
output_size = 10
data_process.DownloadProcessData(mnist_path)


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

def CEL_backward(y,tar):
    grad = []
    for i in range(y.shape[0]):
        logit = y[i,:]-max(y[i,:])
        logit = np.exp(logit)/np.sum(np.exp(logit))
        tar_onehot = np.zeros((output_size),dtype=np.int32)
        tar_onehot[tar[i]] = 1
        grad.append(logit-tar_onehot)
    return np.asarray(grad,dtype=np.float32)

def relu_diff(input,grad_backward):
    return np.multiply(grad_backward,((input>0).astype(np.int32)))

def add_diff(grad_backward):
    return grad_backward


weight1 = np.random.normal(size=(input_size,middle_size))/(input_size*middle_size)    # input_size*middle_size
biase1 = np.random.normal(size=(1,middle_size))    # 1*middle_size

weight2 = np.random.normal(size=(middle_size,output_size))/(middle_size*output_size)  # middle_size*output_size
biase2 = np.random.normal(size=(1,output_size))   # 1*output_size

f = open(os.path.join(root_path,'training.log'),'w')
f.write('Training Log\n')
f.write('---------------------------------------\n')


for epoch in range(epochs):
    if (epoch+1)%5==0:
        lr = lr/10
    train_loader = data_process.DataLoader(os.path.join(mnist_path, 'train_data.npy'), os.path.join(mnist_path, 'train_label.npy'),batch_size)
    for step,(data,label) in enumerate(train_loader):
        # forward
        biase1_repeat = np.repeat(biase1,batch_size,axis=0)
        biase2_repeat = np.repeat(biase2,batch_size,axis=0)
        o1 = np.dot(data,weight1)+biase1_repeat  # batch_size*middle_size
        o2 = relu(o1)                            # batch_size*middle_size
        o3 = np.dot(o2,weight2)+biase2_repeat    # batch_size*output_size
        loss = CrossEntropyLoss(o3,label)
        # print(np.mean(loss))

        # backward
        diff_loss_o3 = CEL_backward(o3,label)                      # batch_size * output_size
        diff_loss_o2 = np.dot(diff_loss_o3,np.transpose(weight2))  # batch_size * middle_size

        grad_w2 = np.dot(np.transpose(o2),diff_loss_o3)            # middle_size * output_size
        grad_b2 = add_diff(diff_loss_o3)                           # batch * output_size

        diff_loss_o1 = relu_diff(o1,diff_loss_o2)                  # batch_size * middle_size

        grad_w1 = np.dot(np.transpose(data),diff_loss_o1)          # input_size * middle_size
        grad_b1 = add_diff(diff_loss_o1)                           # batch_size * middle_size

        #optimize
        weight2 += -lr*grad_w2
        biase2 += np.mean(-lr*grad_b2,axis=0)
        weight1 += -lr*grad_w1
        biase1 += np.mean(-lr*grad_b1,axis=0)

        #eval
        if step%100 == 0:
            test_loader = data_process.DataLoader(os.path.join(mnist_path, 'test_data.npy'),os.path.join(mnist_path, 'test_label.npy'), batch_size)
            total_tloss = 0
            for i,(test_data,test_label) in enumerate(test_loader):
                biase1_repeat = np.repeat(biase1, batch_size, axis=0)
                biase2_repeat = np.repeat(biase2, batch_size, axis=0)
                to1 = np.dot(test_data, weight1) + biase1_repeat
                to2 = relu(to1)
                to3 = np.dot(to2, weight2) + biase2_repeat
                tloss = CrossEntropyLoss(to3, test_label)
                total_tloss += np.mean(tloss)
            print('epoch {:<3d}, step {:<3d}, train_loss {:<10f}, test_loss {:<10f}'.format(epoch,step,np.mean(loss),total_tloss/(i+1)))
            f.write('epoch {:<3d}, step {:<3d}, train_loss {:<10f}, test_loss {:<10f}\n'.format(epoch,step,np.mean(loss),total_tloss/(i+1)))

f.close()

print('Saving Model...')
with open(os.path.join(root_path,'model.mod'),'wb') as f:
    pickle.dump([weight1,biase1,weight2,biase2],f)
print('Doen!')







