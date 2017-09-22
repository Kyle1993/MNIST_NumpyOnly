import os,sys
import errno

# don't modify me
root_path = sys.path[0]

# you can modify these
mnist_path = os.path.join(root_path,'data')
inference_path = os.path.join(root_path,'inferenceData')
#mnist_path = '/home/jianglibin/PythonProject/mnist/data'
#inference_path = '/home/jianglinbin/PythonProject/mnist/inferenceData'

try:
    os.makedirs(mnist_path)
    os.makedirs(inference_path)
except OSError as e:
    if e.errno == errno.EEXIST:
        pass
    else:
        raise
