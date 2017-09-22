from six.moves import urllib
import gzip
import os
import os.path
import errno
import codecs
import numpy as np
from PIL import Image

train_data_file = 'train_data.npy'
train_label_file = 'train_label.npy'
test_data_file = 'test_data.npy'
test_label_file = 'test_label.npy'

data_urls = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]

def DownloadProcessData(data_path):
    try:
        os.makedirs(data_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    if not _check_exists_gz(data_path):
        print('You can Download .gz file manually, and put them in {} '.format(os.path.join(data_path)))
        download(data_path)
    if not _check_exists_npy(data_path):
        process_data(data_path)
    else:
        print('Data existsed!')
    print('Data Process Down!')

def _check_exists_gz(data_path):
    for url in data_urls:
        filename = url.strip().split('/')[-1]
        if not os.path.exists(os.path.join(data_path, filename)):
            return False
    return True

def _check_exists_npy(data_path):
    filenames = ['train_data.npy','train_label.npy','test_data.npy','test_label.npy']
    for name in filenames:
        if not os.path.exists(os.path.join(data_path, name)):
            return False
    return True

def download(data_path):
    for url in data_urls:
        filename = url.rpartition('/')[2]
        file_path = os.path.join(data_path, filename)
        print('Downloading ' + url)
        data = urllib.request.urlopen(url).read()
        with open(file_path, 'wb') as f:
            f.write(data)


def process_data(data_path):
    # process and save as torch files
    print('Unziping files... ')
    for url in data_urls:
        filename = url.rpartition('/')[2]
        file_path = os.path.join(data_path, filename)
        with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())

    print('Processing Data...')

    training_set = (
        read_image_file(os.path.join(data_path, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(data_path, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join(data_path, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(data_path, 't10k-labels-idx1-ubyte'))
    )

    np.save(os.path.join(data_path, train_data_file), training_set[0])
    np.save(os.path.join(data_path, train_label_file), training_set[1])
    np.save(os.path.join(data_path, test_data_file),test_set[0])
    np.save(os.path.join(data_path, test_label_file), test_set[1])

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return np.asarray(labels,dtype=int)

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return np.asarray(images,dtype=float).reshape((-1,28*28))


class DataLoader():
    def __init__(self,data_path,label_path,batch_size):
        self.data_path = data_path
        self.data = (np.load(data_path)/255-0.1307)/0.3081
        self.label = np.load(label_path)
        self.batch_size = batch_size
        self.index = 0
        self.data_len = self.data.shape[0]

    def __getitem__(self, index):
        if (index+1)*self.batch_size<self.data_len:
            batch_data = self.data[index*self.batch_size:(index+1)*self.batch_size]
            batch_label = self.label[index*self.batch_size:(index+1)*self.batch_size]
            return batch_data,batch_label
        return '__index__ ERROR'

    def __iter__(self):
        return self

    def __next__(self):
        if self.index+self.batch_size >self.data_len:
            raise StopIteration()
        batch_data = self.data[self.index:self.index+self.batch_size]
        batch_label = self.label[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        return batch_data,batch_label

class InferenceLoader():
    def __init__(self,file_path,batch_size=100):
        data = []
        self.file_list = os.listdir(file_path)
        for filename in self.file_list:
            img = Image.open(os.path.join(file_path,filename))
            img = img.convert('L')
            img = np.asarray(img).reshape((28,28))
            img = (img/255-0.1307)/0.3081
            data.append(img)
        self.data = np.asarray(data,dtype=np.float32)
        self.data = self.data.reshape((self.data.shape[0],-1))
        self.data_len = self.data.shape[0]
        self.batch_size = batch_size
        self.index = 0

    def __getitem__(self, index):
        if (index+1)*self.batch_size<self.data_len:
            batch_data = self.data[index*self.batch_size:(index+1)*self.batch_size]
            batch_filenames = self.file_list[index*self.batch_size:(index+1)*self.batch_size]
            return batch_data,batch_filenames
        return '__index__ ERROR'

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > self.data_len:
            raise StopIteration()
        if self.index+self.batch_size >self.data_len:
            batch_data = self.data[self.index:]
            batch_filenames = self.file_list[self.index:]
        else:
            batch_data = self.data[self.index:self.index+self.batch_size]
            batch_filenames = self.file_list[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        return batch_data,batch_filenames

if __name__ == '__main__':
    root_path = '/home/jianglibin/PythonProject/mnist/data'
    DownloadProcessData(root_path)
    dl = DataLoader(os.path.join(root_path,'train_data.npy'),os.path.join(root_path,'train_label.npy'),100)
    for i,(data,label) in enumerate(dl):
        print(data.shape)
        print(label.shape)