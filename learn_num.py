import numpy as np
from myNet import NetWork as net
from struct import unpack
import pickle


def read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))      # 读meta_data
        # 可以加一个判断magic是否为目标的分支
        img = np.fromfile(f, dtype=np.uint8).reshape(num, rows*cols, 1)
    return img, rows, cols


def read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        label = np.fromfile(f, dtype=np.uint8)
    return label
    

# 归一化数据
def normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


# 将标签变成单列的数据
def one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab.reshape(lab.shape[0], lab.shape[1], 1)   # 训练时需要张成张量



# 加载数据集
def load_data():
    train_set, rows, cols = read_image(r'.\data\MNIST\raw\train-images-idx3-ubyte')
    train_lab = read_label(r'.\data\MNIST\raw\train-labels-idx1-ubyte')
    test_set, _, _  = read_image(r'.\data\MNIST\raw\t10k-images-idx3-ubyte')
    test_lab  = read_label(r'.\data\MNIST\raw\t10k-labels-idx1-ubyte')

    train_set = normalize_image(train_set)
    train_lab = one_hot_label(train_lab)
    test_set  = normalize_image(test_set)
    test_lab  = one_hot_label(test_lab)

    return train_set, train_lab, test_set, test_lab, rows, cols



epochs = 50
train_set, train_lab, test_set, test_lab, rows, cols = load_data()
learn_net = net([rows*cols, 100, 30, 10])
for i in range(epochs):
    learn_net.SGD(train_set, train_lab, 10, 3)
    print("Epoch{0}: accuracy is {1}/{2}".format(i+1, learn_net.evaluate(test_set, test_lab), len(test_set))) 


# save the model
# with open('models/num_learn.pkl', 'wb') as f:
#     pickle.dump(learn_net, f)

# # load the saved model
# with open('my_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # continue training the model
