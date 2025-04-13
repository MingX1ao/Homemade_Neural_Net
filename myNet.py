import numpy as np

class NetWork(object):
    
    def __init__(self, size):
        '''
        初始化神经网络，给权重和偏置赋初值
        :para size: 一个一维数组，包含每一层的神经元个数
        '''
        self.layer_num = len(size)
        self.size = size
        self.weight = [np.random.randn(n, m)/np.sqrt(n) for m, n in zip(size[:-1], size[1:])]
        self.bias = [np.random.randn(n, 1) for n in size[1:]]
        


    def sigmoid(self, x):
        '''
        sigmoid激活函数
        :para x: 上一层的输入加权 + 本层偏置
        :return: 本层输出
        '''
        out = 1.0 / (1.0 + np.exp(-x))
        
        return out

    def sigmoid_derivative(self, x):
        '''
        sigmoid的导数
        '''
        return self.sigmoid(x) * (1-self.sigmoid(x))



    def final_output(self, come_in):
        '''
        给出最终输出
        :para come_in: 原始输入
        '''
        rst = come_in
        for i in range(len(self.weight)):
            rst = self.sigmoid(np.dot(self.weight[i], rst) + self.bias[i])
        return rst


    def Loss(self, out_put, ground_truth):
        '''
        计算损失函数，暂时简单一点
        :para out_put: 模型输出结果
        :para ground_truth: 真实结果
        :return: 损失值
        '''
        return out_put - ground_truth




    def forward_propagation(self, come_in):
        '''
        完成前向传播的过程
        :para come_in: 给输入层的原始向量
        :return rst: 各层的输出
        :return rst_unact: 各层的输入
        '''
        # 注意，rst和rst_unact储存的都是列向量
        rst = [come_in]
        rst_unact = []      # 表示当前层前一级的输入加权，不含输入层

        for i in range(len(self.weight)):
            r_u = np.dot(self.weight[i], rst[-1]) + self.bias[i]
            rst_unact.append(r_u)
            r = self.sigmoid(r_u)
            rst.append(r)

        return rst, rst_unact
    

    def back_propagation(self, come_in, ground_truth):
        '''
        模拟反向传播
        :para come_in: 原始输入
        :ground_truth: 真实值
        :return: 各层的下降梯度
        '''
        rst, rst_unact = self.forward_propagation(come_in)
        loss = self.Loss(self.final_output(come_in), ground_truth)

        delta_nabal_weight = [np.zeros(w.shape) for w in self.weight]
        delta_nabal_bias = [np.zeros(b.shape) for b in self.bias]

        # 对输出层进行全微分可以得到下面的结果
        delta = loss * self.sigmoid_derivative(rst_unact[-1])
        delta_nabal_weight[-1] = np.dot(delta, rst[-2].transpose())
        delta_nabal_bias[-1] = delta

        # 开始反向传播，进行迭代
        for i in range(2, self.layer_num):
            delta = np.dot(self.weight[-i+1].transpose(), delta) * self.sigmoid_derivative(rst_unact[-i])
            delta_nabal_weight[-i] = np.dot(delta, rst[-i-1].transpose())
            delta_nabal_bias[-i] = delta
        

        return delta_nabal_weight, delta_nabal_bias
    

    # 使用mini_batch法
    def update_mini_batch(self, mini_batch_image, mini_batch_label, mini_batch_size, eta):
        '''
        通过一个batch的数据对神经网络参数进行更新
        需要对当前batch中每张图片调用backprop函数将误差反向传播
        求每张图片对应的权重梯度以及偏置梯度，最后进行平均使用梯度下降法更新参数
        '''
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        for x,y in zip(mini_batch_image, mini_batch_label):
            delta_nabla_w, delta_nabla_b = self.back_propagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weight = [w-(eta/mini_batch_size)*nw for w, nw in zip(self.weight, nabla_w)]
        self.bias = [b-(eta/mini_batch_size)*nb for b, nb in zip(self.bias, nabla_b)]



    
    def SGD(self, train_image, train_label, mini_batch_size, eta):
        '''
        Stochastic gradiend descent随机梯度下降法，将训练数据分多个batch
        一次使用一个mini_batch_size的数据，调用update_mini_batch函数更新参数
        '''
        mini_batches_image = [train_image[k:k+mini_batch_size] for k in range(0, len(train_image), mini_batch_size)]
        mini_batches_label = [train_label[k:k+mini_batch_size] for k in range(0, len(train_label), mini_batch_size)]
        for mini_batch_image, mini_batch_label in zip(mini_batches_image, mini_batches_label):
            self.update_mini_batch(mini_batch_image, mini_batch_label, mini_batch_size, eta)

  


    def evaluate(self, images, labels):
        result = 0
        for img, lab in zip(images, labels):
            predict_label = self.final_output(img)
            if np.argmax(predict_label) == np.argmax(lab):      # 取最大的索引
                result += 1
        return result
