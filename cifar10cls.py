# This code is the first attempt of pytorch, in this code, I rewrite the two method includeing __getitem__ and __len__ in datasets class which inherited the basic
# class datasets, and this class would supply the train or test data to Dataloader

# We use Dataloader to load data

# build network

# train and test network




import torch.nn as nn
import pickle
import numpy as np
import glob
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 超参数
BATCH_SIZE = 100
# 损失函数
loss_func = nn.CrossEntropyLoss()
# 可以在CPU或者GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10的输入图片各channel的均值和标准差
mean = [x/255 for x in [125.3, 23.0, 113.9]]
std = [x/255 for x in [63.0, 62.1, 66.7]]
n_train_samples = 50000


def read_cifar10(path=".\\cifar-10\\cifar-10-batches-py\\cifar-10-batches-py\\"):
    print(path)
    traindatapath = path + "data*"
    testdatapath = path + "test*"
    # data_img = np.zeros(shape=(50000, 32, 32, 3), dtype=np.float32)
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    # 训练数据
    for path in sorted(glob.glob(traindatapath)):
        print(path)
        path = open(path, 'rb')

        img_dic = pickle.load(path, encoding='bytes')

        # 查看数据情况
        # print(img_dic.keys())
        # print(img_dic[b"batch_label"])
        # print(img_dic[b"labels"])
        # print(img_dic[b"data"])
        # print(img_dic[b"filenames"])

        train_label.append(np.array(img_dic[b"labels"]))
        img = np.array(img_dic[b"data"])
        for i in range(img.shape[0]):
            tmp = np.array(img[i].reshape((3, 32, 32)))
            tmp = tmp.transpose((1, 2, 0))
            # 输出图像
            # plt.imshow(tmp)
            # plt.show()
            train_data.append(tmp)
    train_label = np.array(train_label).reshape((50000))
    train_data = np.array(train_data, dtype=np.float32)
    train_data /= 255


    # 测试数据
    for path in sorted(glob.glob(testdatapath)):
        print(path)
        path = open(path, 'rb')
        img_dic = pickle.load(path, encoding='bytes')
        test_label.append(np.array(img_dic[b"labels"]))
        img = np.array(img_dic[b"data"])
        for i in range(img.shape[0]):
            tmp = np.array(img[i].reshape((3, 32, 32)))
            tmp = tmp.transpose((1, 2, 0))
            # 输出图像
            # plt.imshow(tmp)
            # plt.show()
            test_data.append(tmp)
    test_label = np.array(test_label).reshape((10000))
    test_data = np.array(test_data, dtype=np.float32)
    test_data /= 255

    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    return train_data, train_label, test_data, test_label


class Mydataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None, train=True):
        imgs = []
        labels = []
        for line in sorted(glob.glob(txt_path)):
            img_dic = pickle.load(open(line, 'rb'), encoding='bytes')
            label = np.array(img_dic[b"labels"])
            img = np.array(img_dic[b"data"])
            for i in range(img.shape[0]):
                imgs.append(img[i])
                labels.append(label[i])
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        assert index < len(self.imgs)
        img = self.imgs[index]
        label = self.labels[index]
        img = np.array(img).reshape((3, 32, 32))

        if self.transform is None:
            img /= 255.
        else:
            img = img.transpose((1, 2, 0))
            img = self.transform(img)

        # if self.train is True:
        label = np.eye(10)[label]
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return len(self.imgs)







class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            #inplace-选择是否进行覆盖运算
            #意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            #意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            #这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #(32-3+2)/1+1=32    32*32*64
            #Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)   #(32-2)/2+1=16         16*16*64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, stride=1, padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)     #(8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)   #(2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            #y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            #nn.Liner(in_features,out_features,bias)
            #in_features:输入x的列数  输入数据:[batchsize,in_features]
            #out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            #bias: bool  默认为True
            #线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self,x):
        x = self.conv(x)
        #这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1

        #如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


def train_data():

    batch_size = 100  # 每次喂入的数据量

    # num_print=int(50000//batch_size//4)
    num_print = 100

    epoch_num = 1  # 总迭代次数

    lr = 0.01
    step_size = 10  # 每n次epoch更新一次学习率

    train_path = '.\\cifar-10\\cifar-10-batches-py\\cifar-10-batches-py\\data*'

    # to tensor将数据除以255
    train_transform = transforms.ToTensor()

    train_dataset = Mydataset(txt_path=train_path, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


    # 模型,优化器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Vgg16_net().to(device)

    # 在多分类情况下一般使用交叉熵
    criterion = nn.CrossEntropyLoss()
    '''
    params(iterable)-待优化参数的iterable或者定义了参数组的dict
    lr(float):学习率

    momentum(float)-动量因子

    weight_decay(float):权重衰减,使用的目的是防止过拟合.在损失函数中,weight decay是放在正则项前面的一个系数,正则项一般指示模型的复杂度
    所以weight decay的作用是调节模型复杂度对损失函数的影响,若weight decay很大,则复杂的模型损失函数的值也就大.

    dampening:动量的有抑制因子


    optimizer.param_group:是长度为2的list,其中的元素是两个字典
    optimzer.param_group:长度为6的字典,包括['amsgrad','params','lr','weight_decay',eps']
    optimzer.param_group:表示优化器状态的一个字典

    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    '''
    scheduler 就是为了调整学习率设置的，我这里设置的gamma衰减率为0.5，step_size为10，也就是每10个epoch将学习率衰减至原来的0.5倍。

    optimizer(Optimizer):要更改学习率的优化器
    milestones(list):递增的list,存放要更新的lr的epoch
    gamma:(float):更新lr的乘法因子
    last_epoch:：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1
    '''

    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

    # 训练
    loss_list = []  # 为了后续画出损失图
    start = time.time()

    # train
    for epoch in range(epoch_num):
        print("epoch: %d" % epoch)
        ww = 0
        running_loss = 0.0
        # 0是对i的给值(循环次数从0开始技术还是从1开始计数的问题):
        # ???
        for i, (inputs, labels) in enumerate(train_loader, 0):

            # 将数据从train_loader中读出来,一次读取的样本是32个
            inputs, labels = inputs.to(device), labels.to(device)
            # 用于梯度清零,在每次应用新的梯度时,要把原来的梯度清零,否则梯度会累加
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels).to(device)

            # 反向传播,pytorch会自动计算反向传播的值
            loss.backward()
            # 对反向传播以后对目标函数进行优化
            optimizer.step()

            running_loss += loss.item()
            loss_list.append(loss.item())

            if (i + 1) % num_print == 0:
                print('[%d epoch,%d]  loss:%.6f' % (epoch + 1, i + 1, running_loss / num_print))
                running_loss = 0.0

        lr_1 = optimizer.param_groups[0]['lr']
        print("learn_rate:%.15f" % lr_1)
        schedule.step()

    end = time.time()
    print("time:{}".format(end - start))
    savepath = ".\\cifar_net.pth"
    torch.save(model.state_dict(), savepath)




def test_model():

    batch_size = 128

    test_path = '.\\cifar-10\\cifar-10-batches-py\\cifar-10-batches-py\\test*'

    # to tensor将数据除以255
    train_transform = transforms.ToTensor()

    test_dataset = Mydataset(txt_path=test_path, transform=train_transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    savepath = ".\\cifar_net.pth"
    model = Vgg16_net()
    model.load_state_dict(torch.load(savepath))
    model.to(device)
    # 由于训练集不需要梯度更新,于是进入测试模式
    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():  # 训练集不需要反向传播
        print("=======================test=======================")
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
            labels = labels.argmax(dim=1)
            total += inputs.size(0)
            correct += torch.eq(pred, labels).sum().item()

    print("Accuracy of the network on the 10000 test images:%.2f %%" % (100 * correct / total))
    print("===============================================")



if __name__ == '__main__':
    # read_cifar10()
    train_data()
    test_model()



