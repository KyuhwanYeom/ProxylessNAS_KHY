import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import math

from supernet import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_cell = 22
model_init = 'he_fout'

super_net = Supernets( # over-parameterized net 생성 (큰 net)
    width_stages=[24,40,80,96,192,320], n_cell_stages=[4,4,4,4,4,1], stride_stages=[2,2,2,1,2,1],
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ], n_classes=10, width_mult=1,
    bn_param=(0.1, 1e-3), dropout_rate=0
)

for m in super_net.modules(): # m은 각종 layer (Conv, BatchNorm, Linear ...)
            if isinstance(m, nn.Conv2d): # conv layer면
                if model_init == 'he_fout': # He initialization?
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): # batch norm이면 weight = 1, bias = 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): # linear면 weight = uniform, bias = 0
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

criterion = nn.CrossEntropyLoss() # loss 정의
optimizer_weight = optim.SGD(super_net.parameters(), lr=0.001, momentum=0.9) # weight optimizer 정의 (momentum-SGD)

weight = [[] for i in range(n_cell)] # init architecture parameter
for i in range(n_cell):
    weight[i] = [0 for j in range(len(n_cell[i]))] 
optimizer_arch = optim.Adam(super_net.parameters(), lr=1e-3, momentum=0.9) # architecture parameter (Adam)
                
nBatch = len(trainloader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # gpu 사용
# print(device)

super_net.to(device)


for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = super_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
