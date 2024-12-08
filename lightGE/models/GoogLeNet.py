from lightGE.core import Model, Conv2d, Linear, ReLU, MaxPool2d, Sequential, AvgPool2d, Dropout, AdaptiveAvgPool2d


class GoogLeNet(Model):
    def __init__(self, in_chann = 3, num_classes=10, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(in_chann, 64, filter_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, filter_size=1)
        self.conv3 = BasicConv2d(64, 192, filter_size=3, padding=1)
        self.maxpool2 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = AdaptiveAvgPool2d()
        self.dropout = Dropout(0.4)
        self.fc = Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        
        x = x.reshape((x.shape[0], -1))
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x.softmax().log(), aux2, aux1
        return x.softmax().log()


#inception结构
class Inception(Model):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, filter_size=1)

        self.branch2 = Sequential([
            BasicConv2d(in_channels, ch3x3red, filter_size=1),
            BasicConv2d(ch3x3red, ch3x3, filter_size=3, padding=1)]   # 保证输出大小等于输入大小
        )

        self.branch3 = Sequential([
            BasicConv2d(in_channels, ch5x5red, filter_size=1),
            BasicConv2d(ch5x5red, ch5x5, filter_size=5, padding=2)]   # 保证输出大小等于输入大小
        )

        self.branch4 = Sequential([
            MaxPool2d(filter_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, filter_size=1)]
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = branch1.concat(branch2).concat(branch3).concat(branch4)
        return outputs


#辅助分类器
class InceptionAux(Model):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.averagePool = AvgPool2d(filter_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, filter_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = Linear(2048, 1024)
        self.fc2 = Linear(1024, num_classes)
        self.relu = ReLU()

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.reshape((x.shape[0], -1))
        # N x 2048
        x = self.relu(self.fc1(x))
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(Model):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, **kwargs)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x