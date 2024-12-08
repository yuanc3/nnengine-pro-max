from lightGE.core import Tensor, Model, Conv2d, Linear, ReLU, \
    Sequential, BatchNorm2d, AvgPool2d, MaxPool2d


class Conv2d_BN_ReLU(Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, activation=True):
        super().__init__()

        self.seq = Sequential(
            [Conv2d(in_channels, out_channels, filter_size=kernel_size, stride=stride, padding=padding, bias=bias),
            BatchNorm2d(out_channels),
            ]
        )
        if activation:
            self.seq.add(ReLU())

    def forward(self, x):
        return self.seq(x)


class BasicBlock(Model):
    block_name = "basic"

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.left = Sequential(
            [Conv2d(in_channels, out_channels, filter_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, filter_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channels)
            ]
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = Sequential(
                [Conv2d(in_channels, out_channels, filter_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels)
                ]
            )

    def forward(self, x):
        out = self.left(x)
        if self.shortcut:
            out = out + self.shortcut(x)
        else:
            out = out + x
        relu = ReLU()
        out = relu(out)
        return out


class BottleNeck(Model):
    block_name = "bottleneck"

    def __init__(self, in_channels, out_channels, strides):
        super().__init__()
        self.conv1 = Conv2d_BN_ReLU(in_channels, out_channels, 1, stride=1, padding=0, bias=False)  # same padding
        self.conv2 = Conv2d_BN_ReLU(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = Conv2d_BN_ReLU(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=False)

        # fit input with residual output
        self.shortcut = Sequential([
            Conv2d(in_channels, out_channels * 4, filter_size=1, stride=strides, padding=0, bias=False),
            BatchNorm2d(out_channels * 4)]
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x)
        relu = ReLU()

        return relu(out)


class ResNet(Model):
    """
    building ResNet
    """

    def __init__(self, block, groups, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.channels = 64  # out channels from the first convolutional layer
        self.block = block
        # self.relu = ReLU()
        self.conv1 = Conv2d(in_channels, self.channels, 3, stride=1, padding=1, bias=False)
        self.bn = BatchNorm2d(self.channels)
        self.pool1 = MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)
        self.pool2 = AvgPool2d(filter_size=4)
        patches = 512 if self.block.block_name == "basic" else 512 * 4
        self.fc = Linear(patches, num_classes)  # for 224 * 224 input size

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = Sequential([])
        for i in range(len(list_strides)):
            conv_x.add(self.block(self.channels, channels, list_strides[i]))
            self.channels = channels if self.block.block_name == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        relu = ReLU()
        out = self.bn(out)
        out = relu(out)
        # out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.reshape((out.shape[0], -1))
        out = self.fc(out)
        return out.softmax().log()


def ResNet_18(num_classes=10, in_channels=3):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

def ResNet_34(num_classes=10, in_channels=3):
    return ResNet(block=BasicBlock, groups=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

def ResNet_50(num_classes=10, in_channels=3):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

def ResNet_101(num_classes=10, in_channels=3):
    return ResNet(block=BottleNeck, groups=[3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)

def ResNet_152(num_classes=10, in_channels=3):
    return ResNet(block=BottleNeck, groups=[3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels)