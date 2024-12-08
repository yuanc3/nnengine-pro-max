from lightGE.core import Model, Conv2d, Linear, ReLU, MaxPool2d, Sequential,BatchNorm2d,Dropout


def vgg_block(in_channels, out_channels, num_convs, kernel_size=3, stride=1, padding=1):
    block = Sequential([])
    in_channels = in_channels
    for i in range(num_convs):
        conv2d = Sequential([
                Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                BatchNorm2d(out_channels),
                ReLU()
                ])
        block.add(conv2d)
        in_channels = out_channels
    block.add(MaxPool2d(2))
    return block


class VGG(Model):
    def __init__(self, in_channels, num_classes, group) -> None:
        super().__init__()
        self.features = Sequential(
            [vgg_block(in_channels=in_channels, out_channels=64, num_convs=group[0]),
             vgg_block(in_channels=64,  out_channels=128, num_convs=group[1]),
             vgg_block(in_channels=128, out_channels=256, num_convs=group[2]),
             vgg_block(in_channels=256, out_channels=512, num_convs=group[3]),
             vgg_block(in_channels=512, out_channels=512, num_convs=group[4]),
             ])

        self.classifier = Sequential(
            [Linear(512 * 7 * 7, 4096),      #需要手动调整
            ReLU(),
            Dropout(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, num_classes)
            ])

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.reshape((x.shape[0], -1))
        x = self.classifier(x)
        return x.softmax().log()

def VGG16(in_channels, num_classes):
    return VGG(in_channels, num_classes, [2,2,3,3,3])

def VGG19(in_channels, num_classes):
    return VGG(in_channels, num_classes, [2,2,4,4,4])