import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class plus_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels=7, classes=1, deep_supervision=True, up_mode='bilinear', input_shape=(224, 224)):
        super(UnetPlusPlus, self).__init__()
        self.output_shape = (1, input_shape[0], input_shape[1])
        self.channels = in_channels
        self.classes = classes
        self.filtersize = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.depth = 4
        self.up_mode = up_mode
        #Max pooling and Upsampling init
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)

        self.down = nn.ModuleList()
        self.N1Up = nn.ModuleList()
        self.N2Up = nn.ModuleList()
        self.N3Up = nn.ModuleList()
        self.N4Up = nn.ModuleList()
        # Down Convolution
        self.down.append(plus_block(self.channels, self.filtersize[0]))
        for i_layer in range(self.depth):
            layer_down = plus_block(self.filtersize[i_layer], self.filtersize[i_layer+1])
            self.down.append(layer_down)

        # Upsampling + Convolution N1 skip
        for i_layer in range(self.depth-1):
            layer_Up1 = plus_block((self.filtersize[i_layer] + self.filtersize[i_layer+1]),
                                     self.filtersize[i_layer])
            self.N1Up.append(layer_Up1)
        self.N1Up.append(plus_block((self.filtersize[3] + self.filtersize[4]), self.filtersize[3]))

        # Upsampling + Convolution N2 skip
        for i_layer in range(self.depth-1):
            layer_Up2 = plus_block((self.filtersize[i_layer] * 2 + self.filtersize[i_layer+1]),
                                    self.filtersize[i_layer])
            self.N2Up.append(layer_Up2)

        # Upsampling + Convolution N3 skip
        for i_layer in range(self.depth-2):
            layer_Up3 = plus_block((self.filtersize[i_layer] * 3 + self.filtersize[i_layer+1]),
                                    self.filtersize[i_layer])
            self.N3Up.append(layer_Up3)

        # Upsampling + Convolution N4 skip
        self.N4Up = plus_block((self.filtersize[0] * 4 + self.filtersize[1]), self.filtersize[0])

        if self.deep_supervision:
            self.Output1 = OutConv(self.filtersize[0], classes)
            self.Output2 = OutConv(self.filtersize[0], classes)
            self.Output3 = OutConv(self.filtersize[0], classes)
            self.Output4 = OutConv(self.filtersize[0], classes)

        self.Output = OutConv(self.filtersize[0], classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_downsample = []
        for idx, layer in enumerate(self.down):
            if idx == 0:
                x = layer(x)
            else:
                x = layer(self.max_pool(x))
            x_downsample.append(x)

        x_upn1 = []
        for idx, n1_layer in enumerate(self.N1Up):
            x = n1_layer(torch.cat([x_downsample[idx], self.Up(x_downsample[idx+1])], dim=1))
            x_upn1.append(x)

        x_upn2 = []
        for idx, n2_layer in enumerate(self.N2Up):
            x = n2_layer(torch.cat([x_downsample[idx], x_upn1[idx], self.Up(x_upn1[idx+1])], dim=1))
            x_upn2.append(x)

        x_upn3 = []
        for idx, n3_layer in enumerate(self.N3Up):
            x = n3_layer(torch.cat([x_downsample[idx], x_upn1[idx], x_upn2[idx], self.Up(x_upn2[idx+1])], dim=1))
            x_upn3.append(x)
        x_upn4 = self.N4Up(torch.cat([x_downsample[0], x_upn1[0], x_upn2[0], x_upn3[0], self.Up(x_upn3[1])], dim=1))

        if self.deep_supervision:
            output1 = self.Output1(x_upn1[0])
            output2 = self.Output1(x_upn2[0])
            output3 = self.Output1(x_upn3[0])
            output4 = self.Output1(x_upn4)
            output = (output1 + output2 + output3 + output4) / 4
        else:
            output = self.Output(x_upn4)
        #output = self.sigmoid(output)

        return output

class Unet(nn.Module):
    def __init__(self, in_channels=1, classes=1, deep_supervision=True, up_mode='bilinear', input_shape=(224, 224)):
        super(Unet, self).__init__()
        self.output_shape = (1, input_shape[0], input_shape[1])
        self.channels = in_channels
        self.classes = classes
        self.filtersize = [34,64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.depth = 4
        self.up_mode = up_mode
        #Max pooling and Upsampling init
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upscale = nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)

        self.down = nn.ModuleList()
        self.Up = nn.ModuleList()
        self.bottleNeck = nn.ModuleList()
        self.down.append(plus_block(self.channels, self.filtersize[0]))

        self.bottle_depth = 2
        #Downsampling
        for i_layer in range(self.depth):
            layer_down = plus_block(self.filtersize[i_layer], self.filtersize[i_layer+1])
            self.down.append(layer_down)

        #BottleNeck
        num = 1
        for layer in range(self.bottle_depth):
            bottleNeck = nn.Sequential(
               nn.Conv2d(self.filtersize[-1] * 2**(num - 1), self.filtersize[-1] * 2**(2 - num), kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(self.filtersize[-1] * 2**(2 - num)),
               nn.ReLU(inplace=True)
            )
            self.bottleNeck.append(bottleNeck)
            num += 1
        # Upsampling
        for i_layer in range(self.depth):
            layer_Up = plus_block((self.filtersize[self.depth - i_layer] + self.filtersize[self.depth - i_layer]), self.filtersize[self.depth - i_layer - 1])
            self.Up.append(layer_Up)

        self.Output = OutConv(self.filtersize[0], classes)

    def forward(self, x):
        x_downsample = []
        for idx, layer in enumerate(self.down):
            if idx == 0:
                x = layer(x)
            else:
                x = layer(self.max_pool(x))
            x_downsample.append(x)

        for layer in self.bottleNeck:
            x = layer(x)

        for idx, layer in enumerate(self.Up):
            x = layer(torch.cat([x, x_downsample[(-idx-1)]], dim=1))
            x = self.Upscale(x)
        output = self.Output(x)

        return output


class SimpleDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(SimpleDiscriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)