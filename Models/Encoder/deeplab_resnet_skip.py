from Models.Encoder.DeepResNet import resnet101, Res_Deeplab
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchvision.transforms as transforms
from copy import deepcopy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetHead(nn.Module):
    def __init__(self, input_dim=128, final_dim=128):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            final_dim: (int): write your description
        """
        super(NetHead, self).__init__()
        conv_final_1 = nn.Conv2d(input_dim, final_dim, kernel_size=3, padding=1, bias=False)
        bn_final_1 = nn.BatchNorm2d(final_dim)
        relu_final_1 = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(conv_final_1, bn_final_1, relu_final_1)

    def my_load_state_dict(self, state_dict, strict=True):
        """
        : param state dictionary from a dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
            strict: (bool): write your description
        """
        # copy first conv
        new_state_dict = {}
        for k in state_dict:
            if 'encoder.0' in k:
                new_v = torch.zeros_like(self.encoder[0].weight.data)
                old_v = state_dict[k]
                new_v[:,:old_v.shape[1],:,:] = deepcopy(old_v)
                for i in range(old_v.shape[1], new_v.shape[1]):
                    new_v[:, i,:,:] = deepcopy(old_v[:, -1,:,:])
            else:
                new_v = state_dict[k]
            new_state_dict[k] = new_v
        # import ipdb
        # ipdb.set_trace()
        self.load_state_dict(new_state_dict, strict=strict)
    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.encoder(x)

class DeepLabResnet(nn.Module):
    def __init__(self, concat_channels=64,
                 final_dim=128,
                 final_resolution=(224, 224),
                 input_channel=3,
                 classifier="psp",
                 n_classes=1,
                 reload=True,
                 cnn_feature_grids=None,
                 normalize_input=True,
                 use_final=True,
                 update_last=False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            concat_channels: (todo): write your description
            final_dim: (int): write your description
            final_resolution: (todo): write your description
            input_channel: (todo): write your description
            classifier: (str): write your description
            n_classes: (todo): write your description
            reload: (todo): write your description
            cnn_feature_grids: (todo): write your description
            normalize_input: (bool): write your description
            use_final: (bool): write your description
            update_last: (todo): write your description
        """

        super(DeepLabResnet, self).__init__()
        self.use_final = use_final
        self.input_channel = input_channel
        # Default transform for all torchvision models
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.cnn_feature_grids =cnn_feature_grids
        self.concat_channels = concat_channels
        self.final_resolution = final_resolution
        self.final_dim = final_dim
        self.feat_size = 28
        self.reload = reload
        self.update_last = update_last

        self.image_feature_dim = 256
        self.n_classes = n_classes
        self.classifier = classifier
        self.normalize_input = normalize_input
        print('Input Channel is %d'%(input_channel))
        self.resnet = resnet101(n_classes, input_channel=input_channel, classifier=classifier)

        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)

        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)

        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)

        self.res1_concat = nn.Sequential(concat2, bn2, relu2)

        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)

        self.res2_concat = nn.Sequential(concat3, bn3, relu3)

        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)

        self.res4_concat = nn.Sequential(concat4, bn4, relu4)

        self.layer5_concat_channels = concat_channels

        concat5 = nn.Conv2d(512, self.layer5_concat_channels, kernel_size=3, padding=1, bias=False)
        bn5 = nn.BatchNorm2d(self.layer5_concat_channels)
        relu5 = nn.ReLU(inplace=True)

        self.res5_concat = nn.Sequential(concat5, bn5, relu5)
        if self.use_final:
            conv_final_1 = nn.Conv2d(5 * concat_channels, final_dim, kernel_size=3, padding=1, bias=False)
            bn_final_1 = nn.BatchNorm2d(final_dim)
            relu_final_1 = nn.ReLU(inplace=True)
            self.final = nn.Sequential(conv_final_1, bn_final_1, relu_final_1)

        '''
        conv_final_1 = nn.Conv2d(4*concat_channels, 128, kernel_size=3, padding=1, stride=2,
            bias=False)
        bn_final_1 = nn.BatchNorm2d(128)
        conv_final_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False)
        bn_final_2 = nn.BatchNorm2d(128)
        conv_final_3 = nn.Conv2d(128, final_dim, kernel_size=3, padding=1, bias=False)
        bn_final_3 = nn.BatchNorm2d(final_dim)

        self.conv_final = nn.Sequential(conv_final_1, bn_final_1, conv_final_2, bn_final_2,
            conv_final_3, bn_final_3)

        if self.classifier != 'psp' :
            self.final_dim = 64 * 4
        else:
            self.final_dim = 64 * 5
        '''
        self.final_upsample = nn.Upsample(size=self.final_resolution, mode='bilinear', align_corners=True)
        self.mid_upsample = nn.Upsample(size=(self.final_resolution[0] // 2, self.final_resolution[1] // 2),
                                        mode='bilinear', align_corners=True)
        if self.reload:
            self.reload_model()

    def reload_model(self, path='/scratch/ssd001/home/jungao/pretrained_model/MS_DeepLab_resnet_trained_VOC.pth'):
        """
        Reloads a model

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        if not os.path.exists(path):
            path = '/scratch/gobi1/jungao/pretrained_model/MS_DeepLab_resnet_trained_VOC.pth'
        if not os.path.exists(path):
            path = '/u/jungao/pretrain_w/MS_DeepLab_resnet_trained_VOC.pth'
        model_full = Res_Deeplab(self.n_classes, pretrained=True, reload_path=path).to(device)
        self.resnet.load_pretrained_ms(model_full, input_channel=self.input_channel)
        del model_full

    def forward(self, x):
        """
        Perform computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        batch_size = x.shape[0]
        assert self.classifier == 'psp' # TODO: we're always using psp net
        if self.normalize_input:
            x = self.normalize(x)
        conv1_f, layer1_f, layer2_f, layer3_f, layer4_f, layer5_f = self.resnet(x)
        if self.update_last:
            # torch.Size([1, 64, 112, 112])
            conv1_f_gcn = self.mid_upsample(self.conv1_concat(conv1_f))
            # torch.Size([1, 64, 56, 56])
            layer1_f_gcn = self.mid_upsample(self.res1_concat(layer1_f))
            # torch.Size([1, 64, 28, 28])
            layer2_f_gcn = self.mid_upsample(self.res2_concat(layer2_f))
            # torch.Size([1, 64, 28, 28])
            layer4_f_gcn = self.mid_upsample(self.res4_concat(layer4_f))
            layer5_f_gcn = self.mid_upsample(self.res5_concat(layer5_f))
        else:
            # torch.Size([1, 64, 112, 112])
            conv1_f_gcn = self.final_upsample(self.conv1_concat(conv1_f))
            # torch.Size([1, 64, 56, 56])
            layer1_f_gcn = self.final_upsample(self.res1_concat(layer1_f))
            # torch.Size([1, 64, 28, 28])
            layer2_f_gcn = self.final_upsample(self.res2_concat(layer2_f))
            # torch.Size([1, 64, 28, 28])
            layer4_f_gcn = self.final_upsample(self.res4_concat(layer4_f))
            layer5_f_gcn = self.final_upsample(self.res5_concat(layer5_f))

        final_features = [conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn, layer5_f_gcn]
        final_cat_features = torch.cat(final_features, dim=1)
        if self.use_final:
            final_features = self.final(final_cat_features)
        else:
            final_features = final_cat_features
        return final_features, final_cat_features

    def normalize(self, x):
        """
        Normalize the layer.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """

        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            # only normalize first three channel, the last channel (dextr) will remain unchanged.
            x[:3] = self.normalizer(x[:3])
            out.append(x)

        return torch.stack(out, dim=0)

    def my_load_state_dict(self, state_dict, strict=True):
        """
        Loads a dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
            strict: (bool): write your description
        """
        # copy first conv
        new_state_dict = {}
        for k in state_dict:
            if 'resnet.conv1' in k:
                new_v = torch.zeros_like(self.resnet.conv1.weight.data)
                old_v = state_dict[k]
                new_v[:,:old_v.shape[1],:,:] = deepcopy(old_v)
                for i in range(old_v.shape[1], new_v.shape[1]):
                    new_v[:, i,:,:] = deepcopy(old_v[:, -1,:,:])
            else:
                new_v = state_dict[k]
            new_state_dict[k] = new_v
        self.load_state_dict(new_state_dict, strict=strict)
