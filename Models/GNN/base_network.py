"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """
    def __init__(self, in_channels=128, out_channels=128, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_channels, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_channels * (len(sizes)//4 + 1), out_channels)
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def _make_stage_1(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False)
        bn = nn.InstanceNorm2d(in_channels//4, affine=False)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.InstanceNorm2d(in_channels, affine=False)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        out = self.final(bottle)
        
        return out


class MyResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, use_instance_norm=True):
        super(MyResBlock, self).__init__()
        #self.pad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.use_instance_norm = use_instance_norm
        if self.use_instance_norm:
            self.in1 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)
        #self.pad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_instance_norm:
            self.in2 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)

    def forward(self, x):
        residual = x
        #out = self.pad1(x)
        out = self.conv1(x)
        if self.use_instance_norm:
            out = self.in1(out)
        out = F.relu(out)
        #out = self.pad2(out)
        out = self.conv2(out)
        if self.use_instance_norm:
            out = self.in2(out)
        out += residual
        out = F.relu(out)
        return out

class MyEncoder(nn.Module):
    def __init__(self, nr_channel=128, input_channel=3, use_instance_norm=True):
        
        super(MyEncoder, self).__init__()
        print('==> %d input channel' % (input_channel))
        
        self.input_channel = input_channel
        self.additional_input = self.input_channel > 3
        if not use_instance_norm:
            self.first_base_module = nn.Sequential(OrderedDict([

                ('base_conv1', nn.Conv2d(3, nr_channel, kernel_size=7, stride=1, padding=3, bias=False)),
                # ('base_in1', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
                ('base_relu1', nn.ReLU()),

                ('base_res2', MyResBlock(nr_channel, nr_channel, use_instance_norm)),
                ('base_res3', MyResBlock(nr_channel, nr_channel, use_instance_norm)),
                ('base_res4', MyResBlock(nr_channel, nr_channel, use_instance_norm)),
                ('base_res5', MyResBlock(nr_channel, nr_channel, use_instance_norm)),
                ('base_res6', MyResBlock(nr_channel, nr_channel, use_instance_norm)),

                ('base_conv5', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False)),
                # ('base_in5', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
            ]))
        else:
            self.first_base_module = nn.Sequential(OrderedDict([

                ('base_conv1', nn.Conv2d(3, nr_channel, kernel_size=7, stride=1, padding=3, bias=False)),
                ('base_in1', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
                ('base_relu1', nn.ReLU()),

                ('base_res2', MyResBlock(nr_channel, nr_channel)),
                ('base_res3', MyResBlock(nr_channel, nr_channel)),
                ('base_res4', MyResBlock(nr_channel, nr_channel)),
                ('base_res5', MyResBlock(nr_channel, nr_channel)),
                ('base_res6', MyResBlock(nr_channel, nr_channel)),

                ('base_conv5', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False)),
                ('base_in5', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
            ]))
        
        if self.additional_input:
            if not use_instance_norm:
                self.first_add_module = nn.Sequential(OrderedDict([
                    ('add_conv6', nn.Conv2d(nr_channel + self.input_channel - 3, nr_channel, kernel_size=1, stride=1, bias=False)),
                    # ('add_in6', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
                    ('add_relu6', nn.ReLU()),
                    ('add_conv7', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, bias=False)),
                    # ('add_in7', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
                ]))
            else:
                self.first_add_module = nn.Sequential(OrderedDict([
                    ('add_conv6',
                     nn.Conv2d(nr_channel + self.input_channel - 3, nr_channel, kernel_size=1, stride=1, bias=False)),
                    ('add_in6', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
                    ('add_relu6', nn.ReLU()),
                    ('add_conv7', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, bias=False)),
                    ('add_in7', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
                ]))

        if not use_instance_norm:

            self.first_branch = nn.Sequential(OrderedDict([
                ('fb_res8', MyResBlock(nr_channel, nr_channel)),
                ('fb_conv9', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False)),
                # ('fb_in9', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
            ]))
        else:
            self.first_branch = nn.Sequential(OrderedDict([
                ('fb_res8', MyResBlock(nr_channel, nr_channel)),
                ('fb_conv9', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False)),
                ('fb_in9', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)),
            ]))

    def forward(self, x):
        if self.additional_input:
            first_image = x[:, :3, :, :]
        else:
            first_image = x

        first_out = self.first_base_module(first_image)
        
        if self.additional_input:
            first_add = x[:, 3:, :, :]
            first_out = torch.cat((first_out, first_add), 1)
            first_out = self.first_add_module(first_out)

        first_out = F.relu(first_out)
        first_branch_out = self.first_branch(first_out)
        return first_branch_out, first_branch_out


