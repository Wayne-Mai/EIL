
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F

import queue
# from utils.utils_cuda_wrapper import *
import pickle

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

layer_mapping_vgg = OrderedDict([('features.0.weight', 'conv1_1.weight'), ('features.0.bias', 'conv1_1.bias'), ('features.2.weight', 'conv1_2.weight'), ('features.2.bias', 'conv1_2.bias'), ('features.5.weight', 'conv2_1.weight'), ('features.5.bias', 'conv2_1.bias'), ('features.7.weight', 'conv2_2.weight'), ('features.7.bias', 'conv2_2.bias'), ('features.10.weight', 'conv3_1.weight'), ('features.10.bias', 'conv3_1.bias'), ('features.12.weight', 'conv3_2.weight'), ('features.12.bias', 'conv3_2.bias'), ('features.14.weight', 'conv3_3.weight'), (
    'features.14.bias', 'conv3_3.bias'), ('features.17.weight', 'conv4_1.weight'), ('features.17.bias', 'conv4_1.bias'), ('features.19.weight', 'conv4_2.weight'), ('features.19.bias', 'conv4_2.bias'), ('features.21.weight', 'conv4_3.weight'), ('features.21.bias', 'conv4_3.bias'), ('features.24.weight', 'conv5_1.weight'), ('features.24.bias', 'conv5_1.bias'), ('features.26.weight', 'conv5_2.weight'), ('features.26.bias', 'conv5_2.bias'), ('features.28.weight', 'conv5_3.weight'), ('features.28.bias', 'conv5_3.bias')])


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def get_model(pretrained=False, progress=True, **kwargs):
    kwargs['init_weights'] = True
    kwargs['drop_rate'] = 0.75
    kwargs['drop_thr'] = 0.8
    model = VGG(features=None, **kwargs)
    model_dict = model.state_dict()
    if pretrained:

        pretrained_dict = models.vgg16(pretrained=True).state_dict()
        # state_dict = load_url(model_urls[arch], progress=progress)
        pretrained_dict = remove_layer(pretrained_dict, 'classifier.')

        for pretrained_k in pretrained_dict:
            if pretrained_k not in layer_mapping_vgg.keys():
                continue

            my_k = layer_mapping_vgg[pretrained_k]
            if my_k not in model_dict.keys():
                my_k = "module."+my_k
            if my_k not in model_dict.keys():
                raise Exception("Try to load not exist layer...")
            model_dict[my_k] = pretrained_dict[pretrained_k]
            # print("corresponding\t",my_k,"\t",pretrained_k)

    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(model_dict)
    return model




class EIL(nn.Module):
    def __init__(self, drop_rate=0.75, drop_thr=0.9):
        super(EIL, self).__init__()
        assert 0 <= drop_rate <= 1 and 0 <= drop_thr <= 1
        self.drop_rate = drop_rate
        self.drop_thr = drop_thr

        self.attention = None
        self.drop_mask = None

    def extra_repr(self):
        return 'drop_rate={}, drop_thr={}'.format(
            self.drop_rate, self.drop_thr
        )

    def forward(self, x):
        b = x.size(0)
        attention = torch.mean(x, dim=1, keepdim=True)
        self.attention = attention
        max_val, _ = torch.max(attention.view(b, -1), dim=1, keepdim=True)
        thr_val = max_val * self.drop_thr
        thr_val = thr_val.view(b, 1, 1, 1).expand_as(attention)
        drop_mask = (attention < thr_val).float()
        self.drop_mask = drop_mask
        output = x.mul(drop_mask)
        return output

    def get_maps(self):
        return self.attention, self.drop_mask


class VGG(nn.Module):
    def __init__(self, features=None, num_classes=1000, mode='base', init_weights=True, **kwargs):
        super(VGG, self).__init__()

        self.mode = mode
        self.bn = kwargs['batch_norm']
        self.num_classes = num_classes
        self.cam = None
        self.criterion = nn.CrossEntropyLoss()
        self.feature_map = None
        self.feature_map_erase = None
        self.score_erase = None
        self.score_erase5 = None
        self.score_erase4 = None
        self.score_erase3 = None
        self.score_erase2 = None
        self.feat = None
        self.erased_maps = []

        self.eil3 = EIL(drop_thr=0.7)
        self.eil4 = EIL(drop_thr=0.7)
        self.eil5 = EIL(drop_thr=0.7)



        # 64 x 224 x 224
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)

        # 64 x 224 x 224
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # 64 x 112 x 112
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128 x 112 x 112
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)

        # 128 x 112 x 112
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        # 128 x 56 x 56
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256 x 56 x 56
        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)

        # 256 x 28 x 28
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512 x 28 x 28
        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)

        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)

        self.conv4_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)

        #  512 x 14 x 14
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512 x 14 x 14
        self.conv5_1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)

        self.conv5_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv5_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)


        # above is backbone network
        if self.mode == 'base':

            self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(1024, self.num_classes)

        # Weight initialization
        if init_weights:
            self._initialize_weights()


    def forward(self, source, label=None):
        x = self.conv1_1(source)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)

        # below code --- don't touch any more

        x = self.conv2_1(x)
        x = self.relu2_1(x)

        x = self.conv2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)


        x = self.conv3_1(x)
        x = self.relu3_1(x)

        x = self.conv3_2(x)
        x = self.relu3_2(x)

        x = self.conv3_3(x)
        x = self.relu3_3(x)

        x = self.pool3(x)


        x = self.conv4_1(x)
        x = self.relu4_1(x)

        x = self.conv4_2(x)
        x = self.relu4_2(x)

        x = self.conv4_3(x)
        x = self.relu4_3(x)

        x = self.pool4(x)

        pool4 = self.eil4(x)  

        pool4 = self.conv5_1(pool4)
        pool4 = self.relu5_1(pool4)


        pool4 = self.conv5_2(pool4)
        pool4 = self.relu5_2(pool4)


        pool4 = self.conv5_3(pool4)
        pool4 = self.relu5_3(pool4)


        pool4 = self.conv6(pool4)
        pool4 = self.relu(pool4)



        pool4 = self.avgpool(pool4)
        pool4 = pool4.view(pool4.size(0), -1)

        pool4 = self.fc(pool4)
        self.score_erase4 = pool4

        unerased = self.conv5_1(x)
        unerased = self.relu5_1(unerased)

        self.unerased_conv51=unerased

        unerased = self.conv5_2(unerased)
        unerased = self.relu5_2(unerased)

        self.unerased_conv52=unerased

        unerased = self.conv5_3(unerased)
        unerased = self.relu5_3(unerased)

        self.unerased_conv53=unerased

        if self.mode == 'base':
            unerased = self.conv6(unerased)
            unerased = self.relu(unerased)

            self.feature_map = unerased
            unerased = self.avgpool(unerased)
            unerased = unerased.view(unerased.size(0), -1)
            unerased = self.fc(unerased)
            self.score = unerased
            return self.score

    def get_feature_map(self, target=None):
        return self.feature_map, self.score


    def get_fused_cam(self, target=None):
        if target is None:
            target = self.score.topk(1, 1, True, True)
        if self.mode == 'base':
            batch, channel, _, _ = self.feature_map.size()
            fc_weight = self.fc.weight.squeeze()
            # get prediction in shape (batch)
            if target is None:
                _, target = score.topk(1, 1, True, True)
            target = target.squeeze()

            # get fc weight (num_classes x channel) -> (batch x channel)
            cam_weight = fc_weight[target]

            # get final cam with weighted sum of feature map and weights
            # (batch x channel x h x w) * ( batch x channel)
            cam_weight = cam_weight.view(
                batch, channel, 1, 1).expand_as(self.feature_map)
            cam = (cam_weight * self.feature_map)
            cam = cam.mean(1)

            return cam, self.score


    def get_loss(self, target, separate=False):
        if self.mode == 'base':
            if not separate:
                return self.criterion(self.score_erase4, target)+self.criterion(self.score, target)
            else:
                return self.criterion(self.score, target), self.criterion(self.score_erase4, target)

        elif self.mode == 'ACoL':
            return self.criterion(self.score, target), self.criterion(self.score_erase, target)
        elif self.mode == 'Recurrent':
            loss = self.criterion(self.score, target)


            N, C, H, W = self.feat.shape

            self.feature_map_erase = self.feature_map.detach().clone()

            erase_thr = [0.8, 0.8, 0.8]
            self.erased_maps = []
            for i in range(len(erase_thr)):

                attention_erase = self.feature_map_erase
                gt_attention_erase = torch.zeros(
                    [attention_erase.shape[0], attention_erase.shape[2], attention_erase.shape[3]]).cuda()
                for batch_idx in range(N):
                    gt_attention_erase[batch_idx, :, :] = torch.squeeze(
                        attention_erase[batch_idx, target[batch_idx], :, :].clone().detach())

                max_val, _ = torch.max(gt_attention_erase.view(
                    gt_attention_erase.shape[0], -1), dim=1)
                # set mask
                erase_mask = gt_attention_erase < (
                    max_val.view(-1, 1, 1))*erase_thr[i]
                # mul NOTE
                self.feat = self.feat*erase_mask.unsqueeze(1).float()
                # feed forward
                self.feature_map_erase = self.classifier(
                    self.extraReLU(self.extraConv(self.feat)))

                self.score_erase = self.avgpool(
                    self.feature_map_erase).view(self.feature_map.shape[0], -1)
                # accumulate loss
                loss += (erase_thr[i]/(i+1)) * \
                    self.criterion(self.score_erase, target)

            self.erased_maps.append(
                gt_attention_erase.detach().clone())

            return loss

    # accept tensor in shape 3 only
    def do_normalize(self, heat_map):
        N, H, W = heat_map.shape

        # normalize
        batch_mins, _ = torch.min(heat_map.view(N, -1), dim=-1)
        batch_maxs, _ = torch.max(heat_map.view(N, -1), dim=-1)
        normalized_map = (heat_map-batch_mins.view(N, 1, 1)) / \
            (batch_maxs.view(N, 1, 1)-batch_mins.view(N, 1, 1))

        return normalized_map


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
