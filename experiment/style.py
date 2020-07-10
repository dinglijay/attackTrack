import cv2
import kornia
import torch
import torch.nn as nn
import torchvision.models as models

from style_trans import Normalization, ContentLoss, StyleLoss, image_loader

def image_loader(image_name):
    img = cv2.imread(image_name) / 255.0
    img = cv2.resize(img, (400, 200))
    img = kornia.image_to_tensor(img).unsqueeze(0)
    return img.to(torch.float)

def get_style_model_and_losses(device):

    style_img = image_loader("data/styleTrans/style2.jpg").to(device)
    content_img = image_loader("data/styleTrans/content.jpg").to(device)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    normalization = Normalization(device).to(device)
    model = nn.Sequential(normalization)

    # just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []

    content_layers=['conv_4']
    style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss and StyleLoss we insert below.
            # So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses