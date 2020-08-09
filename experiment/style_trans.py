import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import cv2
import kornia

class Normalization(nn.Module):
    def __init__(self, device='cuda'):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def gram_matrix(input):
    B, C, H, W = input.size() 
    features = input.view(B, C, H * W) 
    G = torch.bmm(features, features.transpose(1, 2) ) # (B, C, C)
    return G.div(C * H * W) # normalize the values of the gram matrix

def image_loader(image_name, imsize=None):
    img = cv2.imread(image_name) / 255.0
    if imsize:
        img = cv2.resize(img, (imsize[1], imsize[0]))
    img = kornia.image_to_tensor(img).unsqueeze(0)
    return img.to(device, torch.float)

def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization().to(device)

    # just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
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

def imshow(tensor, title=None):
    image = kornia.tensor_to_image(tensor)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1e1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    # optimizer = optim.Adam([input_img.requires_grad_()], lr=1e-2)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        # input_img.data.clamp_(0, 1)

        # optimizer.zero_grad()
        # model(input_img)
        # style_score = 0
        # content_score = 0

        # for sl in style_losses:
        #     style_score += sl.loss
        # for cl in content_losses:
        #     content_score += cl.loss

        # style_score *= style_weight
        # content_score *= content_weight

        # loss = style_score + content_score
        # loss.backward()
        # optimizer.step(closure)

        # run[0] += 1
        # if run[0] % 50 == 0:
        #     print("run {}:".format(run))
        #     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
        #         style_score.item(), content_score.item()))
        #     print()

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
    
if __name__ == "__main__":
        
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_img = image_loader("data/styleTrans/style1.jpg")
    content_img = image_loader("data/styleTrans/tennis_person.png", imsize=style_img.shape[-2:])
    input_img = content_img.clone()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    output = run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300)

    plt.figure()
    imshow(output, title='Output Image')
    plt.show()
