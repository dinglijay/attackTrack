import torch
from torch import nn

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


if __name__=='__main__':
    import cv2
    import kornia
    from matplotlib import pyplot as plt

    patch = cv2.imread('data/patchnew0.jpg')
    patch = kornia.image_to_tensor(patch).to(torch.float).unsqueeze(dim=0)

    # ColorJitter
    trans = kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1)
    fig, axes = plt.subplots(1,2,num=type(trans).__name__)
    for i in range(100):
        res = trans(patch/255.0)
        ax = axes[0]
        ax.set_title('patch')
        ax.imshow(kornia.tensor_to_image(patch.byte()))
        ax = axes[1]
        ax.set_title('res')
        ax.imshow(kornia.tensor_to_image(res))
        plt.pause(0.001)

    # RandomMotionBlur
    # trans = kornia.augmentation.RandomMotionBlur(9, 35, 0.5)
    # res = trans(patch)
    # fig, axes = plt.subplots(1,2,num=type(trans).__name__)
    # ax = axes[0]
    # ax.set_title('patch')
    # ax.imshow(kornia.tensor_to_image(patch.byte()))
    # ax = axes[1]
    # ax.set_title('res')
    # ax.imshow(kornia.tensor_to_image(res.byte()))

    # # MedianBlur
    # trans = kornia.filters.MedianBlur((3, 3))
    # res = trans(patch)
    # fig, axes = plt.subplots(1,2,num=type(trans).__name__)
    # ax = axes[0]
    # ax.set_title('patch')
    # ax.imshow(kornia.tensor_to_image(patch.byte()))
    # ax = axes[1]
    # ax.set_title('res')
    # ax.imshow(kornia.tensor_to_image(res.byte()))

    # # RandomRotation
    # trans = kornia.augmentation.RandomRotation(degrees=45)
    # for i in range(100):
    #     res = trans(patch)
    #     fig, axes = plt.subplots(1,2,num=type(trans).__name__)
    #     ax = axes[0]
    #     ax.set_title('patch')
    #     ax.imshow(kornia.tensor_to_image(patch.byte()))
    #     ax = axes[1]
    #     ax.set_title('res')
    #     ax.imshow(kornia.tensor_to_image(res.byte()))
    #     plt.pause(0.001)
    plt.show()


    
