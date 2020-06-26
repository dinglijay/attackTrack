import torch
from torch import nn

if __name__=='__main__':
    import cv2
    import kornia
    from matplotlib import pyplot as plt

    patch = cv2.imread('data/patchnew0.jpg')
    patch = kornia.image_to_tensor(patch).to(torch.float)#.unsqueeze(dim=0)

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

    # # RandomAffine
    # trans = kornia.augmentation.RandomAffine(degrees=10, translate=[-0.2, 0.2],scale=[0.5, 0.7], shear=[-10, 10])
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


    
