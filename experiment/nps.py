import torch
import numpy as np


class NPSCalculator(torch.nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_sz):
        super(NPSCalculator, self).__init__()
        self.printability_array = torch.nn.Parameter(self.get_printability_array(printability_file, patch_sz),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, patch_sz):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full(patch_sz, red))
            printability_imgs.append(np.full(patch_sz, green))
            printability_imgs.append(np.full(patch_sz, blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa