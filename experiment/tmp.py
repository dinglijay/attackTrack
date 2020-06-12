


        pert = torch.tensor(pert, requires_grad=True)
        # define momentum
        g_old = torch.zeros_like(pert)
        lr_moment = 1.0
        alpha = 1.0 #perturbation step-size

        ## after loss.backward(),  get gradient wrt pert, use momentum to update gradient estimates
        normalized_grad = pert.grad.data / torch.sum(torch.abs(pert.grad.data))
        g_new = lr_moment * g_old + normalized_grad
        pert.data = pert.data + alpha * g_new.sign()
        g_old = g_new
        pert.grad.data = torch.zeros_like(pert)
        
        # fig, ax = plt.subplots(1,1,num='bbox')
        # ax.imshow(kornia.tensor_to_image(template_img.byte()))
        # x, y, w, h = template_bbox.squeeze().cpu().numpy()
        # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # x, y, w, h = bbox_pert_temp.squeeze().cpu().numpy()
        # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        # ax.add_patch(rect)  
        # plt.show()

        # # #########################################################33
        # fig, axes = plt.subplots(2,2,num='imgs')
        # ax = axes[0,0]
        # ax.set_title('template_img')
        # ax.imshow(kornia.tensor_to_image(template_img.byte()))
        # x, y, w, h = template_bbox.squeeze().cpu().numpy()
        # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        
        # ax = axes[0,1]
        # ax.set_title('search_img')
        # ax.imshow(kornia.tensor_to_image(search_img.byte()))
        # x, y, w, h = search_bbox.squeeze().cpu().numpy()
        # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect) 

        # ax = axes[1,0]
        # ax.set_title('template_mask')
        # ax.imshow(kornia.tensor_to_image((template_img*mask_template).byte()))
        # x, y, w, h = bbox_pert_temp.squeeze().cpu().numpy()
        # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        # ax.add_patch(rect)

        # ax = axes[1,1]
        # ax.set_title('search_img')
        # ax.imshow(kornia.tensor_to_image((search_img*mask_search).byte()))
        # x, y, w, h = bbox_pert_xcrop.squeeze().cpu().numpy()
        # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()
        