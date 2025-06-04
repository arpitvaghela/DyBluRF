import torch
import numpy as np
import os
import sys
import time
import models

from config import config_parser
from load_llff_data import *
from nerf_model import *
from spline import *
from render import *
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpipsPyTorch import lpips as lpips_model
from utils.loss_utils import ssim as ssim_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def evaluation():
    parser = config_parser()
    args = parser.parse_args()

    print('spline numbers: ', args.deblur_images)
    
    K = None
    if args.dataset_type == 'llff':
        images, poses_start, bds, \
        sharp_images, inf_images, \
        render_poses, ref_c2w, poses_train = load_llff_data_eva(args.datadir, args.start_frame, args.end_frame, 
                                                                target_idx=args.target_idx, recenter=True, 
                                                                bd_factor=.9, spherify=args.spherify, 
                                                                final_height=args.final_height)
        hwf = poses_start[0, :3,- 1]
        i_test = []
        i_val = [] 
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        
        poses = poses_start[:, :3, :4]
        poses_start = torch.Tensor(poses_start)
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = poses_start_se3
        poses = torch.Tensor(poses).to(device)

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'iphone_llff':
        images, poses_start, bds, \
        sharp_images, depths, \
        masks, motion_coords, \
        render_poses, ref_c2w = load_llff_data(args.datadir, args.start_frame, args.end_frame, 
                                               target_idx=args.target_idx, recenter=True, 
                                               bd_factor=.9, spherify=args.spherify, 
                                               final_height=args.final_height,is_iphone=True)
        hwf = poses_start[0, :3,- 1]
        i_test = np.array([i for i in range(3, int(images.shape[0]), 4)])
        i_val = [] 
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        
        poses_start = torch.Tensor(poses_start)
        poses_end = poses_start
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = poses_start_se3
        poses_org = poses_start.repeat(args.deblur_images, 1, 1)
        poses = poses_org[:, :, :4]

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()
    H_old, W_old, focal = hwf
    H, W = 960, 720  # Scale up from 288, 216
    focal = focal * (H / H_old)  # Scale focal length proportionally
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
    
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    render_kwargs_train, render_kwargs_test, \
    _, _, _, _ = create_nerf(args, poses_start_se3, poses_end_se3, images.shape[0])
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    input_poses = torch.Tensor(poses[:, :3, :4])
    # input_poses_test = poses_start[:, :3, :4]

    num_img = float(images.shape[0])
    num = args.deblur_images    
   
    os.makedirs(os.path.join(basedir, expname, 'deblur'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'test'), exist_ok=True)
    
    save_deblur_path = os.path.join(basedir, expname, 'deblur')
    save_test_path = os.path.join(basedir, expname, 'test')
    
    with torch.no_grad():

        # model = models.PerceptualLoss(model='net-lin',net='alex',
        #                               use_gpu=True,version=0.1)

        deblur_total_psnr = 0.
        deblur_total_ssim = 0.
        deblur_total_lpips = 0.
        deblur_count = 0.

        test_total_psnr = 0.
        test_total_ssim = 0.
        test_total_lpips = 0.
        test_count = 0.
        
        t = time.time()

        # poses = se3_to_SE3(se3)
        # poses = torch.Tensor(poses)

        for i in range(0, int(num_img)):
            # print(time.time() - t)
            t = time.time()

            img_idx_embed = i * num + num // 2
            # img_idx_embed = img_idx_embed / (num_img * num - 1) * 2. - 1.0

            i_pose = convert3x4_4x4(input_poses[i])
            # input_test_pose = convert3x4_4x4(input_poses[i])
            if i in i_train:
                spline_poses = get_pose(args, i, render_kwargs_test['se3'])
                i_pose = convert3x4_4x4(torch.Tensor(spline_poses[num // 2]))

            # output_test_pose = input_test_pose @ torch.inverse(input_train_pose) @ output_train_pose

            ret = render_image_test(args, 0, img_idx_embed, num_img, i_pose[:3, :4], H, W, K, **render_kwargs_test)
            # ret = render_image_test(args, 0, img_idx_embed, num_img, input_poses_test[i], H, W, K, **render_kwargs_test)

            rgb = ret['rgb_map'].cpu().numpy()

            gt_img_path = os.path.join(args.datadir, 'sharp_images', '%05d.png'%i)
            gt_img = cv2.imread(gt_img_path)[:, :, ::-1]
            gt_img = cv2.resize(gt_img, dsize=(rgb.shape[1], rgb.shape[0]), 
                                interpolation=cv2.INTER_AREA)
            gt_img = np.float32(gt_img) / 255

            psnr = peak_signal_noise_ratio(gt_img, rgb)

            gt_img_0 = im2tensor(gt_img).cuda()
            rgb_0 = im2tensor(rgb).cuda()
            
            ssim = ssim_loss(gt_img_0, rgb_0).item()
            # lpips = model.forward(gt_img_0, rgb_0)
            lpips = lpips_model(rgb_0, gt_img_0, net_type="alex").item()
            # lpips = lpips.item()
            print(f"{'deblur ' if i in i_train else 'test '}", i, psnr, ssim, lpips, file=sys.stderr)

            if i in i_train:
                deblur_total_psnr += psnr
                deblur_total_ssim += ssim
                deblur_total_lpips += lpips
                deblur_count += 1

                filename = os.path.join(save_deblur_path, 'rgb_{}.jpg'.format(i))
                imageio.imwrite(filename, (rgb * 255).astype(np.uint8))
                filename = os.path.join(save_deblur_path, 'rgb_{}_gt.jpg'.format(i))
                imageio.imwrite(filename, (gt_img * 255).astype(np.uint8))

            elif i in i_test:
                test_total_psnr += psnr
                test_total_ssim += ssim
                test_total_lpips += lpips
                test_count += 1

                filename = os.path.join(save_test_path, 'rgb_{}.jpg'.format(i))
                imageio.imwrite(filename, (rgb * 255).astype(np.uint8))
                filename = os.path.join(save_test_path, 'rgb_{}_gt.jpg'.format(i))
                imageio.imwrite(filename, (gt_img * 255).astype(np.uint8))    
            
        
        print("========== DEBLUR =================")
        deblur_mean_psnr = deblur_total_psnr / deblur_count
        deblur_mean_ssim = deblur_total_ssim / deblur_count
        deblur_mean_lpips = deblur_total_lpips / deblur_count

        print('mean_psnr ', deblur_mean_psnr)
        print('mean_ssim ', deblur_mean_ssim)
        print('mean_lpips ', deblur_mean_lpips)


        print("========== TEST =================")

        test_mean_psnr = test_total_psnr / test_count
        test_mean_ssim = test_total_ssim / test_count
        test_mean_lpips = test_total_lpips / test_count

        print('mean_psnr ', test_mean_psnr)
        print('mean_ssim ', test_mean_ssim)
        print('mean_lpips ', test_mean_lpips)




if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluation()