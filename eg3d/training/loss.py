# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

# from torchvision import transforms
import lpips
import dnnlib
from torch.autograd import Variable
import itertools
import PIL.Image

# ---------------------------------- arcface --------------------------------- #
from training.mystyle import Backbone
arcface_path = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
# ---------------------------------- magface --------------------------------- #
# from training.network_inf import builder_inf
# import torch.nn.functional as F
# from torchvision import transforms
# magface_path = '/playpen-nas-ssd/luchao/projects/eg3d/eg3d/networks/magface_epoch_00025.pth'
# ---------------------------------- adalora --------------------------------- #
# from training.adalora import compute_orth_regu

from camera_utils import LookAtPoseSampler
def get_lookat_pose_sampler_pair(camera_lookat_point, device):
    w_frames = 120
    frame_idx_1 = np.random.randint(0, w_frames // 2)
    frame_idx_2 = w_frames - frame_idx_1 - 1
    pitch_range = 0.25
    yaw_range = 0.35
    num_keyframes = 1
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)

    cam2world_pose_1 = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx_1 / (num_keyframes * w_frames)),
                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx_1 / (num_keyframes * w_frames)),
                                            camera_lookat_point, radius=2.7, device=device)
    cam2world_pose_2 = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx_2 / (num_keyframes * w_frames)),
                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx_2 / (num_keyframes * w_frames)),
                                            camera_lookat_point, radius=2.7, device=device)
    
    c_1 = torch.cat([cam2world_pose_1.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)
    c_2 = torch.cat([cam2world_pose_2.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)
    return c_1, c_2

def get_lookat_pose_sampler(camera_lookat_point, device):
    w_frames = 120
    frame_idx_1 = np.random.randint(0, w_frames // 2)
    frame_idx_2 = w_frames - frame_idx_1 - 1
    pitch_range = 0.25
    yaw_range = 0.35
    num_keyframes = 1
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx_1 / (num_keyframes * w_frames)),
                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx_1 / (num_keyframes * w_frames)),
                                            camera_lookat_point, radius=2.7, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)
    return c

def denorm(img):
    # img: [b, c, h, w]
    img = (img + 1) * 127.5
    # print('before clamp: ', img.requires_grad)
    img = img.permute(0, 2, 3, 1).clamp(0, 255)
    # print('after clamp: ', img.requires_grad)
    # img = img.to(torch.uint8)
    # print('after to uint8: ', img.requires_grad)
    return img # [b, h, w, c]

# for arcface
class PersonIdentifier:
    def __init__(self, model_path, num_examples, threshold):
        super().__init__()
        if model_path is None:
            raise ValueError('class cannot be init without path')
        # copied from mystyle
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').cuda()
        self.facenet.load_state_dict(torch.load(model_path))
        # ! 112x112 is required by arcface/magface
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112)).cuda()
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def transform_image(img):
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        x = torch.unsqueeze(img, 0)
        x = x / 127.5 - 1
        x = torch.clamp(x, -1, 1)
        if x.shape[-1] == 3:
            x = torch.permute(x, [0, 3, 1, 2])
        return x # [b, c, h, w]

    def get_feature(self, img):
        img = torch.clamp(img, -1, 1)
        if len(img.shape) == 3:
            # [c, h, w] -> [1, c, h, w]
            img = torch.unsqueeze(img, 0)

        # save img for sanity check
        # check_img = img[0].permute(1, 2, 0).clone().detach().cpu().numpy()
        # check_img = (check_img + 1) * 127.5
        # check_img = check_img.clip(0, 255).astype(np.uint8)
        # PIL.Image.fromarray(check_img).save('/playpen-nas-ssd/luchao/projects/eg3d/check_img.png')
        # print('saved check img')

        x = self.face_pool(img)
        x_feats = self.facenet(x)
        return x_feats

# for magface
# todo: takecare of RGB vs BGR order issue

#----------------------------------------------------------------------------

class Loss(PersonIdentifier):
    def __init__(self):
        # net type can be changed to alex/vgg depending on the scenario
        self.lpips = lpips.LPIPS(net='alex').eval() # eval_mode=True by default
        # mystyle
        self.person_identifier = PersonIdentifier(arcface_path, None, None)
        # magface
        # self.person_identifier = PersonIdentifier(magface_path)

    def calc_sim(self, imgs1, imgs2):
        # receive two tensors of shape [b, c, h, w]
        # return a tensor of shape [b]
        sims = torch.tensor([], device=imgs1.device)
        for img1, img2 in zip(imgs1, imgs2):
            feature_1 = self.person_identifier.get_feature(img1)
            feature_2 = self.person_identifier.get_feature(img2)
            cur_sim = torch.nn.functional.cosine_similarity(feature_1, feature_2)
            sims = torch.cat([sims, cur_sim], 0)
        return sims
    
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=128, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)


    def accumulate_gradients(self, phase, real_img, real_c, real_w, gen_z, gen_c, gain, cur_nimg):
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Gid_reg']
        assert phase.startswith('G') # only need to update G
        # if self.G.rendering_kwargs.get('density_reg', 0) == 0:
        #     if phase == 'Greg':
        #         phase = 'none'
        #     elif phase == 'Gboth':
        #         phase = 'Gmain'
            # phase = {'Greg': 'none', 'Gboth': 'Gmain', 'Gmultiview': 'Gmultiview'}.get(phase, phase)

        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial
        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw} # [b, c, h, w]
        # real_img['image'] is the original image (normalized) and real_img['image_raw'] is the downsampled-then-upsampled image (here we don't use it)
        # real_img_raw (64x64) is affected by neural_rendering_resolution_initial=64 by default (here we use 128)
        # print('real_img dictionary in loss.py')
        # print( (real_img['image'] + 1) * 127.5, real_img['image'].shape)

        # gen_img = self.G.synthesis(real_w, real_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=True) # [b, c, h, w]
        # print( (gen_img['image'] + 1) * 127.5, gen_img['image'].shape)

        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.G.synthesis(real_w, real_c, neural_rendering_resolution=neural_rendering_resolution, noise_mode='const')
                device = gen_img['image'].device

                real_img_low = real_img['image_raw']
                gen_img_low = gen_img['image_raw']
                real_img_high = real_img['image']
                gen_img_high = gen_img['image']
                
                l2_loss_low = torch.nn.MSELoss(reduction='none')(gen_img_low, real_img_low)
                l2_loss_low = l2_loss_low.mean(dim=tuple(range(1, len(l2_loss_low.shape))))
                l2_loss_high = torch.nn.MSELoss(reduction='none')(gen_img_high, real_img_high)
                l2_loss_high = l2_loss_high.mean(dim=tuple(range(1, len(l2_loss_high.shape))))

                lpips_loss_low = self.lpips.to(device).forward(gen_img_low, real_img_low).squeeze()
                lpips_loss_high = self.lpips.to(device).forward(gen_img_high, real_img_high).squeeze()
                
                # todo: beta hyperparameter for lpips low resolution loss - here we use 1 for now
                loss = (l2_loss_low + l2_loss_high) + (lpips_loss_low + lpips_loss_high) * 1
                loss_Gmain = loss.mean()

                training_stats.report(f'Loss/G/main', loss_Gmain)
                training_stats.report(f'Loss/G/gain_main', loss_Gmain.mul(gain))

                # l1_loss_low = torch.nn.L1Loss(reduction='none')(gen_img_low, real_img_low)
                # l1_loss_low = l1_loss_low.mean(dim=tuple(range(1, len(l1_loss_low.shape))))
                # l1_loss_high = torch.nn.L1Loss(reduction='none')(gen_img_high, real_img_high)
                # l1_loss_high = l1_loss_high.mean(dim=tuple(range(1, len(l1_loss_high.shape))))

                # lpips_loss_low = self.lpips.to(device).forward(gen_img_low, real_img_low).squeeze()
                # lpips_loss_high = self.lpips.to(device).forward(gen_img_high, real_img_high).squeeze()

                # loss = (l1_loss_low + l1_loss_high) + (lpips_loss_low + lpips_loss_high) * 1
                # loss_Gmain = loss


                # record identity loss
                sims = self.calc_sim(gen_img_high, real_img_high)
                loss = 1 - sims.mean()
                loss_id_reg = loss
                
                training_stats.report('Loss/G/id_reg', loss_id_reg)
                training_stats.report(f'Loss/G/gain_id_reg', loss_id_reg.mul(gain))

                # adalora
                # loss_Reg = compute_orth_regu(self.G, regu_weight=0.1)
                # training_stats.report('Loss/G/rank_reg', loss_Reg.mean())
                # training_stats.report(f'Loss/G/gain_rank_reg', loss_Reg.mean().mul(gain))
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()

                # # regularization for identity
                # loss = loss_Gmain.mul(gain) + loss_id_reg.mul(gain)
                # loss.backward()
                # # loss_Reg.mean().mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            gen_c = real_c.clone()
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
            # ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False) # gen_z is torch.Size([batch_gpu, 512]) ws is (batch_gpu, 14, 512)
            ws = real_w
            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()
            # training_stats.report('Loss/G/TVloss', TVloss)
            training_stats.report(f'Loss/G/density_reg', TVloss)
            training_stats.report(f'Loss/G/gain_density_reg', TVloss.mul(gain))

        if phase in ['Gid_reg']:
            # regularization for identity
            device = self.device
            gen_img = self.G.synthesis(real_w, real_c, neural_rendering_resolution=neural_rendering_resolution, noise_mode='const')['image']
            real_img = real_img['image']
            sims = self.calc_sim(gen_img, real_img)
            loss = 1 - sims.mean()
            loss_id_reg = loss
            
            gain = 1
            loss_id_reg.mean().mul(gain).backward()
            training_stats.report('Loss/G/id_reg', loss_id_reg.mean())
            training_stats.report(f'Loss/G/gain_id_reg', loss_id_reg.mean().mul(gain))


        if phase in ['Gsym_reg']:
            # regularization for symmetric views
            device = self.device            
            camera_lookat_point = torch.tensor([0, 0, 0], device=device)

            # ------------------------------ symmetric view ------------------------------ #
            
            c1, c2 = get_lookat_pose_sampler_pair(camera_lookat_point,device=device)
            # repeat c1 and c2 to match c_current
            c1 = c1.repeat(real_c.shape[0], 1)
            c2 = c2.repeat(real_c.shape[0], 1)

            # todo: check noise_mode for G.synthesis (const?)
            gen_img = self.G.synthesis(real_w, c1, neural_rendering_resolution=neural_rendering_resolution, noise_mode='const')
            gen_img_1 = gen_img['image'] # [b, c, h, w]
            gen_img = self.G.synthesis(real_w, c2, neural_rendering_resolution=neural_rendering_resolution, noise_mode='const')
            gen_img_2 = gen_img['image'] # [b, c, h, w]

            # ------------------------------- sanity check ------------------------------- #
            # print('gen_img_1.shape', gen_img_1.shape)
            # print('gen_img_2.shape', gen_img_2.shape)
            # save img1 and img2 for sanity check
            # check_img_1 = gen_img_1[0].permute(1, 2, 0).clone().detach().cpu().numpy()
            # check_img_1 = (check_img_1 + 1) * 127.5
            # check_img_1 = check_img_1.clip(0, 255).astype(np.uint8)
            # check_img_2 = gen_img_2[0].permute(1, 2, 0).clone().detach().cpu().numpy()
            # check_img_2 = (check_img_2 + 1) * 127.5
            # check_img_2 = check_img_2.clip(0, 255).astype(np.uint8)
            # check_img = np.concatenate([check_img_1, check_img_2], axis=1)
            # PIL.Image.fromarray(check_img).save('/playpen-nas-ssd/luchao/projects/eg3d/check_img_sym.png')

            # flip the image left-right for calculating similarity
            gen_img_2 = torch.flip(gen_img_2, [3])
            # similarity between two symmetric views for the same sample
            sims = self.calc_sim(gen_img_1, gen_img_2)
            # print('id network parameters(mean):')
            # print(np.mean([p.detach().cpu().numpy().mean() for p in self.person_identifier.facenet.parameters()]))
            loss_same_sample = 1 - sims.mean()

            # ------------------------------- random poses ------------------------------- #

            # gen_img_real_c = self.G.synthesis(real_w, real_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            # gen_img_real_c = denorm(gen_img_real_c['image']) # [b, h, w, c]
            # input_img = denorm(real_img['image']) # [b, h, w, c]
            # n_random_pose = 2
            # sims = torch.tensor([], device=device)
            # for _ in range(n_random_pose):
            #     random_c = get_lookat_pose_sampler(camera_lookat_point, device=device)
            #     random_c = random_c.repeat(real_w.shape[0], 1)
            #     gen_img = self.G.synthesis(real_w, random_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            #     gen_img_random = denorm(gen_img['image'])
            #     # normalize the effect of pose
            #     sim_1 = self.calc_sim(input_img, gen_img_random)
            #     sim_2 = self.calc_sim(gen_img_real_c, gen_img_random)
            #     sim = torch.mean(sim_1 / sim_2).unsqueeze(0)
            #     sims = torch.cat([sims, sim], 0)
            # # MSEloss between two random poses for the same sample
            # loss = torch.nn.MSELoss()(sims[0], sims[1])


            # random_c = get_lookat_pose_sampler(camera_lookat_point, device=device)
            # random_c = random_c.repeat(real_c.shape[0], 1)
            # gen_img = self.G.synthesis(real_w, random_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            # gen_img_random = denorm(gen_img['image'])
            # # normalize the effect of pose
            # sim_1 = self.calc_sim(input_img, gen_img_random)
            # sim_2 = self.calc_sim(gen_img_real_c, gen_img_random)
            # sims = torch.mean(sim_1 / sim_2)
            # loss = 1 - sims.mean()

            loss = loss_same_sample
            gain = 1
            loss.mean().mul(gain).backward()
            training_stats.report('Loss/G/sym_reg', loss.mean())
            training_stats.report(f'Loss/G/gain_sym_reg', loss.mean().mul(gain))

        if phase in ['Grank_reg']:
            # adalora
            loss = compute_orth_regu(self.G, regu_weight=0.1)
            loss.mean().mul(gain).backward()
            training_stats.report('Loss/G/rank_reg', loss.mean())
            training_stats.report(f'Loss/G/gain_rank_reg', loss.mean().mul(gain))

        if phase in ['Gpose']:
            # --------------- across different samples under the same pose --------------- #

            camera_lookat_point = torch.tensor([0, 0, 0], device=self.device)
            gen_c = get_lookat_pose_sampler(camera_lookat_point, device=self.device)
            gen_c = gen_c.repeat(real_c.shape[0], 1)
            gen_img = self.G.synthesis(real_w, gen_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            gen_img = denorm(gen_img['image']) # [b, h, w, c]
            sims = torch.tensor([], device=self.device)
            combinations = torch.combinations(torch.arange(real_w.shape[0]))
            for i, j in combinations:
                sims = torch.cat([sims, self.calc_sim(gen_img[i], gen_img[j])], 0)
            loss_same_pose = 1 - sims.mean()


            loss = loss_same_pose
            loss.mean().mul(gain).backward()
            training_stats.report(f'Loss/G/id_same_pose', loss.mean())
            training_stats.report('Loss/G/gain_id_same_pose', loss.mean().mul(gain))


        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()


#----------------------------------------------------------------------------
