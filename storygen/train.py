from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import pickle
from tqdm import tqdm
import json

from .config import cfg
from .utils import mkdir_p
from .utils import weights_init
from .utils import save_story_results, save_model, save_test_samples
from .utils import KL_loss
from .utils import compute_discriminator_loss, compute_generator_loss, compute_dual_densecap_loss
from shutil import copyfile
from torchvision.models import vgg16


class gan_trainer(object):
    def __init__(self, cfg, output_dir, ratio = 1.0):
        self.video_len = cfg.VIDEO_LEN
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        self.cfg = cfg
    
    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from .model import StoryGAN, STAGE1_D_IMG, STAGE1_D_STY_V2, StoryMartGAN
        netG = StoryGAN(self.cfg, self.video_len)
        netG.apply(weights_init)
        print(netG)
        netD_im = None
        netD_st = None
        if self.cfg.CUDA:
            netG.cuda()

        total_params = sum(p.numel() for p in netD_st.parameters() if p.requires_grad) + sum(
            p.numel() for p in netD_im.parameters() if p.requires_grad) + sum(
            p.numel() for p in netG.parameters() if p.requires_grad)
        print("Total Parameters: %s", total_params)

        return netG, netD_im, netD_st
    
    def sample_real_image_batch(self):
        """
        Iterate the ImageLoader
        """
        if self.imagedataset is None:
            self.imagedataset = enumerate(self.imageloader)
            batch_idx, batch = next(self.imagedataset)
        b = batch
        if self.cfg.CUDA:
            for k, v in batch.items():
                if k == 'text':
                    continue
                else:
                    b[k] = v.cuda()

        if batch_idx == len(self.imageloader) - 1:
            self.imagedataset = enumerate(self.imageloader)
        return b
        
    def train(self, imageloader, storyloader, testloader, stage=1):
        self.imageloader = imageloader
        self.imagedataset = None
        if stage == 1:
            netG, netD_im, netD_st = self.load_network_stageI()
        else:
            netG, netD_im, netD_st = self.load_network_stageII()
        
        im_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
        im_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
        st_real_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(1))
        st_fake_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(0))
        
        if self.cfg.CUDA:
            im_real_labels, im_fake_labels = im_real_labels.cuda(), im_fake_labels.cuda()
            st_real_labels, st_fake_labels = st_real_labels.cuda(), st_fake_labels.cuda()

        generator_lr = self.cfg.TRAIN.GENERATOR_LR
        discriminator_lr = self.cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = self.cfg.TRAIN.LR_DECAY_EPOCH

        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=self.cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
        
        loss_collector = []
        count = 0
        # save_test_samples(netG, testloader, self.test_dir, epoch=0, mart=self.use_mart)

        #save_test_samples(netG, testloader, self.test_dir)
        for epoch in range(self.max_epoch + 1):
            l = self.ratio * (2. / (1. + np.exp(-10. * epoch)) - 1)
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5

            for i, data in tqdm(enumerate(storyloader, 0)):
                ######################################################
                # (1) Prepare training data
                ######################################################
                im_batch = self.sample_real_image_batch() # im_batch iterate images
                st_batch = data 

                im_real_cpu = im_batch['images']
                im_motion_input = im_batch['description'][:, :self.cfg.TEXT.DIMENSION]
                im_content_input = im_batch['content'][:, :, :self.cfg.TEXT.DIMENSION]
                im_real_imgs = Variable(im_real_cpu)
                im_motion_input = Variable(im_motion_input)
                im_content_input = Variable(im_content_input)
                im_labels = Variable(im_batch['labels'])

                st_real_cpu = st_batch['images']
                st_motion_input = st_batch['description'][:, :, :self.cfg.TEXT.DIMENSION]
                st_content_input = st_batch['description'][:, :, :self.cfg.TEXT.DIMENSION]
                st_texts = st_batch['text']
                st_real_imgs = Variable(st_real_cpu)
                st_motion_input = Variable(st_motion_input)
                st_content_input = Variable(st_content_input)
                st_labels = Variable(st_batch['labels'])
                if self.use_mart or self.story_dual:
                    st_input_ids = Variable(st_batch['input_ids'])
                    st_masks = Variable(st_batch['masks'])

                if self.cfg.CUDA:
                    st_real_imgs = st_real_imgs.cuda()
                    im_real_imgs = im_real_imgs.cuda()
                    st_motion_input = st_motion_input.cuda()
                    im_motion_input = im_motion_input.cuda()
                    st_content_input = st_content_input.cuda()
                    im_content_input = im_content_input.cuda()
                    im_labels = im_labels.cuda()
                    st_labels = st_labels.cuda()
                    if self.use_mart or self.img_dual:
                        im_input_ids = im_input_ids.cuda()
                        im_masks = im_masks.cuda()
                    if self.story_dual or self.use_mart:
                        st_input_ids = st_input_ids.cuda()
                        st_masks = st_masks.cuda()

                im_motion_input = torch.cat((im_motion_input, im_labels), 1)
                st_motion_input = torch.cat((st_motion_input, st_labels), 2)
        
        
                #######################################################
                # (2) Generate fake stories and images
                ######################################################

                if len(self.gpus) > 1:
                    netG = nn.DataParallel(netG)
                st_inputs = (st_motion_input, st_content_input)
                #lr_st_fake, st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                #    nn.parallel.data_parallel(netG.sample_videos, st_inputs, self.gpus)
                lr_st_fake, st_fake, m_mu, m_logvar, c_mu, c_logvar = netG.sample_videos(*st_inputs)
                im_inputs = (im_motion_input, im_content_input)
                #lr_im_fake, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = \
                #nn.parallel.data_parallel(netG.sample_images, im_inputs, self.gpus)
                lr_im_fake, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = netG.sample_images(*im_inputs)
                # print(st_fake.shape, im_fake.shape)
                characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor).cuda()
                st_mu = torch.cat((c_mu, st_motion_input[:,:, :self.cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)
                im_mu = torch.cat((im_motion_input, cim_mu), 1)
                
                ############################
                # (3) Update D network
                ###########################
                im_errD = torch.tensor(0)
                imgD_loss_report = {}
                st_errD = torch.tensor(0)
                stD_loss_report = {}
                
                ############################
                # (2) Update G network
                ###########################
                # TODO: Add config parameter for number of generator steps
                for g_iter in range(2):
                    netG.zero_grad()

                    st_inputs = (st_motion_input, st_content_input)
                    # _, st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                    #    nn.parallel.data_parallel(netG.sample_videos, st_inputs, self.gpus)
                    _, st_fake, m_mu, m_logvar, c_mu, c_logvar = netG.sample_videos(*st_inputs)
                    im_inputs = (im_motion_input, im_content_input)
                    # _, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = \
                    # nn.parallel.data_parallel(netG.sample_images, im_inputs, self.gpus)
                    _, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = netG.sample_images(*im_inputs)

                    characters_mu = (st_labels.mean(1) > 0).type(torch.FloatTensor).cuda()
                    st_mu = torch.cat((c_mu, st_motion_input[:, :, :self.cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)
                    im_mu = torch.cat((im_motion_input, cim_mu), 1)
                    im_errG = torch.tensor(0)
                    imG_loss_report = {}
                    st_errG = torch.tensor(0)
                    stG_loss_report = {}
                    im_kl_loss = KL_loss(cim_mu, cim_logvar)
                    st_kl_loss = KL_loss(c_mu, c_logvar)
                    kl_loss = im_kl_loss + self.ratio * st_kl_loss
                    errG_total = im_kl_loss * self.cfg.TRAIN.COEFF.KL + self.ratio * (st_errG + st_kl_loss * self.cfg.TRAIN.COEFF.KL)

                    if self.cfg.TRAIN.PERCEPTUAL_LOSS:
                        if self.cfg.CUDA:
                            per_loss = self.perceptual_loss_net(im_fake, im_real_cpu.cuda())
                        else:
                            per_loss = self.perceptual_loss_net(im_fake, im_real_cpu)
                        errG_total += per_loss

                    errG_total.backward()
                    optimizerG.step()

                # loss_collector.append([imgD_loss_report, imG_loss_report, stD_loss_report, stG_loss_report])
                count = count + 1

            end_t = time.time()
            print('''[%d/%d][%d/%d] %s Total Time: %.2fsec'''
                  % (epoch, self.max_epoch, i, len(storyloader), cfg.DATASET_NAME, (end_t - start_t)))

            for loss_report in [imgD_loss_report, imG_loss_report, stD_loss_report, stG_loss_report]:
                for key, val in loss_report.items():
                    print(key, val)

            print('--------------------------------------------------------------------------------')

            if epoch % self.snapshot_interval == 0:
                save_test_samples(netG, testloader, self.test_dir, epoch)
                save_model(netG, netD_im, netD_st, optimizerG, im_optimizerD, st_optimizerD, epoch, self.model_dir)

        # np.save(os.path.join(self.model_dir, 'losses.npy'), loss_collector)
        with open(os.path.join(self.model_dir, 'losses.pkl'), 'wb') as f:
            pickle.dump(loss_collector, f)

        save_model(netG, netD_im, netD_st, self.max_epoch, self.model_dir)
    
        
    def sample(self, testloader, generator_weight_path, out_dir, stage=1):

        if stage == 1:
            netG, _, _ = self.load_network_stageI()
        else:
            netG, _, _ = self.load_network_stageII()
        netG.load_state_dict(torch.load(generator_weight_path)['netG_state_dict'])
        save_test_samples(netG, testloader, out_dir, 50, mart=self.use_mart)
        
        