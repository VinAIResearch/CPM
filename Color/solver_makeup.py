import torch
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import os
import time
import datetime

import tools.plot as plot_fig
import net
from ops.histogram_matching import *
from ops.loss_added import GANLoss
from tqdm import tqdm as tqdm

import torchvision.models as models
from utils import to_var, de_norm, get_mask

class Solver_makeupGAN(object):
    def __init__(self, data_loaders, config, dataset_config):
        # dataloader
        self.checkpoint = config.checkpoint
        # Hyper-parameteres
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.ndis = config.ndis
        self.num_epochs = config.num_epochs  # set 200
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.whichG = config.whichG
        self.norm = config.norm

        # Training settings
        self.snapshot_step = config.snapshot_step
        self.log_step = config.log_step
        self.vis_step = config.vis_step

        #training setting
        self.task_name = config.task_name

        # Data loader
        self.data_loader_train = data_loaders[0]
        self.data_loader_test = data_loaders[1]

        # Model hyper-parameters
        self.img_size = config.img_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lips = config.lips
        self.skin = config.skin
        self.eye = config.eye

        # Hyper-parameteres
        self.lambda_idt = config.lambda_idt
        self.lambda_A = config.lambda_A
        self.lambda_B = config.lambda_B
        self.lambda_his_lip = config.lambda_his_lip
        self.lambda_his_skin_1 = config.lambda_his_skin_1
        self.lambda_his_skin_2 = config.lambda_his_skin_2
        self.lambda_his_eye = config.lambda_his_eye
        self.lambda_vgg = config.lambda_vgg

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.cls = config.cls_list
        self.content_layer = config.content_layer
        self.direct = config.direct
        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path + '_' + config.task_name
        self.vis_path = config.vis_path + '_' + config.task_name
        self.snapshot_path = config.snapshot_path + '_' + config.task_name
        self.result_path = config.vis_path + '_' + config.task_name

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        self.build_model()
        # Start with trained model
        if self.checkpoint:
            print('Loaded pretrained model from: ', self.checkpoint)
            self.load_checkpoint()

        writer_path = os.path.join('./', 'runs', config.task_name)
        print('TensorBoard will be saved in: ', writer_path)
        self.writer = SummaryWriter(writer_path)
        if not os.path.isdir(os.path.join('./', 'runs', config.task_name)):
            os.makedirs(os.path.join('./runs', config.task_name))
        #for recording
        self.start_time = time.time()
        self.e = 0
        self.i = 0
        self.loss = {}

#         if not os.path.exists(self.log_path):
#             os.makedirs(self.log_path)
#         if not os.path.exists(self.vis_path):
#             os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for i in self.cls:
            for param_group in getattr(self, "d_" + i + "_optimizer").param_groups:
                param_group['lr'] = d_lr

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def save_models(self):
        torch.save(self.G.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        for i in self.cls:
            torch.save(getattr(self, "D_" + i).state_dict(),
                       os.path.join(self.snapshot_path, '{}_{}_D_'.format(self.e + 1, self.i + 1) + i + '.pth'))

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def load_checkpoint(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_G.pth'.format(self.checkpoint))))
        for i in self.cls:
            getattr(self, "D_" + i).load_state_dict(torch.load(os.path.join(
                self.snapshot_path, '{}_D_'.format(self.checkpoint) + i + '.pth')))
        print('loaded trained models (step: {})..!'.format(self.checkpoint))

    def build_model(self):
        # Define generators and discriminators
        if self.whichG=='normal':
            self.G = net.Generator_makeup(self.g_conv_dim, self.g_repeat_num)
        if self.whichG=='branch':
            self.G = net.Generator_branch(self.g_conv_dim, self.g_repeat_num)
        for i in self.cls:
            setattr(self, "D_" + i, net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm))

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor =torch.cuda.FloatTensor)
        # self.vgg = net.VGG()
        # self.vgg.load_state_dict(torch.load('addings/vgg_conv.pth'))
        self.vgg=models.vgg16(pretrained=True)
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        for i in self.cls:
            setattr(self, "d_" + i + "_optimizer", \
                    torch.optim.Adam(filter(lambda p: p.requires_grad, getattr(self, "D_" + i).parameters()), \
                                     self.d_lr, [self.beta1, self.beta2]))

        # Weights initialization
        self.G.apply(self.weights_init_xavier)
        for i in self.cls:
            getattr(self, "D_" + i).apply(self.weights_init_xavier)

        # Print networks
#         self.print_network(self.G, 'G')
#         for i in self.cls:
#             self.print_network(getattr(self, "D_" + i), "D_" + i)

        if torch.cuda.is_available():
            self.G.cuda()
            self.vgg.cuda()
            for i in self.cls:
                getattr(self, "D_" + i).cuda()

    def vgg_forward(self, model, x):
        for i in range(18):
            x=model.features[i](x)
        return x

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
        mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
        mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A = self.to_var(mask_A, requires_grad=False)
        mask_B = self.to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2

    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        # dstImg = (input_masked.data).cpu().clone()
        # refImg = (target_masked.data).cpu().clone()
        input_match = histogram_matching(input_masked, target_masked, index)
        input_match = self.to_var(input_match, requires_grad=False)
        loss = self.criterionL1(input_masked, input_match)
        return loss

    def train(self):
        """Train StarGAN within a single dataset."""
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)
        # Start with trained model if exists
        cls_A = self.cls[0]
        cls_B = self.cls[1]
        g_lr = self.g_lr
        d_lr = self.d_lr
        if self.checkpoint:
            start = int(self.checkpoint.split('_')[0])
#             self.vis_test()
        else:
            start = 0
        # Start training
        self.start_time = time.time()
        for self.e in tqdm(range(start, self.num_epochs)):
            
            for self.i, (img_A, img_B, mask_A, mask_B) in enumerate(tqdm(self.data_loader_train)):
                # Convert tensor to variable
                # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose 
                # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
                if self.checkpoint or self.direct:
                    if self.lips==True:
                        mask_A_lip = (mask_A==7).float() + (mask_A==9).float()
                        mask_B_lip = (mask_B==7).float() + (mask_B==9).float()
                        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
                    if self.skin==True:
                        mask_A_skin = (mask_A==1).float() + (mask_A==6).float() + (mask_A==13).float()
                        mask_B_skin = (mask_B==1).float() + (mask_B==6).float() + (mask_B==13).float()
                        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
                    if self.eye==True:
                        mask_A_eye_left = (mask_A==4).float()
                        mask_A_eye_right = (mask_A==5).float()
                        mask_B_eye_left = (mask_B==4).float()
                        mask_B_eye_right = (mask_B==5).float()
                        mask_A_face = (mask_A==1).float() + (mask_A==6).float()
                        mask_B_face = (mask_B==1).float() + (mask_B==6).float()
                        # avoid the situation that images with eye closed
                        if not ((mask_A_eye_left>0).any() and (mask_B_eye_left>0).any() and \
                            (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any()):
                            continue
                        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
                        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
                        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
                            self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
                        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
                            self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)

                org_A = self.to_var(img_A, requires_grad=False)
                ref_B = self.to_var(img_B, requires_grad=False)
                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B
                # Real
                out = getattr(self, "D_" + cls_A)(ref_B)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                fake_A, fake_B = self.G(org_A, ref_B)
                fake_A = Variable(fake_A.data).detach()
                fake_B = Variable(fake_B.data).detach()
                out = getattr(self, "D_" + cls_A)(fake_A)
                #d_loss_fake = self.get_D_loss(out, "fake")
                d_loss_fake =  self.criterionGAN(out, False)
               
                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                getattr(self, "d_" + cls_A + "_optimizer").zero_grad()
                d_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_A + "_optimizer").step()

                # Logging
                self.loss = {}
                # self.loss['D-A-loss_real'] = d_loss_real.item()

                # training D_B, D_B aims to distinguish class A
                # Real
                out = getattr(self, "D_" + cls_B)(org_A)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                out = getattr(self, "D_" + cls_B)(fake_B)
                #d_loss_fake = self.get_D_loss(out, "fake")
                d_loss_fake =  self.criterionGAN(out, False)
               
                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                getattr(self, "d_" + cls_B + "_optimizer").zero_grad()
                d_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_B + "_optimizer").step()

                # Logging
                # self.loss['D-B-loss_real'] = d_loss_real.item()

                # ================== Train G ================== #
                if (self.i + 1) % self.ndis == 0:
                    # adversarial loss, i.e. L_trans,v in the paper 

                    # identity loss
                    if self.lambda_idt > 0:
                        # G should be identity if ref_B or org_A is fed
                        idt_A1, idt_A2 = self.G(org_A, org_A)
                        idt_B1, idt_B2 = self.G(ref_B, ref_B)
                        loss_idt_A1 = self.criterionL1(idt_A1, org_A) * self.lambda_A * self.lambda_idt
                        loss_idt_A2 = self.criterionL1(idt_A2, org_A) * self.lambda_A * self.lambda_idt
                        loss_idt_B1 = self.criterionL1(idt_B1, ref_B) * self.lambda_B * self.lambda_idt
                        loss_idt_B2 = self.criterionL1(idt_B2, ref_B) * self.lambda_B * self.lambda_idt
                        # loss_idt
                        loss_idt = (loss_idt_A1 + loss_idt_A2 + loss_idt_B1 + loss_idt_B2) * 0.5
                    else:
                        loss_idt = 0
                        
                    # GAN loss D_A(G_A(A))
                    # fake_A in class B, 
                    fake_A, fake_B = self.G(org_A, ref_B)
                    pred_fake = getattr(self, "D_" + cls_A)(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)
                    #g_loss_adv = self.get_G_loss(out)
                    # GAN loss D_B(G_B(B))
                    pred_fake = getattr(self, "D_" + cls_B)(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)
                    rec_B, rec_A = self.G(fake_B, fake_A)

                    # color_histogram loss
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    if self.checkpoint or self.direct:
                        if self.lips==True:
                            g_A_lip_loss_his = self.criterionHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_his = self.criterionHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_his += g_A_lip_loss_his
                            g_B_loss_his += g_B_lip_loss_his
                        if self.skin==True:
                            g_A_skin_loss_his = self.criterionHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_his = self.criterionHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_his += g_A_skin_loss_his
                            g_B_loss_his += g_B_skin_loss_his
                        if self.eye==True:
                            g_A_eye_left_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_his = self.criterionHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_his = self.criterionHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_his += g_A_eye_left_loss_his + g_A_eye_right_loss_his
                            g_B_loss_his += g_B_eye_left_loss_his + g_B_eye_right_loss_his

	                # cycle loss
                    g_loss_rec_A = self.criterionL1(rec_A, org_A) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, ref_B) * self.lambda_B

                    # vgg loss
                    # vgg_org = self.vgg(org_A, self.content_layer)[0]
                    # vgg_org = Variable(vgg_org.data).detach()
                    # vgg_fake_A = self.vgg(fake_A, self.content_layer)[0]
                    # g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_org) * self.lambda_A * self.lambda_vgg
                    vgg_org=self.vgg_forward(self.vgg,org_A)
                    vgg_org = Variable(vgg_org.data).detach()
                    vgg_fake_A=self.vgg_forward(self.vgg,fake_A)
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_org) * self.lambda_A * self.lambda_vgg

                    # vgg_ref = self.vgg(ref_B, self.content_layer)[0]
                    # vgg_ref = Variable(vgg_ref.data).detach()
                    # vgg_fake_B = self.vgg(fake_B, self.content_layer)[0]
                    # g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_ref) * self.lambda_B * self.lambda_vgg
                    vgg_ref=self.vgg_forward(self.vgg, ref_B)
                    vgg_ref = Variable(vgg_ref.data).detach()
                    vgg_fake_B=self.vgg_forward(self.vgg,fake_B)
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_ref) * self.lambda_B * self.lambda_vgg

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5
					
                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt
                    if self.checkpoint or self.direct:
                        g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his
                    
                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()

                    # # Logging
                    # self.loss['G-A-loss-adv'] = g_A_loss_adv.item()
                    # self.loss['G-B-loss-adv'] = g_A_loss_adv.item()
                    # self.loss['G-loss-org'] = g_loss_rec_A.item()
                    # self.loss['G-loss-ref'] = g_loss_rec_B.item()
                    # self.loss['G-loss-idt'] = loss_idt.item()
                    # self.loss['G-loss-img-rec'] = (g_loss_rec_A + g_loss_rec_B).item()
                    # self.loss['G-loss-vgg-rec'] = (g_loss_A_vgg + g_loss_B_vgg).item()
                    # if self.direct:
                    #     self.loss['G-A-loss-his'] = g_A_loss_his.item()
                    #     self.loss['G-B-loss-his'] = g_B_loss_his.item()

                # Print out log info


                #plot the figures
                # for key_now in self.loss.keys():
                #     plot_fig.plot(key_now, self.loss[key_now])

                #save the images
                if (self.i + 1) % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train([org_A, ref_B, fake_A, fake_B, rec_A, rec_B])
                if self.i%10==0:
                    
                    self.writer.add_scalar('losses/GA-loss-adv', g_A_loss_adv.item(), self.i)
                    self.writer.add_scalar('losses/GB-loss-adv', g_B_loss_adv.item(), self.i)
                    self.writer.add_scalar('losses/rec-org', g_loss_rec_A.item(), self.i)
                    self.writer.add_scalar('losses/rec-ref', g_loss_rec_B.item(), self.i)
                    self.writer.add_scalar('losses/vgg-A', g_loss_A_vgg.item(), self.i)
                    self.writer.add_scalar('losses/vgg-B', g_loss_B_vgg.item(), self.i)
                    if self.eye:
                        self.writer.add_scalar('mkup-hist/eyes', (g_A_eye_left_loss_his + g_A_eye_right_loss_his).item(), self.i)
                    if self.lips:
                        self.writer.add_scalar('mkup-hist/lips', (g_A_lip_loss_his+g_B_lip_loss_his).item(), self.i)
                    if self.skin:
                        self.writer.add_scalar('mkup-hist/skin', (g_A_skin_loss_his+g_B_skin_loss_his).item(), self.i)
                    #-- Images
                    self.writer.add_images('Original/org_A', de_norm(org_A), self.i)
                    self.writer.add_images('Original/ref_B', de_norm(ref_B), self.i)
                    self.writer.add_images('Fake/fake_A', de_norm(fake_A), self.i)
                    self.writer.add_images('Fake/fake_B', de_norm(fake_B), self.i)
                    self.writer.add_images('Rec/rec_A', de_norm(rec_A), self.i)
                    self.writer.add_images('Rec/rec_B', de_norm(rec_B), self.i)
                
                # Save model checkpoints
                if (self.i + 1) % self.snapshot_step == 0:
                    self.save_models()

                # plot_fig.tick()
            os.remove(os.path.join('./runs/txt', os.listdir('./runs/txt/')[0]))
            # Decay learning rate
            if (self.e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))
            # if self.e % 2 == 0:
            #     print("Saving output...")
            #     self.vis_test()
    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = os.path.join(self.result_path, mode)
        if not os.path.exists(result_path_train):
            os.mkdir(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    def vis_test(self):
        # saving test results
        mode = "test_vis"
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            real_org = self.to_var(img_A)
            real_ref = self.to_var(img_B)

            image_list = []
            image_list.append(real_org)
            image_list.append(real_ref)

            # Get makeup result
            fake_A, fake_B = self.G(real_org, real_ref)
            rec_B, rec_A = self.G(fake_B, fake_A)

            image_list.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            vis_train_path = os.path.join(self.result_path, mode)
            result_path_now = os.path.join(vis_train_path, "epoch" + str(self.e))
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path = os.path.join(result_path_now, '{}_{}_{}_fake.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list.data), save_path, normalize=True)
            #print('Translated test images and saved into "{}"..!'.format(save_path))

    def test(self):
        # Load trained parameters
        G_path = os.path.join(self.snapshot_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        #time_total = time.time()
        time_total = 0
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            #start = time.time()
            start = time.time()
            real_org = self.to_var(img_A)
            real_ref = self.to_var(img_B)

            image_list = []
            image_list_0 = []
            image_list.append(real_org)
            image_list.append(real_ref)

            # Get makeup result
            fake_A, fake_B = self.G(real_org, real_ref)
            rec_B, rec_A = self.G(fake_B, fake_A)
            time_total += time.time() - start
            image_list.append(fake_A)
            image_list_0.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            image_list_0 = torch.cat(image_list_0, dim=3)

            result_path_now = os.path.join(self.result_path, "multi")
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path = os.path.join(result_path_now, '{}_{}_{}_fake.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list.data), save_path, nrow=1, padding=0, normalize=True)
            result_path_now = os.path.join(self.result_path, "single")
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path_0 = os.path.join(result_path_now, '{}_{}_{}_fake_single.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list_0.data), save_path_0, nrow=1, padding=0, normalize=True)
            print('Translated test images and saved into "{}"..!'.format(save_path))
        print("average time : {}".format(time_total/len(self.data_loader_test)))
