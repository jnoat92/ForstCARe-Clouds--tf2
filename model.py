# from __future__ import division
import os
import time
# import glob
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.keras.models import save_model, load_model
import numpy as np
# from six.moves import xrange
# from sklearn import preprocessing as pre
import joblib
# import scipy.io as io
import matplotlib.pyplot as plt
# from skimage.util.shape import view_as_windows
import scipy.io as sio
from tqdm import trange
# import json

# from ops import *
from utils import *

import network
import sys


class cGAN(object):

    def __init__(self, sess, args, image_size_tr=256, image_size = 256, load_size=286,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=11, output_c_dim=7, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):

        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size_tr = image_size_tr
        self.image_size = image_size
        self.sample_size = sample_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        
        self.args = args
        self.sampling_type = args.sampling_type
        self.norm_type = args.norm_type
        self.rate_dropout = 0.5

        self.checkpoint_dir = checkpoint_dir
        self.visible_bands = [2, 1, 0]
        self.build_model()

        self.data_path = args.datasets_dir + args.dataset_name

        if args.dataset_name == 'Para_10m':
            self.lims = np.array([0, 17730, 0, 9200])
            self.sar_path = self.data_path + '/Sentinel1_'
            self.opt_path = self.data_path + '/Sentinel2_'
            self.opt_cloudy_path = self.data_path + '/Sentinel2_Clouds_'
            self.labels_path = self.data_path + '/Reference'

            self.sar_name_t0 = ['2018/COPERNICUS_S1_20180719_20180726_VV', 
                                '2018/COPERNICUS_S1_20180719_20180726_VH']
            self.opt_name_t0 = ['2018/COPERNICUS_S2_20180721_20180726_B1_B2_B3',
                                '2018/COPERNICUS_S2_20180721_20180726_B4_B5_B6',
                                '2018/COPERNICUS_S2_20180721_20180726_B7_B8_B8A',
                                '2018/COPERNICUS_S2_20180721_20180726_B9_B10_B11',
                                '2018/COPERNICUS_S2_20180721_20180726_B12']
            self.opt_cloudy_name_t0 = ['2018/COPERNICUS_S2_20180611_B1_B2_B3',
                                       '2018/COPERNICUS_S2_20180611_B4_B5_B6',
                                       '2018/COPERNICUS_S2_20180611_B7_B8_B8A',
                                       '2018/COPERNICUS_S2_20180611_B9_B10_B11',
                                       '2018/COPERNICUS_S2_20180611_B12']
            self.opt_cloudmask_name_t0 = '2018/cloudmask_s2_2018'
            self.opt_cloudy_cloudmask_name_t0 = '2018/cloudmask_s2_cloudy_2018'

            self.sar_name_t1 = ['2019/COPERNICUS_S1_20190721_20190726_VV', 
                                '2019/COPERNICUS_S1_20190721_20190726_VH']
            self.opt_name_t1 = ['2019/COPERNICUS_S2_20190721_20190726_B1_B2_B3',
                                '2019/COPERNICUS_S2_20190721_20190726_B4_B5_B6',
                                '2019/COPERNICUS_S2_20190721_20190726_B7_B8_B8A',
                                '2019/COPERNICUS_S2_20190721_20190726_B9_B10_B11',
                                '2019/COPERNICUS_S2_20190721_20190726_B12']
            self.opt_cloudy_name_t1 = ['2019/COPERNICUS_S2_20190706_B1_B2_B3',
                                       '2019/COPERNICUS_S2_20190706_B4_B5_B6',
                                       '2019/COPERNICUS_S2_20190706_B7_B8_B8A',
                                       '2019/COPERNICUS_S2_20190706_B9_B10_B11',
                                       '2019/COPERNICUS_S2_20190706_B12']
            self.opt_cloudmask_name_t1 = '2019/cloudmask_s2_2019'
            self.opt_cloudy_cloudmask_name_t1 = '2019/cloudmask_s2_cloudy_2019'

            self.labels_name = '/mask_label_17730x9203'

            self.mask_tr_vl_ts_name = '/tile_mask_0tr_1vl_2ts'

        elif args.dataset_name == 'MG_10m':
            self.lims = np.array([0, 20795-4000, 0+3000, 13420])
            self.sar_path = self.data_path + '/S1/'
            self.opt_path = self.data_path + '/S2/'
            self.opt_cloudy_path = self.data_path + '/S2_cloudy/'
            self.labels_path = self.data_path

            self.sar_name_t0 = ['2019/S1_R1_MT_2019_08_02_2019_08_09_VV', 
                                '2019/S1_R1_MT_2019_08_02_2019_08_09_VH']
            self.opt_name_t0 = ['2019/S2_R1_MT_2019_08_02_2019_08_05_B1_B2',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B3_B4',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B5_B6',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B7_B8',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B8A_B9',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B10_B11',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B12']
            self.opt_cloudy_name_t0 = ['2019/S2CL_R1_MT_2019_09_26_2019_09_29_B1_B2',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B3_B4',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B5_B6',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B7_B8',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B8A_B9',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B10_B11',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B12']
            self.opt_cloudmask_name_t0 = '2019/cloudmask_s2_2019_MG'
            self.opt_cloudy_cloudmask_name_t0 = '2019/cloudmask_s2_cloudy_2019_MG'

            self.sar_name_t1 = ['2020/S1_R1_MT_2020_08_03_2020_08_08_VV', 
                                '2020/S1_R1_MT_2020_08_03_2020_08_08_VH']
            self.opt_name_t1 = ['2020/S2_R1_MT_2020_08_03_2020_08_15_B1_B2',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B3_B4',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B5_B6',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B7_B8',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B8A_B9',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B10_B11',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B12']
            self.opt_cloudy_name_t1 = ['2020/S2CL_R1_MT_2020_09_15_2020_09_18_B1_B2',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B3_B4',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B5_B6',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B7_B8',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B8A_B9',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B10_B11',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B12']
            self.opt_cloudmask_name_t1 = '2020/cloudmask_s2_2020_MG'
            self.opt_cloudy_cloudmask_name_t1 = '2020/cloudmask_s2_cloudy_2020_MG'

            self.labels_name = '/ref_2019_2020_20798x13420'

            self.mask_tr_vl_ts_name = '/MT_tr_0_val_1_ts_2_16795x10420_new'

    def build_model(self):

        # ============== PLACEHOLDERS ===============
        self.SAR = tf1.placeholder(tf.float32,
                                    [None, None, None, self.input_c_dim],
                                    name='sar')
        self.OPT = tf1.placeholder(tf.float32,
                                    [None, None, None, self.output_c_dim],
                                    name='opt')
        self.OPT_cloudy = tf1.placeholder(tf.float32,
                                    [None, None, None, self.output_c_dim],
                                    name='opt_cloudy')
        self.learning_rate = tf1.placeholder(tf.float32, [], name="learning_rate")

        self.GAN_condition = tf.concat([self.SAR, self.OPT_cloudy], 3)
    
        # =============== NETWORKS =================
        # Generator
        generator_func = getattr(network, self.args.generator)
        self.generator = generator_func(self, self.GAN_condition.get_shape()[1:], name="Generator")
        self.OPT_fake = self.generator(self.GAN_condition)

        # Discriminator
        self.OPT_pair = tf.concat([self.GAN_condition, self.OPT], 3)
        self.OPT_pair_fake = tf.concat([self.GAN_condition, self.OPT_fake], 3)
        discriminator_func = getattr(network, self.args.discriminator + '_discriminator')
        self.discriminator = discriminator_func(self, self.OPT_pair.get_shape()[1:], name="Discriminator")
        
        self.D , self.D_logits  = self.discriminator(self.OPT_pair)
        self.D_, self.D_logits_ = self.discriminator(self.OPT_pair_fake)

        # Loss Function
        self.d_loss_real = self.cross_entropy_loss(labels=tf.ones_like (self.D) , logits=self.D_logits)
        self.d_loss_fake = self.cross_entropy_loss(labels=tf.zeros_like(self.D_), logits=self.D_logits_)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss      = self.cross_entropy_loss(labels=tf.ones_like (self.D_), logits=self.D_logits_) \
                         + self.L1_lambda * self.l1_loss(self.OPT, self.OPT_fake)
        
        # =============== OPTIMIZERS =================
        t_vars = tf1.trainable_variables()
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]
        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        
        self.g_optim = tf1.train.AdamOptimizer(self.learning_rate, beta1=self.args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        lr = self.learning_rate/10 if self.args.discriminator == "atrous" else self.learning_rate
        # lr = self.learning_rate
        self.d_optim = tf1.train.AdamOptimizer(lr, beta1=self.args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)

        # # ========== This updates moving_mean and moving_variance 
        # # ========== in batch normalization layers when training
        # update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)
        # self.g_ops = [ops for ops in update_ops if 'generator' in ops.name]
        # self.d_ops = [ops for ops in update_ops if 'discriminator' in ops.name]
        # self.g_optim = tf.group([self.g_optim, self.g_ops])
        # self.d_optim = tf.group([self.d_optim, self.d_ops])

        self.model = "%s_bs%s_%s_ps%s" % \
                      (self.args.discriminator, self.batch_size, self.norm_type, self.image_size_tr)

        self.saver = tf1.train.Saver(max_to_keep=3)

        print('_____Generator_____')
        self.count_params(self.g_vars)
        print('_____Discriminator_____')
        self.count_params(self.d_vars)
        print('_____Full Model_____')
        self.count_params(t_vars)


    def cross_entropy_loss(self, labels, logits):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        return loss
    def lsgan_loss(self, labels, logits):
            loss = tf.reduce_mean(tf.squared_difference(logits, labels))
            return loss 
    def l1_loss(self, a, b):
        loss = tf.reduce_mean(tf.abs(a - b))
        return loss

    def save(self, checkpoint_dir, step):
        model_name = "cGAN.model"

        checkpoint_dir = os.path.join(checkpoint_dir, self.model, self.args.dataset_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_model(self.generator, os.path.join(checkpoint_dir, "Generator.h5"))
        # save_model(self.discriminator, os.path.join(checkpoint_dir, "Discriminator.h5"))

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        print("Saving checkpoint!")
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return int(aux)
        else:
            return int(0)

    def count_params(self, t_vars):
        """
        print number of trainable variables
        """
        n = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        print("Model size: %dK params" %(n/1000))

        # w = self.sess.run(self.g_vars)
        # for val, var in zip(w, self.g_vars):
        #     if 'generator' in var.name:
        #         print(var.name)
        #         print(val.shape)
        # #         # break
        # sys.exit()


    def train(self, args):
        """Train cGAN"""

        # Model
        model_dir = os.path.join(self.checkpoint_dir, self.model, args.dataset_name)
        sample_dir = os.path.join(model_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        #================== CREATE DATASET ==================
        if args.date == 'both':
            train_patches, val_patches, test_patches, self.data_dic, \
                self.sar_norm, self.opt_norm = create_dataset_both_images(self)
        elif args.date == 'd0':
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            train_patches, val_patches, test_patches, self.data_dic, \
                self.sar_norm, self.opt_norm = create_dataset_coordinates(self, prefix = 0)
        elif args.date == 'd1':
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            train_patches, val_patches, test_patches, self.data_dic, \
                self.sar_norm, self.opt_norm = create_dataset_coordinates(self, prefix = 1)
        
        # print("mask_shape:", Split_Image(self, random_tiles="fixed").shape)
        # print(self.data_dic['sar_t0'].shape, self.data_dic['sar_t0'].min(), self.data_dic['sar_t0'].max())
        # print(self.data_dic['opt_t0'].shape, self.data_dic['opt_t0'].min(), self.data_dic['opt_t0'].max())
        # print(self.data_dic['opt_cloudy_t0'].shape, self.data_dic['opt_cloudy_t0'].min(), self.data_dic['opt_cloudy_t0'].max())
        # print(self.data_dic['sar_t1'].shape, self.data_dic['sar_t1'].min(), self.data_dic['sar_t1'].max())
        # print(self.data_dic['opt_t1'].shape, self.data_dic['opt_t1'].min(), self.data_dic['opt_t1'].max())
        # print(self.data_dic['opt_cloudy_t1'].shape, self.data_dic['opt_cloudy_t1'].min(), self.data_dic['opt_cloudy_t1'].max())
        # plot_hist(self.data_dic['sar_t0'], 2**16-1, None, "sar_t0_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_t0'], 2**16-1, None, "opt_t0_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t0'], 2**16-1, None, "opt_cloudy_t0_nonnorm", sample_dir)
        # plot_hist(self.data_dic['sar_t1'], 2**16-1, None, "sar_t1_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_t1'], 2**16-1, None, "opt_t1_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t1'], 2**16-1, None, "opt_cloudy_t1_nonnorm", sample_dir)

        # Normalize
        if args.date == 'both' or args.date == 'd0':
            self.data_dic["sar_t0"] = self.sar_norm.Normalize(self.data_dic["sar_t0"])
            self.data_dic["opt_t0"] = self.opt_norm.Normalize(self.data_dic["opt_t0"])
            self.data_dic["opt_cloudy_t0"] = self.opt_norm.Normalize(self.data_dic["opt_cloudy_t0"])
        if args.date == 'both' or args.date == 'd1':
            self.data_dic["sar_t1"] = self.sar_norm.Normalize(self.data_dic["sar_t1"])
            self.data_dic["opt_t1"] = self.opt_norm.Normalize(self.data_dic["opt_t1"])
            self.data_dic["opt_cloudy_t1"] = self.opt_norm.Normalize(self.data_dic["opt_cloudy_t1"])

        # plot_hist(self.data_dic['sar_t0'], 2**16-1, None, "sar_t0", sample_dir)
        # plot_hist(self.data_dic['opt_t0'], 2**16-1, None, "opt_t0", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t0'], 2**16-1, None, "opt_cloudy_t0", sample_dir)
        # plot_hist(self.data_dic['sar_t1'], 2**16-1, None, "sar_t1", sample_dir)
        # plot_hist(self.data_dic['opt_t1'], 2**16-1, None, "opt_t1", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t1'], 2**16-1, None, "opt_cloudy_t1", sample_dir)

        # save normalizers
        joblib.dump(self.sar_norm, self.args.datasets_dir + self.args.dataset_name  + '/' + 'sar_norm.pkl')
        joblib.dump(self.opt_norm, self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')
        with open(sample_dir + '/' + 'normalization_values.txt', 'w') as f:
            f.write("SAR min-max values\n")
            q = self.sar_norm.__dict__
            for i in q.keys():
                f.write("{}: {}\n".format(i, str(q[i])))
            
            f.write("\n\n\n")
            f.write("OPT min-max values\n")
            q = self.opt_norm.__dict__
            for i in q.keys():
                f.write("{}: {}\n".format(i, str(q[i])))


        # Initialize graph
        init_op = tf1.global_variables_initializer()
        self.sess.run(init_op)
        counter = self.load(model_dir)
        if counter:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        idx = 1500
        generate_samples(self, output_path=sample_dir, idx=idx, 
                         patch_list = val_patches, epoch=counter, real_flag = True)
        
        # Train
        loss_trace_G, loss_trace_D = [], []
        start_time = time.time()
        for e in range(counter+1, args.epoch+1):

            # Learning rate
            p = max(0.0, np.floor((e - (self.args.init_e - self.args.epoch_drop)) / self.args.epoch_drop)) 
            lr = self.args.lr * (self.args.lr_decay ** p)

            errG, errD = self.Routine_batches(train_patches, lr, e, start_time)
                
            loss_trace_G.append(errG)
            loss_trace_D.append(errD)
            np.save(model_dir + '/loss_trace_G', loss_trace_G)
            np.save(model_dir + '/loss_trace_D', loss_trace_D)

            # save sample
            generate_samples(self, output_path=sample_dir, idx=idx, 
                            patch_list = val_patches, epoch=e)

            self.save(args.checkpoint_dir, e)

    def Routine_batches(self, patch_list, lr, e, start_time):

        np.random.shuffle(patch_list)

        errD, errG = 0, 0

        batches = trange(len(patch_list) // self.batch_size)
        for batch in batches:

            # Taking the Batch
            s1, s2, s2_cloudy = [], [], []
            for im in range(batch*self.batch_size, (batch+1)*self.batch_size):
                batch_image = Take_patches(patch_list, idx=im,
                                           data_dic=self.data_dic,
                                           fine_size=self.image_size_tr,
                                           random_crop_transformation=True)
                s1.append(batch_image[0])
                s2.append(batch_image[1])
                s2_cloudy.append(batch_image[2])

            s1 = np.asarray(s1)
            s2 = np.asarray(s2)
            s2_cloudy = np.asarray(s2_cloudy)

            ###
                # sar = self.sar_norm.Denormalize(s1[0,:,:,:])
                # opt = self.opt_norm.Denormalize(s2[0,:,:,:])            # Save Sentinel 1
                # opt_cloudy = self.opt_norm.Denormalize(s2_cloudy[0,:,:,:])            # Save Sentinel 1

                # k = batch
                # image = (sar - self.sar_norm.min_val) / (self.sar_norm.max_val - self.sar_norm.min_val)
                # file_name = "s1_" + str(k)
                # save_image(image, file_name, sensor = "s1")

                # # Save Sentinel 2
                # image = opt[:, :, self.visible_bands] / self.opt_norm.max_val.max()
                # file_name = "s2_" + str(k)
                # save_image(image, file_name, sensor = "s2")

                # image = opt_cloudy[:, :, self.visible_bands] / self.opt_norm.max_val.max()
                # file_name = "s2_cludy" + str(k)
                # save_image(image, file_name, sensor = "s2")
                # exit(0)

            # Update D network
            _ = self.sess.run([self.d_optim],
                                        feed_dict={self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy, self.learning_rate: lr})

            # Update G network
            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            for _ in range(2):
                _ = self.sess.run([self.g_optim],
                                feed_dict={self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy, self.learning_rate: lr})
            
            if np.mod(batch + 1, 1000) == 0:
                errD = self.d_loss.eval({ self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy })
                errG = self.g_loss.eval({ self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy })
                print("Epoch: [%2d] [%4d/%4d] lr: %.6f time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (e, (batch+1)*self.batch_size, len(patch_list), lr,
                        time.time() - start_time, errD, errG))

        return errG, errD


    def Translate_complete_image(self, args, date):


        print( 'Generating Image for ' + args.dataset_name + ' dataset')

        output_path = os.path.join(args.test_dir, self.model, args.dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        model_dir = os.path.join(self.checkpoint_dir, self.model, args.dataset_name)
        self.generator = load_model(os.path.join(model_dir, "Generator.h5"))

        # Loading normalizers used during training
        self.sar_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'sar_norm.pkl')
        self.opt_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')

        if date == "t0":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t0
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            prefix = 0
        elif date == "t1":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t1
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            prefix = 1

        # Loading masks
        mask_tr_vl_ts = Split_Image(self, random_tiles='fixed')
        opt_cloudy_cloudmask = np.load(self.opt_cloudy_path + opt_cloudy_cloudmask_name + '.npy')
        opt_cloudy_cloudmask = opt_cloudy_cloudmask[:mask_tr_vl_ts.shape[0], :mask_tr_vl_ts.shape[1]]
        test_mask, mask_cloud_free, \
            mask_cloud, mask_shadow = [np.zeros_like(mask_tr_vl_ts) for i in range(4)]
                
        # mask_shadow    [opt_cloudy_cloudmask==-1] = 1
        mask_cloud     [opt_cloudy_cloudmask==1 ] = 1
        # mask_cloud_shadow = mask_cloud + mask_shadow
        mask_cloud_free = 1 - mask_cloud

        test_mask[mask_tr_vl_ts==2] = 1
        mask_cloud *= test_mask
        mask_cloud_free *= test_mask

        img = Image.fromarray(np.uint8((mask_cloud)*255))
        img.save(output_path + '/test_mask_cloud_' + date + '.tiff')

        img = Image.fromarray(np.uint8((mask_cloud_free)*255))
        img.save(output_path + '/test_mask_cloud_free_' + date + '.tiff')


        # ====================== LOAD DATA =====================
        # Loading images
        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False,
                                                                   flag_image = [1, 0, 1], cut=False)        
        sar = self.sar_norm.Normalize(self.data_dic["sar_" + date])
        opt_cloudy = self.opt_norm.Normalize(self.data_dic["opt_cloudy_" + date])
        del self.data_dic

        start_time = time.time()
        print("Start Inference {}".format(date))
        opt_fake = Image_reconstruction([self.SAR, self.OPT_cloudy], self.generator, 
                                        self.output_c_dim, patch_size=3840, # 4096, 3840
                                        overlap_percent=0.02).Inference(np.concatenate((sar, opt_cloudy), axis=2))
        print("Inference complete --> {} segs".format(time.time()-start_time))
        del sar
        
        opt_cloudy = self.opt_norm.Denormalize(opt_cloudy)
        print("Saving opt_cloudy image")
        GeoReference_Raster_from_Source_data(self.opt_path + self.opt_name[prefix] + '.tif', 
                                             opt_cloudy.transpose(2, 0, 1),
                                             output_path + '/S2_cloudy_' + date + '_10bands.tif')
        del opt_cloudy

        opt_fake = self.opt_norm.Denormalize(opt_fake)
        print("Saving opt_fake image")
        GeoReference_Raster_from_Source_data(self.opt_path + self.opt_name[prefix] + '.tif', 
                                             opt_fake.transpose(2, 0, 1),
                                             output_path + '/S2_' + date + '_10bands' + '_Fake_.tif')
        np.save(output_path + '/S2_' + date + '_10bands' + '_Fake_', opt_fake)


        # Loading cloud-free image
        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False,
                                                                   flag_image = [0, 1, 0], cut=False)
        opt = self.opt_norm.clip_image(self.data_dic["opt_" + date])
        del self.data_dic
        print("Saving opt_cloudy image")
        GeoReference_Raster_from_Source_data(self.opt_path + self.opt_name[prefix] + '.tif', 
                                             opt.transpose(2, 0, 1),
                                             output_path + '/S2_' + date + '_10bands.tif')

        ########### METRICS ##################
        opt =           opt[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]
        opt_fake = opt_fake[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]

        with open(output_path + '/' + 'Similarity_Metrics.txt', 'a') as f:
            # test area (cloudy)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloudy)", mae, mse, rmse, psnr, sam, ssim)
            # test area (cloud-free)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud_free)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloud-free)", mae, mse, rmse, psnr, sam, ssim)
            # test area
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, test_mask, ssim_flag=True, dataset=args.dataset_name)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area", mae, mse, rmse, psnr, sam, ssim)

        del opt, opt_fake

    def Meraner_metrics(self, args, date):

        path = args.test_dir + '/Meraner_approach/' + args.dataset_name + '/'
        
        # Loading normalizers used during training
        self.opt_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')
        
        if date == "t0":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t0
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            prefix = 0
            
            # Pará
            # file_ = 'predictions_pretrained_2018.tif'
            # output_file = 'predictions_pretrained.txt'

            # file_ = 'predictions_scratch_2018.tif'
            # output_file = 'predictions_scratch.txt'

            # file_ = 'predictions_remove60m_2018.tif'
            # output_file = 'predictions_remove60m.txt'

            # file_ = 'predictions_scratch_2018_60epoch.tif'
            # output_file = 'predictions_scratch_60epoch.txt'

            # Mato Grosso
            file_ = 'predictions_scratch_MG_2019.tif'
            output_file = 'predictions_scratch_MG.txt'

        elif date == "t1":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t1
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            prefix = 1
            
            # Pará
            # file_ = 'predictions_pretrained_2019.tif'
            # output_file = 'predictions_pretrained.txt'

            # file_ = 'predictions_scratch_2019.tif'
            # output_file = 'predictions_scratch.txt'

            # file_ = 'predictions_remove60m_2019.tif'
            # output_file = 'predictions_remove60m.txt'

            # file_ = 'predictions_scratch_2019_60epoch.tif'
            # output_file = 'predictions_scratch_60epoch.txt'

            # Mato Grosso
            file_ = 'predictions_scratch_MG_2020.tif'
            output_file = 'predictions_scratch_MG.txt'

        # Loading masks
        mask_tr_vl_ts = Split_Image(self, random_tiles='fixed')
        opt_cloudy_cloudmask = np.load(self.opt_cloudy_path + opt_cloudy_cloudmask_name + '.npy')
        opt_cloudy_cloudmask = opt_cloudy_cloudmask[:mask_tr_vl_ts.shape[0], :mask_tr_vl_ts.shape[1]]
        test_mask, mask_cloud_free, \
            mask_cloud, mask_shadow = [np.zeros_like(mask_tr_vl_ts) for i in range(4)]
                
        # mask_shadow    [opt_cloudy_cloudmask==-1] = 1
        mask_cloud     [opt_cloudy_cloudmask==1 ] = 1
        # mask_cloud_shadow = mask_cloud + mask_shadow
        mask_cloud_free = 1 - mask_cloud

        test_mask[mask_tr_vl_ts==2] = 1
        mask_cloud *= test_mask
        mask_cloud_free *= test_mask

        # img = Image.fromarray(np.uint8((mask_cloud)*255))
        # img.save(path + '/test_mask_cloud_' + date + '.tiff')

        # img = Image.fromarray(np.uint8((mask_cloud_free)*255))
        # img.save(path + '/test_mask_cloud_free_' + date + '.tiff')

        # ====================== LOAD DATA =====================
        opt_fake = load_tiff_image(path + file_).astype('float32')
        if opt_fake.shape[0] == 13:
            opt_fake = opt_fake[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt_fake = opt_fake.transpose([1, 2, 0])
        opt_fake[np.isnan(opt_fake)] = np.nanmean(opt_fake)
        opt_fake = self.opt_norm.clip_image(opt_fake)

        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False,
                                                                   flag_image = [0, 1, 0], cut=False)
        opt = self.opt_norm.clip_image(self.data_dic["opt_" + date])
        del self.data_dic

        ########### METRICS ##################
        opt =           opt[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]
        opt_fake = opt_fake[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]

        with open(path + output_file, 'a') as f:
            # test area (cloudy)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloudy)", mae, mse, rmse, psnr, sam, ssim)
            # test area (cloud-free)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud_free)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloud-free)", mae, mse, rmse, psnr, sam, ssim)
            # test area
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, test_mask, ssim_flag=True, dataset=args.dataset_name)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area", mae, mse, rmse, psnr, sam, ssim)

        del opt, opt_fake

    def GEE_metrics(self, args, date):

        # Loading normalizers used during training
        self.opt_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')
        output_file = '/Similarity_Metrics.txt'
        
        if date == "t0":
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            prefix = 0
            
            # Pará
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/img_2018.tif'

            # Mato Grosso OK
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/2019_09_15_2019_09_30.tif'

        elif date == "t1":
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            prefix = 1
            
            # Pará
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/img_2019.tif'

            # path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            # file_ = '/2019_01_01_2019_02_01.tif'
            # output_file = '/one month.txt'

            # path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            # file_ = '/2019_01_01_2019_04_01.tif'
            # output_file = '/three months.txt'

            # Mato Grosso OK
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/2020_09_10_2020_09_30.tif'

            # path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            # file_ = '/MG_1month_2020.tif'
            # output_file = '/one month.txt'

            path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            file_ = '/MG_3months_2020.tif'
            output_file = '/three months.txt'

        # Loading mask
        mask_tr_vl_ts = Split_Image(self, random_tiles='fixed')
        test_mask  = np.zeros_like(mask_tr_vl_ts)                
        test_mask[mask_tr_vl_ts==2] = 1

        # ====================== LOAD DATA =====================
        opt_fake = load_tiff_image(path + file_).astype('float32')
        opt_fake = opt_fake[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt_fake = opt_fake.transpose([1, 2, 0])
        opt_fake[np.isnan(opt_fake)] = np.nanmean(opt_fake)
        opt_fake = self.opt_norm.clip_image(opt_fake)

        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False, 
                                                                   flag_image = [0, 1, 0], cut=False)
        opt = self.opt_norm.clip_image(self.data_dic["opt_" + date])
        del self.data_dic

        ########### METRICS ##################
        opt =           opt[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]
        opt_fake = opt_fake[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]

        with open(path + output_file, 'a') as f:
            # Complete image
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, ssim_flag=True)
            Write_metrics_on_file(f, "Metrics " + date + "-- Complete Image", mae, mse, rmse, psnr, sam, ssim)

            # test area
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, test_mask, ssim_flag=True, dataset=args.dataset_name)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area", mae, mse, rmse, psnr, sam, ssim)




