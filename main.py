import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.keras.backend as K
from model import cGAN
import sys

parser = argparse.ArgumentParser(description='')

parser.add_argument('--epoch', dest='epoch', type=int, default=60, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=20000, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=143, help='scale images to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=2, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=10, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')

parser.add_argument('--lr', dest='lr', type=float, default=2e-4, help='initial learning rate for adam')
parser.add_argument("--lr_decay", type=float, default=0.7, help='learning rate decay')
parser.add_argument("--init_e", type=float, default=40, help='initial epoch')
parser.add_argument("--epoch_drop", type=float, default=1, help='epochs between steps lr decay')

parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, generate_image, create_dataset')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

#####___No@___#####

parser.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
parser.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
# Argument: --number_of_classes = --output_nc for using Deeplabv3
parser.add_argument("--number_of_classes", type=int, default=7, help="Number of classes to be predicted.")
parser.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
parser.add_argument("--resnet_model", default="resnet_v2_50", choices=["resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")

parser.add_argument("--sampling_type", default="dropout", choices=["dropout", 'none'], help="Noise used in the GAN")
parser.add_argument("--generator", default="deeplab", choices=["deeplab", "unet"], help="Generator's architecture ")
parser.add_argument("--discriminator", default="atrous", choices=["atrous", "pix2pix"], help="Discriminator's architecture ")
parser.add_argument("--spectral_norm", type=bool, default=True, choices=[True, False], help="Use spectral normalization to estabilize the network")
parser.add_argument("--data_augmentation", type=bool, default=True, choices=[True, False], help="Data Augmentation Flag")

parser.add_argument('--dataset_name', dest='dataset_name', default='SEN2MS-CR', choices=["SEN2MS-CR", "Para_10m", "MG_10m"], help='name of the dataset')
parser.add_argument('--datasets_dir', dest='datasets_dir', default='../../Datasets/', help='root dir')

parser.add_argument("--norm_type", default="min_max", choices=["min_max", "std", "wise_frame_mean"], help="Type of normalization ")
parser.add_argument('--image_size_tr', dest='image_size_tr', type=int, default=256, help='patch_size')
parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='image size in the dataset SEN2MS-CR')
parser.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")

parser.add_argument("--mask", default='fixed', choices=['fixed', 'random', 'k-fold'], help="Mask for training, validation and test regions")
parser.add_argument("--date", default="both", choices=["both", "d0", "d1"], help="Indicate which date generate ")
parser.add_argument("--patch_overlap", type=float, default=0.50, help="Overlap percentage between patches")


#####_________#####

args = parser.parse_args()


def actions():

    tf1.reset_default_graph()
    K.clear_session()

    if args.phase == 'train':
        config = tf1.ConfigProto()
        config.gpu_options.allow_growth = True
    else:   # Use CPU
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
        config = tf1.ConfigProto(log_device_placement=True)

    with tf1.Session(config=config) as sess:

        model = cGAN(sess, args, image_size_tr=args.image_size_tr, image_size=args.image_size, 
                        batch_size=args.batch_size, input_c_dim=args.input_nc,
                        output_c_dim=args.output_nc, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir) #args added for using DeepLabv3

        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'generate_complete_image':
            model.Translate_complete_image(args, date = "t0")
            model.Translate_complete_image(args, date = "t1")
        elif args.phase == 'GEE_metrics':
            # model.GEE_metrics(args, date = "t0")
            model.GEE_metrics(args, date = "t1")
        elif args.phase == 'Meraner_metrics':
            model.Meraner_metrics(args, date = "t0")
            model.Meraner_metrics(args, date = "t1")
        else:
            print ('...')


if __name__ == '__main__':
    actions()

