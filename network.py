import tensorflow as tf
# import tensorflow.compat.v1 as tf1
import tf_slim as slim
from resnet import resnet_v2, resnet_utils

import sys
import numpy as np

from tensorflow_addons.layers import SpectralNormalization
# from SpectralNormalizationKeras import DenseSN, ConvSN2D
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
                                    LeakyReLU, Activation, Conv2DTranspose, \
                                    Dropout
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal

def encoder_block(input_data, n_filters, k_size=3, strides=2, dilation_rate=1, activation='None', padding='same',
                  normalization = 'None', name='None', is_training=True):

    with tf.name_scope(name):
        # weight initialization
        init = RandomNormal(stddev=0.02)

        conv = Conv2D(n_filters, k_size, strides=strides, dilation_rate=dilation_rate, 
                      padding=padding, kernel_initializer=init)
        if normalization == "spectral":
            conv = SpectralNormalization(conv)
        x = conv(input_data)

        if normalization == "batch":
            x = BatchNormalization(momentum=0.9)(x, training=is_training)

        if activation == 'LReLU':
            x = LeakyReLU(alpha=0.2)(x)
        elif activation == 'ReLU':
            x = Activation('relu')(x)
        elif activation == 'sigmoid':
            x = Activation('sigmoid')(x)

        return x

def decoder_block(input_data, n_filters, k_size=3, strides=2, activation='None', padding='same',
                  normalization = 'None', name='None', is_training=True):

    with tf.name_scope(name):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        x = Conv2DTranspose(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init)(input_data)

        if normalization == "batch":
            x = BatchNormalization(momentum=0.9)(x, training=is_training)

        if activation == 'LReLU':
            x = LeakyReLU(alpha=0.2)(x)        
        elif activation == 'ReLU':
            x = Activation('relu')(x)

        return x

# ========================== PIX2PIX ==========================
def unet(self, input_shape, name="", is_train=True):

    with tf.name_scope(name):
        with tf.name_scope("U-Net"):

            input_layer  = Input(shape=input_shape)
                    
            e1 = encoder_block(input_layer, self.gf_dim, activation='LReLU', name='e1', is_training=is_train)
            e2 = encoder_block(e1, self.gf_dim*2, normalization = 'batch', activation='LReLU', name='e2', is_training=is_train)
            e3 = encoder_block(e2, self.gf_dim*4, normalization = 'batch', activation='LReLU', name='e3', is_training=is_train)
            e4 = encoder_block(e3, self.gf_dim*8, normalization = 'batch', activation='LReLU', name='e4', is_training=is_train)
            e5 = encoder_block(e4, self.gf_dim*8, normalization = 'batch', activation='LReLU', name='e5', is_training=is_train)
            e6 = encoder_block(e5, self.gf_dim*8, normalization = 'batch', activation='LReLU', name='e6', is_training=is_train)
            e7 = encoder_block(e6, self.gf_dim*8, normalization = 'batch', activation='LReLU', name='e7', is_training=is_train)
            enc = e7
            
            if self.image_size_tr >= 256:
                enc = encoder_block(e7, self.gf_dim*8, normalization = 'batch', activation='LReLU', name='e8', is_training=is_train)
                enc = decoder_block(enc, self.gf_dim*8, normalization = 'batch', activation='ReLU', name='d1', is_training=is_train)
                if 'dropout' in self.sampling_type:
                    enc = Dropout(self.rate_dropout)(enc, training=is_train)
                enc = tf.concat([enc, e7], 3)

            d2 = decoder_block(enc, self.gf_dim*8, normalization = 'batch', activation='ReLU', name='d2', is_training=is_train)
            if 'dropout' in self.sampling_type:
                d2 = Dropout(self.rate_dropout)(d2, training=is_train)
            d2 = tf.concat([d2, e6], 3)

            d3 = decoder_block(d2, self.gf_dim*8, normalization = 'batch', activation='ReLU', name='d3', is_training=is_train)
            if 'dropout' in self.sampling_type:
                d3 = Dropout(self.rate_dropout)(d3, training=is_train)
            d3 = tf.concat([d3, e5], 3)

            d4 = decoder_block(d3, self.gf_dim*8, normalization = 'batch', activation='ReLU', name='d4', is_training=is_train)
            d4 = tf.concat([d4, e4], 3)

            d5 = decoder_block(d4, self.gf_dim*4, normalization = 'batch', activation='ReLU', name='d5', is_training=is_train)
            d5 = tf.concat([d5, e3], 3)

            d6 = decoder_block(d5, self.gf_dim*2, normalization = 'batch', activation='ReLU', name='d6', is_training=is_train)
            d6 = tf.concat([d6, e2], 3)

            d7 = decoder_block(d6, self.gf_dim  , normalization = 'batch', activation='ReLU', name='d7', is_training=is_train)
            d7 = tf.concat([d7, e1], 3)

            d8 = decoder_block(d7, self.output_c_dim, name='d8', is_training=is_train)

            output = Activation('tanh', name='act_tanh')(d8)

            model = Model(inputs=[input_layer], outputs=[output], name=name)
            model.summary()
            
            return model

def pix2pix_discriminator(self, input_shape, name="", is_train=True, normalization='spectral'):
    """
    normalization {'spectral', 'batch'}
    """

    with tf.name_scope(name):
        with tf.name_scope("Patch-GAN"):

            input_layer  = Input(shape=input_shape)

            e1 = encoder_block(input_layer, self.gf_dim  , normalization = normalization, activation='LReLU', name='e1', is_training=is_train)
            e2 = encoder_block(e1         , self.gf_dim*2, normalization = normalization, activation='LReLU', name='e2', is_training=is_train)
            e3 = encoder_block(e2         , self.gf_dim*4, normalization = normalization, activation='LReLU', name='e3', is_training=is_train)
            e4 = encoder_block(e3         , self.gf_dim*8, normalization = normalization, activation='LReLU', name='e4', is_training=is_train)
            logits = encoder_block(e4 , 1, k_size=1, strides=1, name='logits', is_training=is_train)
            output = Activation('sigmoid')(logits)

            model = Model(inputs=[input_layer], outputs=[output, logits], name=name)
            model.summary()
            
            return model

# ========================== ATROUS-CGAN ==========================
# # # # # # # # # # # # # DEEPLAB # # # # # # # # # # # # #
@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, rate=None, depth=256, reuse=None):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """
    
    with tf.name_scope(scope):
        feature_map_size = tf.shape(net)
        
        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)

        image_level_features = resnet_utils.conv(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                           activation_fn=None)
        image_level_features = tf.image.resize(image_level_features, feature_map_size[1:3], 
                                               method=tf.image.ResizeMethod.BILINEAR)

        at_pool1x1 = resnet_utils.conv(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = resnet_utils.conv(net, depth, [3, 3], scope="conv_3x3_1", rate=rate[0], activation_fn=None)

        at_pool3x3_2 = resnet_utils.conv(net, depth, [3, 3], scope="conv_3x3_2", rate=rate[1], activation_fn=None)

        at_pool3x3_3 = resnet_utils.conv(net, depth, [3, 3], scope="conv_3x3_3", rate=rate[2], activation_fn=None)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3)

        net = resnet_utils.conv(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

        return net

@slim.add_arg_scope
def Decoder(dec, scope, args, skip_connections, reuse=None, rate_dropout=0.5, is_train=True):
    
    with tf.name_scope(scope):

        depth = dec.get_shape()[3]
        if 'dropout' in args.sampling_type:
            dec = Dropout(rate_dropout)(dec, training=is_train)
        
        if args.image_size_tr == 256 and args.output_stride == 16:
            dec = decoder_block(dec, depth//(2**2), normalization = 'batch', activation='LReLU', name='d1', is_training=is_train)
            dec = tf.concat([dec, skip_connections[0]], 3)

        dec = decoder_block(dec, depth//(2**3), normalization = 'batch', activation='LReLU', name='d2', is_training=is_train)
        dec = tf.concat([dec, skip_connections[1]], 3)
        
        dec = decoder_block(dec, args.output_nc, name='d3', is_training=is_train)
                                                                
    return dec

def deeplab_v3(inputs, args, is_training, reuse, rate_dropout = 0.5):

    with tf.name_scope("DeepLabV3"):
        with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                        args.batch_norm_decay,
                                                        args.batch_norm_epsilon)):
            
            resnet = getattr(resnet_v2, args.resnet_model)
            _, end_points = resnet(inputs,
                                args.number_of_classes,
                                is_training=is_training,
                                global_pool=False,
                                spatial_squeeze=False,
                                output_stride=args.output_stride,
                                reuse=reuse)

            # get outputs for skip connections            
            skip_connections = [end_points['Generator/DeepLabV3/' + args.resnet_model + '/block2/unit_3/bottleneck_v2'],
                                end_points['Generator/DeepLabV3/' + args.resnet_model + '/block1/unit_2/bottleneck_v2']]

            # rates
            rate = [6, 12, 18]
            # get block 4 feature outputs
            net = end_points['Generator/DeepLabV3/' + args.resnet_model + '/block4']

            if 'dropout' in args.sampling_type:
                net = Dropout(rate_dropout)(net, training=is_training)

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", rate = rate, depth=512, reuse=reuse)
            net = LeakyReLU(alpha=0.2, name='_act_LReLU')(net)
            
            net = Decoder(net, "Decoder", args, skip_connections, reuse=reuse, rate_dropout=rate_dropout, is_train=is_training)
            
            
            # for key, value in end_points.items():
            #     print (key)
            #     print (end_points[key].get_shape())
            # sys.exit()

            # Rest of deeplabv3
            # net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,
            #                   normalizer_fn=None, scope='logits')
            # net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,
            #                   scope='logits')
            # # resize the output logits to match the labels dimensions
            # size = tf.shape(inputs)[1:3]
            # net = tf1.image.resize_nearest_neighbor(net, size)
            # net = tf1.image.resize_bicubic(net, size)
            # net = tf1.image.resize_bilinear(net, size)

            return net

def deeplab(self, input_shape, name="", reuse = False, is_train=True):

    with tf.name_scope(name):
        input_layer  = Input(shape=input_shape)

        net = deeplab_v3(input_layer, self.args, is_training = is_train, reuse=reuse, rate_dropout = self.rate_dropout)

        output = Activation('tanh')(net)

        model = Model(inputs=[input_layer], outputs=[output], name=name)
        model.summary()

        return model

def atrous_discriminator(self, input_shape, name="", is_train=True, normalization='spectral'):

    """
    normalization {'spectral', 'batch'}
    """

    def atrous_convs(net, rate=None, depth=256, name="", normalization = 'None'):
        """
        ASPP layer 1×1 convolution and three 3×3 atrous convolutions
        """        
        with tf.name_scope(name):

            at_pool1x1 = encoder_block(net, depth, k_size=1, strides=1, 
                                       padding='same', normalization = normalization, 
                                       name='conv_1x1', is_training=is_train)

            at_pool3x3_1 = encoder_block(net, depth, k_size=3, strides=1, dilation_rate=rate[0],
                                         padding='same', normalization = normalization, 
                                         name='conv3x3_1', is_training=is_train)

            at_pool3x3_2 = encoder_block(net, depth, k_size=3, strides=1, dilation_rate=rate[1],
                                         padding='same', normalization = normalization, 
                                         name='conv3x3_2', is_training=is_train)

            at_pool3x3_3 = encoder_block(net, depth, k_size=3, strides=1, dilation_rate=rate[2],
                                         padding='same', normalization = normalization, 
                                         name='conv3x3_3', is_training=is_train)

            net = tf.concat((at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3)

            net = encoder_block(net, depth, k_size=1, strides=1, 
                                padding='same', normalization = normalization, 
                                name='conv_1x1_output', is_training=is_train)

            return net

    with tf.name_scope(name):
        with tf.name_scope("Atrous"):

            input_layer  = Input(shape=input_shape)

            rate = [2, 3, 4]
            x = atrous_convs (input_layer, rate=rate, depth=self.df_dim, name="d_atrous_1", normalization = normalization)
            x = encoder_block(x, self.df_dim  , normalization = normalization, activation='LReLU', name='e1', is_training=is_train)

            x = atrous_convs (x, rate=rate, depth=self.df_dim  , name="d_atrous_2", normalization = normalization)
            x = encoder_block(x, self.df_dim*2, normalization = normalization, activation='LReLU', name='e2', is_training=is_train)

            x = atrous_convs (x, rate=rate, depth=self.df_dim*2, name="d_atrous_3", normalization = normalization)
            x = encoder_block(x, self.df_dim*4, normalization = normalization, activation='LReLU', name='e3', is_training=is_train)

            x = encoder_block (x, self.df_dim*8, normalization = normalization, activation='LReLU', name='e4', is_training=is_train)

            logits = encoder_block(x , 1, k_size=1, strides=1, name='logits', is_training=is_train)
            output = Activation('sigmoid')(logits)

            model = Model(inputs=[input_layer], outputs=[output, logits], name=name)
            model.summary()

            return model





