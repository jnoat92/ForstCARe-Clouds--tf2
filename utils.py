"""
No@
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
from skimage import exposure
from skimage.metrics import structural_similarity
from sklearn.preprocessing._data import _handle_zeros_in_scale
import matplotlib.pyplot as plt
import sys
import itertools
from osgeo import gdal
import rasterio



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    img = gdal_header.ReadAsArray()
    return img

def GeoReference_Raster_from_Source_data(source_file, numpy_image, target_file):

    with rasterio.open(source_file) as src:
        ras_meta = src.profile

    ras_meta.update(count=10)

    with rasterio.open(target_file, 'w', **ras_meta) as dst:
        dst.write(numpy_image)



def db2linear(x):
    return 10.0**(x/10.0)

def filter_outliers(img, bins=2**16-1, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)] = np.mean(img) # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    min_value, max_value = [], []
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(), bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        min_value.append(hist[1][len(cum_hist[cum_hist<bth])])
        max_value.append(hist[1][len(cum_hist[cum_hist<uth])])
        
    return [np.array(min_value), np.array(max_value)]

class Min_Max_Norm_Denorm():

    def __init__(self, img, mask, feature_range=[-1, 1]):

        self.feature_range = feature_range
        self.clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.0005, uth=0.9995, mask=mask)
        self.min_val = np.nanmin(img, axis=(0,1))
        self.max_val = np.nanmax(img, axis=(0,1))

        self.min_val = np.clip(self.min_val, self.clips[0], None)
        self.max_val = np.clip(self.max_val, None, self.clips[1])
    
    def clip_image(self, img):
        return np.clip(img.copy(), self.clips[0], self.clips[1])

    def Normalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale
        
        img = self.clip_image(img.copy())
        img *= scale
        img += min_
        return img

    def Denormalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale

        img = img.copy() - min_
        img /= scale
        return img



def Split_Tiles(tiles_list, xsz, ysz, stride=256, patch_size=256):
   
    coor = []
    for i in tiles_list:
        b = np.random.choice([-1, 1])
        if b == 1:
            x = np.arange(0, xsz - patch_size + 1, b*stride)
        else:
            x = np.arange(xsz - patch_size, -1, b*stride)
       
        b = np.random.choice([-1, 1])
        if b == 1:
            y = np.arange(0, ysz - patch_size + 1, b*stride)
        else:
            y = np.arange(ysz - patch_size, -1, b*stride)
       
        coor += list(itertools.product(x + i[0], y + i[1]))

    for i in range(len(coor)):
        coor[i] = (coor[i][0], coor[i][1], i)

    return coor

def Split_Image(obj, rows=1000, cols=1000, no_tiles_h=5, no_tiles_w=5, random_tiles = "fixed"):

    xsz = rows // no_tiles_h
    ysz = cols // no_tiles_w

    if random_tiles == 'random':

        # Tiles coordinates
        h = np.arange(0, rows, xsz)
        w = np.arange(0, cols, ysz)
        if (rows % no_tiles_h): h = h[:-1]
        if (cols % no_tiles_w): w = w[:-1]
        tiles = list(itertools.product(h, w))

        np.random.seed(3); np.random.shuffle(tiles)

        # Take test tiles
        idx = len(tiles) * 50 // 100; idx += (idx == 0)
        test_tiles = tiles[:idx]
        train_tiles = tiles[idx:]
        # Take validation tiles
        idx = len(train_tiles) * 10 // 100; idx += (idx == 0)
        val_tiles = train_tiles[:idx]
        train_tiles = train_tiles[idx:]

        mask = np.zeros((rows, cols))
        for i in val_tiles:
            finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
            finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
            mask[i[0]:finx, i[1]:finy] = 1
        for i in test_tiles:
            finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
            finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
            mask[i[0]:finx, i[1]:finy] = 2
        
        # save_mask = Image.fromarray(np.uint8(mask*255/2))
        # save_mask.save('../datasets/' + obj.args.dataset_name + '/mask_train_val_test.tif')
        # np.save('../datasets/' + obj.args.dataset_name + '/tiles', tiles)

    elif random_tiles == 'k-fold':

        k = obj.args.k
        mask = Image.open('../datasets/' + obj.args.dataset_name + '/mask_train_val_test_fold_' + str(k) + '.tif')
        mask = np.array(mask) / 255

        # tiles = np.load('../datasets/' + obj.args.dataset_name + '/tiles.npy')

        # # Split in folds
        # size_fold = len(tiles) // obj.args.n_folds
        # test_tiles = tiles[k*size_fold:(k+1)*size_fold]
        # train_tiles = np.concatenate((tiles[:k*size_fold], tiles[(k+1)*size_fold:]))
        # # Take validation tiles
        # np.random.shuffle(train_tiles)
        # idx = len(train_tiles) * 10 // 100; idx += (idx == 0)
        # val_tiles = train_tiles[:idx]
        # train_tiles = train_tiles[idx:]

        # mask = np.zeros((rows, cols))
        # for i in val_tiles:
        #     finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
        #     finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
        #     mask[i[0]:finx, i[1]:finy] = 1
        # for i in test_tiles:
        #     finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
        #     finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
        #     mask[i[0]:finx, i[1]:finy] = 2
        
        # save_mask = Image.fromarray(np.uint8(mask*255))
        # save_mask.save('../datasets/' + obj.args.dataset_name + '/mask_train_val_test_fold_' + str(k) + '.tif')

    elif random_tiles == 'fixed':
        # Distribute the tiles from a mask
        mask  = np.load(obj.args.datasets_dir + obj.args.dataset_name + obj.mask_tr_vl_ts_name + '.npy')
        img = Image.fromarray(np.uint8((mask/2)*255))
        img.save(obj.args.datasets_dir + obj.args.dataset_name + obj.mask_tr_vl_ts_name + '.tiff')
    
    return mask

def Split_in_Patches(rows, cols, patch_size, mask, 
                     lbl, augmentation_list, cloud_mask, 
                     prefix=0, percent=0):

    """
    Everything  in this function is made operating with
    the upper left corner of the patch
    """

    # Percent of overlap between consecutive patches.
    overlap = round(patch_size * percent)
    overlap -= overlap % 2
    stride = patch_size - overlap
    # Add Padding to the image to match with the patch size
    step_row = (stride - rows % stride) % stride
    step_col = (stride - cols % stride) % stride
    pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
    lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')
    cloud_mask = np.pad(cloud_mask, pad_tuple_msk, mode = 'symmetric')

    k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
    print('Total number of patches: %d x %d' %(k1, k2))

    train_mask = np.zeros_like(mask_pad)
    val_mask = np.zeros_like(mask_pad)
    test_mask = np.zeros_like(mask_pad)
    train_mask[mask_pad==0] = 1
    test_mask [mask_pad==2] = 1
    val_mask = (1-train_mask) * (1-test_mask)

    train_patches, val_patches, test_patches = [], [], []
    only_bck_patches = 0
    cloudy_patches = 0
    lbl[lbl!=1] = 0
    for i in range(k1):
        for j in range(k2):
            # Train
            if train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                if cloud_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    cloudy_patches += 1
                    continue
                for k in augmentation_list:
                    train_patches.append((prefix, i*stride, j*stride, k))
                if not lbl[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    # train_patches.append((prefix, i*stride, j*stride, 0))
                    only_bck_patches += 1
            # Test                !!!!!Not necessary with high overlap!!!!!!!!
            elif test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                test_patches.append((prefix, i*stride, j*stride, 0))
            # Val                 !!!!!Not necessary with high overlap!!!!!!!!
            elif val_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                val_patches.append((prefix, i*stride, j*stride, 0))
    print('Training Patches with background only: %d' %(only_bck_patches))
    print('Patches with clouds in the cloud-free image: %d' %(cloudy_patches))
    
    return train_patches, val_patches, test_patches, step_row, step_col, overlap

def create_dataset_coordinates(obj, prefix = 0, padding=True,
                               flag_image = [1, 1, 1], cut=True):
    
    '''
        Generate patches for trn, val and tst
    '''

    patch_size = obj.image_size_tr

    # number of tiles per axis
    no_tiles_h, no_tiles_w = 5, 5
    rows, cols = obj.lims[1] - obj.lims[0], obj.lims[3] - obj.lims[2]
    mask_tr_vl_ts = Split_Image(obj, rows, cols, no_tiles_h, 
                           no_tiles_w, random_tiles=obj.args.mask)

    # Loading Labels
    lbl = np.load(obj.labels_path + obj.labels_name + '.npy')
    lbl = lbl[obj.lims[0]: obj.lims[1], obj.lims[2]: obj.lims[3]]
    lbl[lbl==2.0] = 3.0; lbl[lbl==1.0] = 2.0; lbl[lbl==3.0] = 1.0
    img = Image.fromarray(np.uint8((lbl/2)*255))
    img.save(obj.labels_path + obj.labels_name + '.tiff')

    # Loading cloud mask
    cloud_mask = np.zeros((rows, cols))    
    
    # Generate Patches for trn, val and tst
    if obj.args.data_augmentation:
        augmentation_list = [-1]                    # A random transformation each epoch
        # augmentation_list = [0, 1, 2, 3, 4, 5]    # All transformation each epoch
    else:
        augmentation_list = [0]                         # Without transformations
    train_patches, val_patches, test_patches, \
    step_row, step_col, overlap = Split_in_Patches(rows, cols, patch_size, 
                                                   mask_tr_vl_ts, lbl, augmentation_list,
                                                   cloud_mask, prefix = prefix,
                                                   percent=obj.args.patch_overlap)
    pad_tuple = ( (overlap//2, overlap//2+step_row), (overlap//2, overlap//2+step_col), (0,0) )
    del lbl, cloud_mask
    
    print('--------------------')
    print('Training Patches: %d' %(len(train_patches)))
    print('Validation Patches: %d' %(len(val_patches)))
    print('Testing Patches: %d' %(len(test_patches)))
    
    data_dic = {}

    # Sentinel 1
    if flag_image[0]:
        sar_vv = load_tiff_image(obj.sar_path + obj.sar_name[0] + '.tif').astype('float32')
        sar_vh = load_tiff_image(obj.sar_path + obj.sar_name[1] + '.tif').astype('float32')
        sar = np.concatenate((np.expand_dims(sar_vv, 2), np.expand_dims(sar_vh, 2)), axis=2)
        del sar_vh, sar_vv
        if cut:
            sar = sar[obj.lims[0]:obj.lims[1], obj.lims[2]:obj.lims[3],:]
        sar[np.isnan(sar)] = np.nanmean(sar)
        sar = db2linear(sar)
        sar_norm = Min_Max_Norm_Denorm(sar, mask_tr_vl_ts)
    else:
        sar, sar_norm = [], []
    
    if padding:
        # Add Padding to the images to match with the patch size
        data_dic["sar_t" + str(prefix)] = np.pad(sar, pad_tuple, mode = 'symmetric')
    else:
        data_dic["sar_t" + str(prefix)] = sar
    del sar

    # Sentinel 2
    if flag_image[1]:
        for i in range(len(obj.opt_name)):        
            img = load_tiff_image(obj.opt_path + obj.opt_name[i] + '.tif').astype('float32')
            if len(img.shape) == 2: img = img[np.newaxis, ...]
            if i:
                opt = np.concatenate((opt, img), axis=0)
            else:
                opt = img
        del img
        opt = opt[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt = opt.transpose([1, 2, 0])
        if cut:
            opt = opt[obj.lims[0]:obj.lims[1], obj.lims[2]:obj.lims[3],:]
        opt[np.isnan(opt)] = np.nanmean(opt)
        opt_norm = Min_Max_Norm_Denorm(opt, mask_tr_vl_ts)
    else:
        opt, opt_norm = [], []

    if padding:
        # Add Padding to the images to match with the patch size
        data_dic["opt_t" + str(prefix)] = np.pad(opt, pad_tuple, mode = 'symmetric')
    else:
        data_dic["opt_t" + str(prefix)] = opt
    del opt

    # Sentinel 2 cloudy
    if flag_image[2]:
        for i in range(len(obj.opt_cloudy_name)):
            img = load_tiff_image(obj.opt_cloudy_path + obj.opt_cloudy_name[i] + '.tif').astype('float32')
            if len(img.shape) == 2: img = img[np.newaxis, ...]
            if i:
                opt_cloudy = np.concatenate((opt_cloudy, img), axis=0)
            else:
                opt_cloudy = img
        del img
        opt_cloudy = opt_cloudy[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt_cloudy = opt_cloudy.transpose([1, 2, 0])
        if cut:
            opt_cloudy = opt_cloudy[obj.lims[0]:obj.lims[1], obj.lims[2]:obj.lims[3],:]
        opt_cloudy[np.isnan(opt_cloudy)] = np.nanmean(opt_cloudy)
    else:
        opt_cloudy = []
    
    if padding:
        # Add Padding to the images to match with the patch size
        data_dic["opt_cloudy_t" + str(prefix)] = np.pad(opt_cloudy, pad_tuple, mode = 'symmetric')
    else:
        data_dic["opt_cloudy_t" + str(prefix)] = opt_cloudy
    del opt_cloudy

    print('Dataset created!!')
    return train_patches, val_patches, test_patches, data_dic, sar_norm, opt_norm

def create_dataset_both_images(obj):
    
    obj.sar_name = obj.sar_name_t0
    obj.opt_name = obj.opt_name_t0
    obj.opt_cloudy_name = obj.opt_cloudy_name_t0
    train_patches_0, val_patches_0, test_patches_0, \
        data_dic_0, sar_norm_0, opt_norm_0 = create_dataset_coordinates(obj, prefix = 0)

    obj.sar_name = obj.sar_name_t1
    obj.opt_name = obj.opt_name_t1
    obj.opt_cloudy_name = obj.opt_cloudy_name_t1
    obj.args.patch_overlap += 0.03 # In case that t0 and t1 images are co-registered,
                                   # this allows to extract patches that are not co-registered between dates.
    train_patches_1, val_patches_1, test_patches_1, \
        data_dic_1, sar_norm_1, opt_norm_1 = create_dataset_coordinates(obj, prefix = 1)
    obj.args.patch_overlap -= 0.03

    train_patches = train_patches_0 + train_patches_1
    val_patches   = val_patches_0   + val_patches_1
    test_patches  = test_patches_0  + test_patches_1
    data_dic    = {**data_dic_0, **data_dic_1}

    # Use the same normalization parameters in all images
    sar_norm_0.min_val = np.minimum(sar_norm_0.min_val, sar_norm_1.min_val)
    sar_norm_0.max_val = np.maximum(sar_norm_0.max_val, sar_norm_1.max_val)
    sar_norm_0.clips[0] = np.minimum(sar_norm_0.clips[0], sar_norm_1.clips[0])
    sar_norm_0.clips[1] = np.maximum(sar_norm_0.clips[1], sar_norm_1.clips[1])

    opt_norm_0.min_val = np.minimum(opt_norm_0.min_val, opt_norm_1.min_val)
    opt_norm_0.max_val = np.maximum(opt_norm_0.max_val, opt_norm_1.max_val)
    opt_norm_0.clips[0] = np.minimum(opt_norm_0.clips[0], opt_norm_1.clips[0])
    opt_norm_0.clips[1] = np.maximum(opt_norm_0.clips[1], opt_norm_1.clips[1])

    return train_patches, val_patches, test_patches, data_dic, sar_norm_0, opt_norm_0




def Transform(arr, b):

    sufix = ''

    if b == 1:
        arr = np.rot90(arr, k = 1)
        sufix = '_rot90'
    elif b == 2:
        arr = np.rot90(arr, k = 2)
        sufix = '_rot180'
    elif b == 3:
        arr = np.rot90(arr, k = 3)
        sufix = '_rot270'
    elif b == 4:
        arr = np.flipud(arr)
        sufix = '_flipud'
    elif b == 5:
        arr = np.fliplr(arr)
        sufix = '_fliplr'
    elif b == 6:
        if len(arr.shape) == 3:
            arr = np.transpose(arr, (1, 0, 2))
        elif len(arr.shape) == 2:
            arr = np.transpose(arr, (1, 0))
        sufix = '_transpose'
    elif b == 7:
        if len(arr.shape) == 3:
            arr = np.rot90(arr, k = 2)
            arr = np.transpose(arr, (1, 0, 2))
        elif len(arr.shape) == 2:
            arr = np.rot90(arr, k = 2)
            arr = np.transpose(arr, (1, 0))
        sufix = '_transverse'

    return arr, sufix

def Data_augmentation(s1, s2, s2_cloudy,
                      id_transform,
                      fine_size=256,
                      random_crop_transformation=False):

    if id_transform == -1:
        id_transform = np.random.randint(6)
        
    s1, _ = Transform(s1, id_transform)
    s2, _ = Transform(s2, id_transform)
    s2_cloudy, _ = Transform(s2_cloudy, id_transform)

    if random_crop_transformation and np.random.rand() > .5:
        dif_size = round(fine_size * 10/100)
        h1 = np.random.randint(dif_size + 1)
        w1 = np.random.randint(dif_size + 1)

        s1 = np.float32(resize(s1, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))
        s2 = np.float32(resize(s2, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))
        s2_cloudy = np.float32(resize(s2_cloudy, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))

        s1 = s1[h1:(h1+fine_size), w1:(w1+fine_size)]
        s2 = s2[h1:h1+fine_size, w1:w1+fine_size]
        s2_cloudy = s2_cloudy[h1:h1+fine_size, w1:w1+fine_size]

    return s1, s2, s2_cloudy

def Take_patches(patch_list, idx, data_dic,
                 fine_size=256,
                 random_crop_transformation=False):

    sar = data_dic['sar_t' + str(patch_list[idx][0])] \
                  [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                   patch_list[idx][2]:patch_list[idx][2]+fine_size, :]
    opt = data_dic['opt_t' + str(patch_list[idx][0])] \
                  [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                   patch_list[idx][2]:patch_list[idx][2]+fine_size, :]
    opt_cloudy = data_dic['opt_cloudy_t' + str(patch_list[idx][0])] \
                         [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                          patch_list[idx][2]:patch_list[idx][2]+fine_size, :]

    sar, opt, opt_cloudy = Data_augmentation(sar, opt, opt_cloudy,
                                             patch_list[idx][3],
                                             fine_size=fine_size,
                                             random_crop_transformation=random_crop_transformation)
    
    return sar, opt, opt_cloudy



def plot_hist(img, bins, lim, name, output_path):
    hist , bins  = np.histogram(img, bins=bins)
    plt.figure(figsize=(6, 4))
    plt.plot(bins[1:], hist/np.prod(img.shape))
    plt.title("{}".format(name))
    plt.xlabel('bins')
    plt.ylabel('count')
    # plt.xlim(lim)
    # plt.ylim([0, 0.011])
    plt.tight_layout()
    plt.savefig(output_path + "/" + name + ".png", dpi=500)
    plt.close()

def save_image(image, file_path, sensor = "s2"):

    if sensor == "s2":        
        image[:,:,0] = exposure.equalize_adapthist(image[:,:,0] , clip_limit=0.01)
        image[:,:,1] = exposure.equalize_adapthist(image[:,:,1] , clip_limit=0.01)
        image[:,:,2] = exposure.equalize_adapthist(image[:,:,2] , clip_limit=0.01)
        image *= 255
        image = Image.fromarray(np.uint8(image))
        image.save(file_path + ".tif")
    
    elif sensor == "s1":
        image = exposure.equalize_adapthist(image[:,:,1] , clip_limit=0.009)
        image = Image.fromarray(np.uint8(image*255))
        image.save(file_path + ".tif")

def generate_samples(self, output_path, idx, patch_list=None, epoch=0, real_flag = False):

    s1, s2, s2_cloudy = Take_patches(patch_list, idx, data_dic = self.data_dic,
                                        fine_size=self.image_size,
                                        random_crop_transformation=False)
    file_name_opt = ""
    file_name_sar = ""

    s2_fake = self.generator.predict(np.concatenate((s1[np.newaxis, ...],
                                                     s2_cloudy[np.newaxis, ...]),
                                                     axis=3))
    plot_hist(s2_fake[0,:,:,:], 1000, [-1, 1], "_histogram FAKE S2 {}".format(epoch), output_path)
    
    s2_fake = self.opt_norm.Denormalize(s2_fake[0,:,:,:]) / self.opt_norm.max_val.max()
    s2_fake = s2_fake[:, :, self.visible_bands] 
    file_name = "/{}_{}_{}".format("FAKE", file_name_opt, epoch)
    save_image(s2_fake, output_path + file_name, sensor = "s2")

    if real_flag:

        plot_hist(s2, 1000, [-1, 1], "_histogram REAL S2", output_path)
        s2 = self.opt_norm.Denormalize(s2) / self.opt_norm.max_val.max()
        s2 = s2[:, :, self.visible_bands] 
        file_name = "/{}_{}".format("REAL", file_name_opt)
        save_image(s2, output_path + file_name, sensor = "s2")

        s2_cloudy = self.opt_norm.Denormalize(s2_cloudy) / self.opt_norm.max_val.max()
        s2_cloudy = s2_cloudy[:, :, self.visible_bands] 
        file_name = "/{}_{}".format("REAL_cloudy", file_name_opt)
        save_image(s2_cloudy, output_path + file_name, sensor = "s2")

        s1 = self.sar_norm.Denormalize(s1) / self.sar_norm.max_val.max()
        file_name = "/{}_{}".format("SAR", file_name_sar)
        save_image(s1, output_path + file_name, sensor = "s1")



class Image_reconstruction(object):

    def __init__ (self, inputs, model, output_c_dim, patch_size=256, overlap_percent=0):

        self.inputs = inputs
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
        self.model = model
    
    def Inference(self, tile):
        
        '''
        Normalize before calling this method
        '''

        num_rows, num_cols, _ = tile.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
        
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
        tile_pad = np.pad(tile, pad_tuple, mode = 'symmetric')

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((k1*stride, k2*stride, self.output_c_dim), dtype='float32')

        for i in range(k1):
            for j in range(k2):
                
                patch = tile_pad[i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size), :]
                patch = patch[np.newaxis,...]
                infer = self.model.predict(patch)

                probs[i*stride : i*stride+stride, 
                      j*stride : j*stride+stride, :] = infer[0, overlap//2 : overlap//2 + stride, 
                                                                overlap//2 : overlap//2 + stride, :]
            print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[:k1*stride-step_row, :k2*stride-step_col]

        return probs



# ============= METRICS =============
def MAE(y_true, y_pred):
    """Computes the MAE over the full image."""
    return np.mean(np.abs(y_pred - y_true))
    # return K.mean(K.abs(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))

def MSE(y_true, y_pred):
    """Computes the MSE over the full image."""
    return np.mean(np.square(y_pred - y_true))
    # return K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))

def RMSE(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return np.sqrt(np.mean(np.square(y_pred - y_true)))
    # return K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))

def SAM(y_true, y_pred):
    """Computes the SAM over the full image."""    
    mat = np.sum(y_true * y_pred, axis=-1)
    mat /= np.sqrt(np.sum(y_true * y_true, axis=-1))
    mat /= np.sqrt(np.sum(y_pred * y_pred, axis=-1))
    mat = np.arccos(np.clip(mat, -1, 1))

    return np.mean(mat)

def PSNR(y_true, y_pred):
    """Computes the PSNR over the full image."""
    # y_true *= 2000
    # y_pred *= 2000
    # rmse = K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))
    # return 20.0 * (K.log(10000.0 / rmse) / K.log(10.0))
    rmse = RMSE(y_true, y_pred)
    return 20.0 * np.log10(10000.0 / rmse)

def SSIM(y_true, y_pred):
    """Computes the SSIM over the full image."""
    y_true = np.clip(y_true, 0, 10000.0)
    y_pred = np.clip(y_pred, 0, 10000.0)
    ssim = structural_similarity(y_true, y_pred, data_range=10000.0, multichannel=True)

    return ssim

def METRICS(y_true, y_pred, mask=None, ssim_flag=False, dataset="Para_10m"):

    if ssim_flag:
        no_tiles_h = 5
        no_tiles_w = 4
    
        rows, cols, _ = y_true.shape 
        xsz = rows // no_tiles_h
        ysz = cols // no_tiles_w

        # Tiles coordinates
        h = np.arange(0, rows, xsz)
        w = np.arange(0, cols, ysz)
        if (rows % no_tiles_h): h = h[:-1]
        if (cols % no_tiles_w): w = w[:-1]
        tiles = list(itertools.product(h, w))
        
        if mask is not None:

            if dataset == "Para_10m":
                test_tiles = [1, 3, 6, 8, 9, 11, 14, 15, 16, 17]
            elif dataset == "MG_10m":
                test_tiles = [0, 2, 6, 7, 8, 9, 11, 13, 14, 16]

            ssim = []
            for i in test_tiles:
                print(i)
                img1 = y_true[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                img2 = y_pred[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                ssim.append(SSIM(img1, img2))

        else:
            # Calculate ssim for the whole image
            test_tiles = range(len(tiles))
            ssim = []
            for i in test_tiles:
                print(i)
                img1 = y_true[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                img2 = y_pred[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                ssim.append(SSIM(img1, img2))
    else:
        ssim = [-1.0]
    ssim = np.asarray(ssim)

    if mask is not None:
        y_true = y_true[mask==1]
        y_pred = y_pred[mask==1]

    psnr = PSNR(y_true, y_pred)
    y_true /= 2000
    y_pred /= 2000
    mae  = MAE (y_true, y_pred)
    mse  = MSE (y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    sam  = SAM (y_true, y_pred)

    # ----------------
    y_true *= 2000
    y_pred *= 2000
    # ----------------

    return mae, mse, rmse, psnr, sam, ssim
    
def Write_metrics_on_file(f, title, mae, mse, rmse, psnr, sam, ssim):

    print("__________ {} __________\n".format(title))
    print("mae, mse, rmse, psnr, sam, ssim")
    print(mae, mse, rmse, psnr, sam, ssim.mean())

    f.write("__________ {} __________\n".format(title))
    f.write("MAE  = %.4f\n"%( mae))
    f.write("MSE  = %.4f\n"%( mse))
    f.write("RMSE = %.4f\n"%(rmse))
    f.write("PSNR = %.4f\n"%(psnr))
    f.write("SAM  = %.4f\n"%( sam))
    f.write("SSIM = %.4f\n"%(ssim.mean()))
    f.write("\n")