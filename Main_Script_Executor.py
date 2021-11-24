"""
No@
"""
import os

Schedule = []                

# ========= EXPERIMENTS TRAINING WITH in the same image (MG_10m) ============
# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase train \
#                                 --batch_size 1 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase generate_complete_image \
#                                 --batch_size 1 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator deeplab --discriminator atrous --phase train \
#                                 --batch_size 2 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 256 --output_stride 16 \
#                                 --patch_overlap 0.35 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator deeplab --discriminator atrous --phase generate_complete_image \
#                                 --batch_size 2 --epoch 60 --dataset_name MG_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 256 --output_stride 16 \
#                                 --patch_overlap 0.35 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --phase GEE_metrics --dataset_name MG_10m --test_dir ./test")
# Schedule.append("python main.py --phase Meraner_metrics --dataset_name MG_10m --test_dir ./test")








# ========= EXPERIMENTS TRAINING WITH in the same image (Para_10m) ============
# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase train \
#                                 --batch_size 1 --epoch 60 --dataset_name Para_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator unet --discriminator pix2pix --phase generate_complete_image \
#                                 --batch_size 1 --epoch 60 --dataset_name Para_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 128  \
#                                 --patch_overlap 0.0 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

Schedule.append("python main.py --generator deeplab --discriminator atrous --phase train \
                                --batch_size 2 --epoch 60 --dataset_name Para_10m \
                                --datasets_dir ../../Datasets/ --image_size_tr 256 --output_stride 16 \
                                --patch_overlap 0.4 --date both\
                                --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --generator deeplab --discriminator atrous --phase generate_complete_image \
#                                 --batch_size 2 --epoch 60 --dataset_name Para_10m \
#                                 --datasets_dir ../../Datasets/ --image_size_tr 256 --output_stride 16 \
#                                 --patch_overlap 0.4 --date both\
#                                 --checkpoint_dir ./checkpoint --sample_dir ./sample --test_dir ./test")

# Schedule.append("python main.py --phase GEE_metrics --dataset_name Para_10m --test_dir ./test")
# Schedule.append("python main.py --phase Meraner_metrics --dataset_name Para_10m --test_dir ./test")




for i in range(len(Schedule)):
    os.system(Schedule[i])
