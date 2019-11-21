set -ex

# 3D U-Net
# G:unet D:pixel [DONE]
python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_unet_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG unet_3d_cust --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d

# G:unet D:patch
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_unet_patch --checkpoints_dir ../checkpoints --model pix2pix --netG unet_3d_cust --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d


# 3D Link-Net
# G:linknet D:pixel [DONE]
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_linknet_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG linknet_3d --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d

# G:linknet D:patch [DONE]
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_linknet_patch --checkpoints_dir ../checkpoints --model pix2pix --netG linknet_3d --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d


# 3D V-Net
# G:vnet D:pixel [DONE]
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_vnet_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG vnet --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d

# G:vnet D:patch [DONE]
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_vnet_patch --checkpoints_dir ../checkpoints --model pix2pix --netG vnet --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d


# 3D Student
# G:student D:pixel [DONE]
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_student_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG student --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d --epoch 100

# G:student D:patch [DONE[
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_student_patch --checkpoints_dir ../checkpoints --model pix2pix --netG student --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d --epoch 100


# 3d deeper student
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_deeper_student_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG deeper_student --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d --epoch 100
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_deeper_student_patch --checkpoints_dir ../checkpoints --model pix2pix --netG deeper_student --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d --epoch 100


# 3D espnet
# G:esp D:pixel [DONE]
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_esp_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG espnet --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d

# G:esp D:patch [DONE]
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_esp_patch --checkpoints_dir ../checkpoints --model pix2pix --netG espnet --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d


# 3d mrf
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_mrf_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG unet_mrf --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_mrf_patch --checkpoints_dir ../checkpoints --model pix2pix --netG unet_mrf --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d


# 3d mrf v2
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_mrf_v2_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG unet_mrf_v2 --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_mrf_v2_patch --checkpoints_dir ../checkpoints --model pix2pix --netG unet_mrf_v2 --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d






[Extra dataset]
# test extra
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_extra --checkpoints_dir ../checkpoints --model pix2pix --netG moreinception_student_v2 --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d

# Taiwan
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_tokyo --checkpoints_dir ../checkpoints --model pix2pix --netG moreinception_student_v2 --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
#python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_tokyo_vnet --checkpoints_dir ../checkpoints --model pix2pix --netG vnet --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
#python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_tokyo_esp --checkpoints_dir ../checkpoints --model pix2pix --netG espnet --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
#python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_tokyo_link --checkpoints_dir ../checkpoints --model pix2pix --netG linknet_3d  --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
#python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_taiwan_student_deeper --checkpoints_dir ../checkpoints --model pix2pix --netG deeper_student --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_tokyo_student --checkpoints_dir ../checkpoints --model pix2pix --netG student --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d

# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name more_inception_tokyo_test --checkpoints_dir ../checkpoints --model pix2pix --netG moreinception_student_v2 --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d
