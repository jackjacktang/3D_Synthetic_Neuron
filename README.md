# 3D_Synthetic_Neuron

**3D_Synthetic_Neuron:  [Project](https://github.com/jackjacktang/3D_Synthetic_Neuron) |  Paper(TBC) **

### Usage

### 1. Dependencies Requirement
The following packages are required to be installed before running the model:

* `opencv_python==3.4.2.17`
* `torch==1.1.0`
* `tifffile==2019.7.26`
* `tqdm==4.32.2`
* `scikit_image==0.15.0`
* `scipy==1.0.0`
* `requests==2.9.1`
* `torchvision==0.3.0`
* `numpy==1.17.0`
* `Pillow==6.2.1`
* `beautifulsoup4==4.8.1`
* `dominate==2.4.0`
* `skimage==0.0`
* `visdom==0.1.8.9`

### 2. Dataset Organization
```
dataset
  => sub-dataset1
    => [train/test]A (TIFF neuron skeleton file converted from swc file using distance transform preprocessing)
    => [train/test]B (original optical images eg. TIFF file)
```

### 3. Train Model
```
**To train the model :
python train.py --dataroot [path_to_dataset] --name [experiment_name] --netG [generator] 
--netD [discriminator] --results_dir [path_to_store_cps]

[netG] generator options: 'unet, linknet, vnet, esp, MRF'
[netD] discriminator options: 'basic_3d(PatchGAN), pixel_3d'
[crop_size_3d] 3d patch size for training, split each dimension by 'x', e.g. 128x128x32
[results_dir] path to store cps

For more details please refer to train_3d_neuron.sh in 'scripts' folder
```

### 4. Test Model
```
python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_unet_pixel --checkpoints_dir ../checkpoints 
--model pix2pix --netG [generator] --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 
--crop_size_3d [3d_patch_size] --norm batch --results_dir [result_directory_to_store --save_type 3d

[netG] generator options: 'unet, linknet, vnet, esp, MRF', make sure it matches the checkpoints loaded
[crop_size_3d] 3d patch size for tesinging, split each dimension by 'x', e.g. 128x128x32

For more details please refer to test_3d_neuron.sh in 'scripts' folder
```

### 5. Optional Parameters
```
[checkpoints_dir] the directory to save the models: './checkpoints' (default)
[input_nc/output_nc] input channel number: 1(default for greyscale), 3
[crop_size_3d] 3d batch size: 128x128x32(default, split each dimension using 'x')
[save_type] img save mode: 3d(default), 2d 



```