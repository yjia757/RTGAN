# SRGAN 
A PyTorch implementation of SRGAN based on CVPR 2017 paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802). 

## Requirments 
- PyTorch 

## Datasets 

### Train Dataset 
The train dataset is sampled from [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/) and [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). The total number of images in train dataset is 17,600. 

### Validation Dataset 
The validation dataset is sampled from [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/). It has 425 images. 

### Test Dataset 
The test dataset are sampled from | **Set 5** |  [Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)  
| **Set 14** |  [Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
| **BSD 100** | [Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
| **Urban 100** | [Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr). 

## Usage 

### Train 
- SRResNet-MSE training 
```
python3.6 train_srresnet.py --mse_or_vgg mse
```
- SRResNet-VGG training 
```
python3.6 train_srresnet.py --mse_or_vgg vgg
```
- SRGAN-MSE training 
```
python3.6 train_srgan.py --mse_or_vgg mse 
```
-SRGAN-VGG training 
```
python3.6 train_srgan.py --mse_or_vgg vgg
```

The output validation super resolution images are on `training_results` directory, or you can view via tensorboard. 

### Test 
```
python3.6 test_benchmark.py <checkpoint location>
```
The output super resolution iamges are on `benchmark_results` directory. 

## Notes 
### Updates from last version
- add SRResNet-MSE in `train_srgan.py` and create `train_srresnet.py` for training SRResNet-VGG  
- correct the tensorboard x-axis to number of update iterations 
- modified loss functions for SRResNet and SRGAN training 
- modified learning rate for different update iterations 
- calcualte PSNR and SSIM on Y-channel using Scikit-image 
- modified generator model with 16 residual blocks 
### Suggestions for training 
- please double check the number of epochs when train SRGAN variant using SRResNet-MSE pretrained model. The total number of epoch for SRGAN variant training should be the sum of used SRResNet-MSE pretrained epochs and 182 epochs, where 182 eopchs is approximately 20^5 updated iteration mentioned in the original paper. 
- keep in mind that `train_srresnet.py` only train netG, which implies there's no discriminator participate and thus there's no adversarial loss. 
