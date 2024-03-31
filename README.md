
# Medical Image Foundation Model Plus
## Overview
This repository offers a pre-trained vision transformer autoencoder (ViTAutoEnc), originally trained on 57,000 brain-enhanced MRI scans. This model was purposed for multi-contrast brain MRIs. The figure below gives an illustration of the training process:
![workflow](workflow.png)
Two mask schemes get utilized: mask block size 16*16*16 with 86 blocks (A); mask block size 4*4*4 with 6000 blocks (B)
![dropout_scheme](dropout_scheme.png)

## Usage Guide

### Dependencies
Dependencies can be installed via the following command:
```bash
pip install -r requirements.txt
```

### Pretrained models
We provide the pre-trained weights in [SSL_ViT_Block16](https://drive.google.com/file/d/1x1VI-0AoMqQZYVcbNoTQxe5ac-t3Ia5R/view?usp=drive_link) and [SSL_ViT-Block4](https://drive.google.com/file/d/1ttHL3IeZwuhjLPKS6SeLYjRQW-p6dD1U/view?usp=drive_link). Download them and transfer them into the **Pretrained_models** directory.

### Preparing Data
Kindly follow the five steps below:

1. Convert DICOM to NIFTI. We suggest using [dcm2niix](https://github.com/rordenlab/dcm2niix).
2. Strip skull. Select T1 weight or enhanced T1w volume with higher resolution for skull stripping. [HD-BET](https://github.com/MIC-DKFZ/HD-BET) is a solid choice.
3. Remove redundant blank area using ExtractRegionFromImageByMask from [ANTs](https://github.com/ANTsX/ANTs).
4. For co-registration of other contrasts (eg, T2w and FLAIR) with the skull-stripped T1w or T1c, please multiply with the brain mask generated in the previous step.
5. Merge the co-registered volumes and the skull-stripped volumes into a 4D volume, following the order T1w, T1c, T2w, and FLAIR.

### Fine-tuning weights for downstream task
Kindly follow the steps below: 

1. Modify ```train_files``` and ```val_files``` in finetune_train.py to accurately point out pre-processed train and validation images.
2. Modify ```test_files``` in finetune_test.py to indicate the preprocessed test images, and evaluate the model using finetune_test.py.
3. (Optional) Adjust Model.py to test various classifier configurations.

## Pre-training with personal data
We highly encourage the use of [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) when performing pre-training with your own data using the NVIDIA Container on a server equipped with multiple GPUs. Kindly follow the steps below: 

1. Pull the MONAI docker image using the command `docker pull projectmonai/monai`
2. Update the PATHs in run_docker.sh appropriately, and execute ```run_docker.sh```
3. While still in the docker container, execute ```run_pretraining.sh```. Your model is then ready for use!