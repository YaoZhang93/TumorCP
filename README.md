# TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation

## Paper

This is the implementation for the paper:

[TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation](https://arxiv.org/pdf/2107.09843.pdf)

Accepted by MICCAI 2021

![image](https://github.com/YaoZhang93/TumorCP/blob/main/figs/TumorCP.png)

## Usage

* Data Preparation

  - Download the data from [MICCAI 2019 KiTS Challenge](https://kits19.grand-challenge.org/).

  - Convert the files' name by

  `python dataset_conversion/Task040_KiTS.py`

  - Preprocess the data by

  `python experiment_planning/nnUNet_plan_and_preprocess.py -t 40 --verify_dataset_integrity`
  
  - Extract the tumor region in advance by

  `python extract_tumors.py`

* Configuration

  The default configuration of `TumorCP` is in `./configuration.py`. You can modify the parameters in the Trainer. Here are the examples: [nnUNetTrainerV2_ObjCPAllInter](https://github.com/YaoZhang93/TumorCP/blob/main/nnunet/training/network_training/nnUNetTrainerV2_ObjCPAllInter.py) and [nnUNetTrainerV2_ImgDAObjCPAllInter](https://github.com/YaoZhang93/TumorCP/blob/main/nnunet/training/network_training/nnUNetTrainerV2_ImgDAObjCPAllInter.py).

* Train

  Train the model by

  `python run/run_training.py 3d_fullres nnUNetTrainerV2_ImgDAObjCPAllInter 40 0`

 `TumorCP` is integrated with the out-of-box [nnUNet](https://github.com/MIC-DKFZ/nnUNet). Please refer to it for more details.

 * Test

  - inference on the test data by

  `python inference/predict_simple.py -i INPUT_PATH -o OUTPUT_PATH -t 40 -f 0 -tr nnUNetTrainerV2_ImgDAObjCPAllInter`

## Citation 

If you find this code and paper useful for your research, please kindly cite our paper.

```
@inproceedings{yang2021tumorcp,
  title={TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation},
  author={Yang, Jiawei and Zhang, Yao and Liang, Yuan and Zhang, Yang and He, Lei and He, Zhiqiang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={579--588},
  year={2021},
  organization={Springer}
}
```

## Acknowledgement

`TumorCP` is integrated with the out-of-box [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

