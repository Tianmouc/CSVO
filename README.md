# Complementary-pathway Spatial-enhanced Visual Odometry 


https://github.com/user-attachments/assets/d306e179-e878-405f-a035-ae09c2b3ef62


This repository contains the source code for our paper:
 
```
@article{Lin2025complementary,
  title={Complementary-pathway Spatial-enhanced Visual Odometry for Extreme Environments with Brain-inspired Vision Sensors},
  author={Yihan, Lin and Zhaoxi, Zhang and Taoyi, Wang and Yuguo, Chen and Rong, Zhao},
  journal={International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```


## Setup and Installation

Clone the repo
```
git clone https://github.com/JesseZZZZZ/CSVO.git
cd CSVO
```
Create and activate the csvo anaconda environment
```
conda env create -f environment.yml
conda install cuda -c nvidia/label/cuda-11.3.1
conda activate csvo
```

Next install the packages
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install CSVO
pip install .

# download models and data (~2GB)
./download_models_and_data.sh
```
## Dataset Preprocessing
### TartanAir
You can refer to https://github.com/castacks/tartanair_tools for the TartanAir dataset. We suggest you to arrange the dataset as follows:

```Shell
├── datasets
    ├── TartanAir.pickle
    ├── TartanAir
        ├── abandonedfactory
        ├── abandonedfactory_night
        ├── ...
        ├── westerndesert
    ...
```

### TartanAirAug
We provide simple augmentation methods for TartanAir. You can run the following command to generate the augmented dataset:
```Shell
python augmentation/augmentation.py /path/to/TartanAir
```

### Tianmouc data
We currently only provide the Tianmouc-VO dataset, which is already preprocessed. You can download it from [here](https://drive.google.com/file/d/1Z1y5h6n3Z2Y3J3XJrX8Z2Y3J3XJrX8/view?usp=sharing). If you want to record some Tianmouc data with your own Tianmouc camera, you can refer to [here](https://github.com/Tianmouc/tianmoucv_preview), where we provide the decoding and alignment code for Tianmouc camera.
## Training


To train the model, you can run the following command:
```
python train.py --steps=240000 --lr=0.00008 --name=[dataset name]  --training_type  [input mode]
```
        
The training process requires about 5 days on a single NVIDIA RTX 3090 GPU.
Example training logs can be seen at ```examples```

## Evaluation
We provide evaluation scripts for TartanAir, TianmoucVO, and Apollo. Up to date result logs on these datasets can be found in the `logs` directory.

**you can use sh scripts under ./scripts flder for convenience**

### TartanAir:
Results on the validation split and test set can be obtained with the command:
```
python evaluate_tartan.py --trials=5 --split=validation --plot --save_trajectory
```

### TartanAirAug (Augmented data of TartanAir):
To run CSVO and our baseline (DPVO), you can directly run these scripts:
```
bash scrips/evaluate_tartan_augmented_csvo.sh
bash scrips/evaluate_tartan_augmented_dpvo.sh
```

### Tianmouc-VO:
```
To run CSVO and our baseline (DPVO), you can directly run these scripts:
bash scrips/evaluate_Tianmouc_csvo.sh
bash scrips/evaluate_Tianmouc_dpvo.sh
```

### Apollo:
```
python evaluate_apollo.py --trials=5 --plot --save_trajectory
```

## tmdat data inference support 

please install tianmoucv first:
```
pip install tianmoucv
```

then try our demo:

```
bash ./scripts/run_demo.sh
```


## Acknowledgements
* This repository is built on [DPVO](https://github.com/princeton-vl/DPVO).
* If you have problems on installation, you can refer to [DPVO's Issues](https://github.com/princeton-vl/DPVO/issues), which has a similar conda environment as ours.
