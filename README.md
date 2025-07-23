# Complementary-pathway Spatial-enhanced Visual Odometry 
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
## Training
You can refer to https://github.com/castacks/tartanair_tools for the TartanAir dataset. 

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

To train . Model will be run on the validation split every 10k iterations
```
python train.py --steps=240000 --lr=0.00008 --name=<your name>
```
Example training logs can be seen at ```examples```

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, TUM-RGBD and ICL-NUIM. Up to date result logs on these datasets can be found in the `logs` directory.

### TartanAir:
Results on the validation split and test set can be obtained with the command:
```
python evaluate_tartan.py --trials=5 --split=validation --plot --save_trajectory
```

### TartanAirAug (Augmented data of TartanAir):
```
python evaluate_tartan_augmented.py --trials=5 --plot --save_trajectory
```

### Tianmouc-VO:
```
python evaluate_Tianmouc.py --trials=5 --plot --save_trajectory
```

### Apollo:
```
python evaluate_apollo.py --trials=5 --plot --save_trajectory
```



## Acknowledgements
* This repository is built on [DPVO](https://github.com/princeton-vl/DPVO).
