## Machine Learning Project

### Introduction
This repository is a simple project for the 2nd semester 2024 Machine Learning lecture.

### Prepare Dataset
```
-|MLProject
    -|dataset
        -|stanford_products
            -|bicycle_final
                -|11085122871_0.JPG
                -|11085122871_1.JPG
                -|11085122871_2.JPG
                ...
            -|cabinet_final
            -|chair_final
            ...
            -|bicycle_final.txt
            -|cabinet_final.txt
            -|chair_final.txt
            ...
    ...
```
### Split Dataset
```bash
./cd MLProject
python ./utils/train_val_split.py
```

### Environments
```bash
conda env create -n metric
conda activate metric 
pip install -r ./requirements.txt
```

### Usage
```bash
bash ./train_scripts/best_setting.sh        # you can change setting.  e.g., loss, sampling method and so on
```
