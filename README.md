# Structure
```
├───data {Folder to put datasets in. Refere to [Get the Data] section}.
│   ├───dataset_1
│   ├───dataset_2
│   └───etc...
│
└───scripts {Main folder for python scripts}.
    ├───cut_with_pillow.py {Script to cut triplets from a large image}.
    ├───top_model.py {Script to prepare and train top triplets model on VGG16 model's base}.
    └───fine_tune.py {Script to finetune final model}.
```
# Setup
- install miniconda from https://conda.io/miniconda.html
- setup new enviroment e.g.:
```
conda create --name similarity
source activate similarity
```
- install keras, tensorflow and other packages as needed e.g.:
```
conda install tensorflow-gpu
```
- check installation:
```
conda list
```
# Get the Data
- our datasets can be downloaded from https://drive.google.com/drive/folders/1B1_SH8q7xjJ-KzNSGlWiVGdPKkXa0y5s
# Prepare And Train the Top Model on VGG16 Base
- run:
```
python top_model.py 
```
- *.npy files will be generated with features extracted by passing data through VGG16 bottom layers
- top_model_weights.h5 file with top model weights will be generated after training new top layers
- take a note at the last two lines of the script, as these actions can be separated
# Fine Tune the Model
- run:
```
python fine_tune.py
```
- should take some time to fine tune the model
