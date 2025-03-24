# CASSED: Personal Data Detection

This repository implements the **CASSED model** for detecting personal data in structured datasets. The repository is based on the original project by [VKuzina](https://github.com/VKuzina/CASSED) and has been adjusted for the tasks in this study.

---

## Requirements
The original project specifies the following requirements:

```text
torch==1.9.0+cu111
pandas
numpy
flair==0.9
```

Directly installing these requirements may lead to errors. The process of achieving a valid installation is described in `installation.txt`. Below is a streamlined way to install the required libraries without encountering errors.

---

## Installation

To set up the environment, follow these steps:

```bash
# Create a new Conda environment
conda create --name cassed python=3.8
conda activate cassed

# Install PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies from requirements.txt
pip install -r requirements.txt

#Update PyTorch
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## CASSED Overview

The CASSED project is desribed on [Github](https://github.com/VKuzina/CASSED) with the following text:

**CASSED**

Cassed is a model for the detection of sensitive data in structured datasets, more specificly, for the multilabel problem of columns in database tables.

The model uses the BERT model#, through the Flair library#, and has an accompanying dataset on kaggle called DeSSI (Dataset for Structured Sensitive Information)#.

To learn more about the model please refer to the full paper ##.

**Setup**
All of the setup for CASSED is made in configs/run_config.py, where inside of the path_params dictionary, all of the paths need to be set.

**Datasets**
Several datasets are present in the repository and can be found in ##, set the path to the dataset in the run_config.py. If a different dataset is desired, for CASSED to work on it, the dataset needs to be in a specific format. To turn the standard format of .csv data and labels, readable into a pandas Dataframe, into CASSED-s required format, you can simply set the parameter "standard_data_path" inside of the path_parameters to the folder with the standardised data, and run prepare_data.py.

**Use trained models**
A wide variets of models are pretrained on different datasets, and are available in the repository under ##. Set the path parameters in the run_config.py file to the desired path, and run test_model.py.

**Train your own model**
To train your own model, set the path to the dataset in the path_parameters inside the run_config.py file, as well as the model_parameters and the processing_parameters to your desired values, prepare the data, as described before in the Datasets section, and run train_classifier.py



---

## Project Extensions

### Extended Features
- The scripts `test_model.py` and `extended_text_classifier.py` have been extended to output and save results in the `test_results` folder when testing a trained model.

### Datasets
- The datasets used in this study are stored in the `datasets` directory.

### Models
- The `models` folder is generated when training a model. It is excluded from this repository as trained models consume significant storage space.

---
