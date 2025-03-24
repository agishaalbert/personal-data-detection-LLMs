# Automated Detection of Personal Data Using Large Language Models (LLMs)

**Authors**: Albert Agisha N., Luca Rück, Martin Heckmann

---

## Most Important Files of This Work

- **`dessi-mf` folder in `datasets`**:
  Contains all files and notebooks related to the creation of the DeSSI-MF dataset.

- **`gpt_predictions.ipynb` in `GPT`**:
  Code to use the GPT-4o model to make predictions on all datasets.

- **All notebooks in `comparison`**:
  Contain the final results of the experiments comparing all three models on all datasets.

- **`CASSED_model_results` folder in `CASSED`**:
  Contains all files of CASSED's results. The predictions were made in another coding project (`CASSED-main`).

- **`presidio_predictions` folder in `Presidio`**:
  Contains all files and notebooks of predictions using Presidio on all datasets.

---

## Conda Environments

To run the provided code, two different Conda environments are required.

### 1. Environment: `data-analysis` (Python 3.9.20)
Creation can take a few minutes.

```bash
conda create --name data-analysis python=3.9.20
conda activate data-analysis
pip install ipykernel notebook pandas numpy plotly scikit-learn openml pyarrow openai presidio-analyzer presidio-structured kaggle langdetect openpyxl spacy transformers flair piicatcher pii-codex
```

This environment can be used for every code file, which does not import the Mimesis and Faker libraries.

### 2. Environment: `data-generation` (Python 3.10.15)

```bash
conda create --name data-generation python=3.10.15
conda activate data-generation
pip install ipykernel notebook pandas scikit-learn faker mimesis
```

As Mimesis requires Python 3.10, this additional environment is required. The environment is used for the following code files, which include the Mimesis or Faker library:
- `datasets/test_languages/mimesis_faker_new_language.ipynb`
- `datasets/dessi-mf/faker/unique_faker.ipynb`
- `datasets/dessi-mf/mimesis/unique_mimesis.ipynb`
- `datasets/Simon_Faker/simon_faker.ipynb`

---

## Folder Structure

### 1. `CASSED`
Contains all Jupyter notebooks and CSV files to evaluate the predictions of the CASSED model.

#### Subfolders and Files:

- **`CASSED_model_results`**:
  - Folder containing all files of CASSED’s results in multi-class classification, binary classification for personal/non-personal, and binary classification for PII/non-PII

- **`evaluation`**:
  - Notebooks to evaluate the performance of CASSED on all datasets in multi-class classification, binary classification for personal/non-personal, and binary classification for PII/non-PII

- **`old_predictions`**:
  - Notebooks and files containing the predictions of CASSED on the original DeSSI dataset, these predictions were used to analyze the original DeSSI dataset

---

### 2. `comparison`
Notebooks to compare all results of the three models in this study for:
- Multi-class classification
- Personal binary classification
- PII binary classification (results for PII binary classification are not complete, as only the results for personal binary classification and mulit-class classification were used for this project)

---

### 3. `datasets`
Contains all Jupyter notebooks and CSV files related to datasets analyzed in this work.

#### Subfolders and Files:

- **`dessi`**:
  - `dessi_cleaned`: CSV files of the cleaned DeSSI dataset
  - `DeSSI_v2`: Original DeSSI dataset
  - `analyze_dessi_data.ipynb`: Analysis of DeSSI data
  - `find_label_errors 1-4`: Notebooks to clean DeSSI, the notebooks must be executed in the order of the numbers

- **`dessi-mf`**:
  - `check_uniqueness`: Contains Notebooks to check if specific semantic classes of Mimesis, Faker, and DeSSI only contain unique values (requirement for the creation of DeSSI-MF)
  - `dessi_unique`: Files and code of the cleaned DeSSI data with unique values used for the DeSSI-MF dataset
  - `dessi-mf`: Files and code of the DeSSI-MF dataset with labels and classes
  - `dessi-mf_gpt`: Files and code of the reduced DeSSI-MF dataset for evaluating GPT
  - `faker`: Files and code of Faker data used for the DeSSI-MF dataset
  - `mimesis`: Files and code of Mimesis data used for the DeSSI-MF dataset


- **`freiburg-medical`**:
  - Files of the medical dataset and notebook to generate it

- **`kaggle_datasets`**:
  - Files of Kaggle datasets and notebook to create and label the Kaggle dataset.
  The connection with kaggle api requires a token, information about creating the token are available on https://www.kaggle.com/docs/api.
  Viewing the `all_datasets.csv` can lead to confusion, the dataset contains a spam email dataset with vulgar and strange descriptions

- **`openml_datasets`**:
  - Files of OpenML 1 datasets and notebook to create and label OpenML 1 dataset

- **`openml_datasets_2`**:
  - Files of OpenML 2 datasets and notebook to create and label the OpenML 2 dataset

- **`Sherlock-VizNet`**:
  - Files of VizNet data used for SHERLOCK and notebook to analyze the dataset

- **`Simon_Faker`**:
  - Files of Faker data used for SIMON and notebook to analyze the dataset

- **`test_languages`**: 
  - Files of the Files of the Test Languages dataset and notebook to create the dataset

- **`WikiTablesCollection`**:
  - Contains WikiTables dataset of SeLaB study and a notebook for analyzing the dataset

---

### 4. `GPT`
Contains all Jupyter notebooks and files related to the GPT predictions in this study.

#### Subfolders and Files:

- **`gpt_predictions`**:
  - TXT files of GPT predictions on all datasets

- **`mapping_results`**:
  - GPT’s mapping of all 50 semantic classes from DeSSI-MF to personal/non-personal and PII/non-PII

- **Evaluation Notebooks**:
  - `evaluation_multiclass.ipynb`: Evaluation of GPT’s prediction in multi-class classification
  - `evaluation_personal.ipynb`: Evaluation of GPT’s prediction in binary classification of personal/non-personal
  - `evaluation_pii.ipynb`: Evaluation of GPT’s prediction in binary classification of PII/non-PII. The results for pii binary classification are not complete (medical dataset missing), as it was decided to only use the results of the personal binary classification for this project

- **Other Notebooks**:
  - `gpt_mapping.ipynb`: Notebook to map all 50 semantic classes of DeSSI-MF to personal/non-personal and PII/non-PII
  - `gpt_predictions.ipynb`: Notebook to make predictions with GPT on every dataset

- **`OpenAIAPIKey.txt`**:
  - File required to store the OpenAI API Key (has to be added independently to this project).

---

### 5. `Presidio`
Contains all notebooks and files related to personal data detection models, which were used in this work.

#### Subfolders and Files:

- **`evaluation`**:
  - Notebooks to evaluate the performance of Presidio on all datasets in multi-class classification, binary classification for personal/non-personal, and binary classification for PII/non-PII

- **`old_predictions`**:
  - Notebooks and files containing the predictions of Microsoft Presidio on the original DeSSI dataset, these predictions were used to clean the DeSSI dataset

- **`predictions`**:
  - Notebooks and files containing the predictions on all datasets using Microsoft Presidio

---

### 6. `testing_libraries`:
Notebooks testing some personal data detection libraries that were not considered for the experiments in this work
