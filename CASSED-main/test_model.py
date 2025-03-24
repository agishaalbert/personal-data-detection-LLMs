import os
import shutil
import torch
import time
import json
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from extended_flair.extended_text_classifier import TextClassifier
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

from sklearn.metrics import recall_score
# from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from config.run_config import path_params, model_params

# Path preparation
PATH_PREFIX = path_params['path']
DATA_FOLDER = os.path.join(PATH_PREFIX, path_params['data_path'])

MODEL_PATH = os.path.join(PATH_PREFIX, path_params['model_path'], path_params['model_name'])

if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

CACHE_DIR = os.path.join(MODEL_PATH, path_params['cache_dir']) if 'cache_dir' in path_params else None

if os.path.isdir(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR)

# 1. Corpus preparation
corpus: Corpus = ClassificationCorpus(
    DATA_FOLDER,
    test_file='test.txt',
    dev_file='dev.txt',
    train_file='train.txt'
)

# Load the trained model
classifier = TextClassifier.load(MODEL_PATH + '/best-model.pt')

# Start timer for evaluation
test_start_time = time.time()
print("Testing started...")

# Evaluate the model
result, y_true, y_pred, target_names, labels = classifier.evaluate(
    corpus.test,
    gold_label_type='class',
    mini_batch_size=model_params["mini_batch_size"],
    return_all=True
)

# End timer for evaluation
test_end_time = time.time()
test_elapsed_time = test_end_time - test_start_time

print(result)

# Assuming y_true and y_pred are your multilabel matrices
y_true_single = np.argmax(y_true, axis=1)
y_pred_single = np.argmax(y_pred, axis=1)

Balanced_accuracy = balanced_accuracy_score(y_true_single, y_pred_single)
print(f'Balanced Accuracy: {Balanced_accuracy}') 
# Print test time details

print(f"Testing completed in {test_elapsed_time:.2f} seconds ({test_elapsed_time / 60:.2f} minutes)")
cm = confusion_matrix(y_true_single, y_pred_single)
print(f'confusion matrix:\n {cm}')

# Convert the confusion matrix to a DataFrame
cm_df = pd.DataFrame(cm, index=['Non-Personal', 'Personal'], columns=['Non-Personal', 'Personal'])

# Print the formatted confusion matrix
print("Confusion Matrix:")
print(cm_df)


# Save results
os.makedirs("test_results", exist_ok=True)
pd.DataFrame(y_pred).to_csv("test_results/predicted.csv", index=False)
pd.DataFrame(y_true).to_csv("test_results/true.csv", index=False)
targets = pd.DataFrame(target_names)
labels = pd.DataFrame(labels)
pd.concat([targets, labels], axis=1).to_csv("test_results/target_names.csv", index=False)

print("Test results saved.")
