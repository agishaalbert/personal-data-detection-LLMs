import os
import shutil
import torch
import time
from torch.optim.lr_scheduler import OneCycleLR
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from extended_flair.extended_text_classifier import TextClassifier
# from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from config.run_config import path_params, model_params

import random
import numpy as np
from transformers import set_seed as set_transformer_seed

# Set random seed for all Python libraries involved
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_transformer_seed(SEED)

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

# 2. Create the label dictionary
LABEL_TYPE = 'class'
label_dict = corpus.make_label_dictionary(LABEL_TYPE)

# 3. Initialize transformer document embeddings
document_embeddings = TransformerDocumentEmbeddings(
    model='distilbert-base-uncased', fine_tune=True, fp=8
)

# 4. Create the text classifier
classifier = TextClassifier(
    document_embeddings=document_embeddings,
    label_type=LABEL_TYPE,
    label_dictionary=label_dict,
    multi_label=True,
    multi_label_threshold=0.1,
    max_token=500,
    max_sentence_parts=4,
    default_delimiter='.'
)

# 5. Initialize trainer with AdamW optimizer
trainer = ModelTrainer(
    classifier,
    corpus,
    optimizer=torch.optim.AdamW
)

# 6. Track training time
start_time = time.time()  # Start time
print("Training started...")

# 7. Run training with fine-tuning
trainer.train(
    base_path=MODEL_PATH,
    learning_rate=model_params["learning_rate"],
    mini_batch_size=model_params["mini_batch_size"],
    max_epochs=model_params["max_epochs"],
    scheduler=OneCycleLR,
    embeddings_storage_mode=model_params["embeddings_storage_mode"],
    weight_decay=model_params["weight_decay"],
    checkpoint=True
)

end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time

# Print training time details
print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

# 8. Evaluate and print classification report
document_embeddings.training = False
result = classifier.evaluate(
    corpus.test,
    gold_label_type='class',
    mini_batch_size=model_params["mini_batch_size"],
)

#print(result)

print("Evaluation complete.")

