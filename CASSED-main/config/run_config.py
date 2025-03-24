# Dessi-MF
path_params = {
    "path": './',
    "cache_dir": 'huggingface',
    "standard_data_path": 'datasets/all_data_personal_standard/',
    "data_path": 'datasets/all_data_personal_prepared/',
    "results_path": "results",
    "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
    "model_path": 'models',
    "model_name": "capped_10_16",
}


# test on kaggle 
# path_params = {
#     "path": './',
#     "cache_dir": 'huggingface',
#     "standard_data_path": 'datasets/kaggle_personal_standard/',
#     "data_path": 'datasets/kaggle_personal_prepared/',
#     "results_path": "results",
#     "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
#     "model_path": 'models',
#     "model_name": "capped_10_16",
# }

#test on openML1

# path_params = {
#     "path": './',
#     "cache_dir": 'huggingface',
#     "standard_data_path": 'datasets/openml_personal_standard/',
#     "data_path": 'datasets/openml_personal_prepared/',
#     "results_path": "results",
#     "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
#     "model_path": 'models',
#     "model_name": "capped_10_16",
# }

#test on openML2

# path_params = {
#     "path": './',
#     "cache_dir": 'huggingface',
#     "standard_data_path": 'datasets/openml_2_personal_standard/',
#     "data_path": 'datasets/openml_2_personal_prepared/',
#     "results_path": "results",
#     "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
#     "model_path": 'models',
#     "model_name": "capped_10_16",
# }

#test on  test language
# path_params = {
#     "path": './',
#     "cache_dir": 'huggingface',
#     "standard_data_path": 'datasets/test_languages_personal_standard/',
#     "data_path": 'datasets/test_languages_personal_prepared/',
#     "results_path": "results",
#     "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
#     "model_path": 'models',
#     "model_name": "capped_10_16",
# }

# MIMIC final DATASET
# path_params = {
#     "path": './',
#     "cache_dir": 'huggingface',
#     "standard_data_path": 'datasets/MIMICfinal_personal_standard/',
#     "data_path": 'datasets/MIMICfinal_personal_prepared/',
#     "results_path": "results",
#     "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
#     "model_path": 'models',
#     "model_name": "capped_10_16",
# }


#openML all DATASET
# path_params = {
#     "path": './',
#     "cache_dir": 'huggingface',
#     "standard_data_path": 'datasets/OpenMLall_personal_standard/',
#     "data_path": 'datasets/OpenMLall_personal_prepared/',
#     "results_path": "results",
#     "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
#     "model_path": 'models',
#     "model_name": "capped_10_16",
# }


# TRAINING AND EVAL PARAMS
model_params = {
    "learning_rate": 5.0e-5,
    "mini_batch_size": 16,
    "max_epochs": 20, #10
    "embeddings_storage_mode": 'none',
    "weight_decay": 0.

}

processing_params = {
    "column_name_separator": '. ',
    "sample_separator": ", ",
    "sentence_end": "."
}
