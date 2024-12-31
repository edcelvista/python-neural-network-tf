#########################################################
# Author: Edcel Vista                                   #
#########################################################

import os, sys, json
import tensorflow as tf
from dotenv import load_dotenv
from utils import printFlush, print_progress_bar
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import enable_op_determinism
from sklearn.model_selection import train_test_split #  Sklearn libraries are used for machine learning operations
import pandas as pd # We need Pandas for data manipulation

# Load environment variables from .env file
load_dotenv()

# Define the model Parameters
output_dim    = int(os.getenv("output_dim"))
max_words     = int(os.getenv("max_words"))
trunc_type    = os.getenv("trunc_type")
oov_tok       = os.getenv("oov_tok")

# Some TensorFlow operations are inherently non-deterministic, especially when running on GPUs
enable_op_determinism()

def _predict(new_sentences):
    # Loading the model back
    model = load_model(f'{os.getenv("outputModelfile")}')

    # Ensure the model is in inference mode
    model.trainable = False  # For manual inference, to prevent training layers from being active

    # Model summary
    model.summary()

    # Load tokenizer from json
    with open(f'{os.getenv("outputTokenizerJsonfile")}', 'r') as json_file:
        tokenizer_json = json.load(json_file)

    tokenizer     = tokenizer_from_json(tokenizer_json)
    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    padded        = pad_sequences(new_sequences, maxlen=max_words)
    predictions   = model.predict(padded)
    print("")
    for i in range(0,len(new_sentences)):
        print('Review: ' + new_sentences[i] + ' | ' + 'Sentiment: ' + str("POSITIVE" if predictions[i] >= 0.5 else "NEGATIVE") + ' | ' + 'Sentiment Score: ' + str(predictions[i]) + '\n')

# Asking for user input
print("")
prompt = input("Enter Prompt: ")

# Displaying the collected information
print(f"Analyzing...")

if prompt:
    _predict([prompt])
else:
    print("No input received.")