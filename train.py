#########################################################
# Author: Edcel Vista                                   #
#########################################################

import os, sys, json
import pandas as pd # We need Pandas for data manipulation
import numpy as np  # NumPy for mathematical calculations
import matplotlib.pyplot as plt # MatplotLib, and Seaborn for visualizations
from sklearn.model_selection import train_test_split #  Sklearn libraries are used for machine learning operations
from dotenv import load_dotenv
from utils import printFlush, print_progress_bar, cleanUp, ask_yes_or_no
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Dropout, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# Load environment variables from .env file
load_dotenv()

# Define the model Parameters
output_dim    = int(os.getenv("output_dim"))
max_words     = int(os.getenv("max_words"))
trunc_type    = os.getenv("trunc_type")
oov_tok       = os.getenv("oov_tok")
batch_size    = int(os.getenv("batch_size"))
epochs        = int(os.getenv("epochs"))

def gpuInit(index=0):
    # List all GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Set only the first GPU to be visible to TensorFlow
            tf.config.set_visible_devices(gpus[index], 'GPU')

            # Verify that only one GPU is being used
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            # Errors can occur if GPUs have already been initialized
            print(e)

def _trainRegression():
    print("")
    isNewModel = True
    if os.path.exists(os.getenv("outputModelfile")):
        if ask_yes_or_no("Do you want to use existing model?") == True:
            isNewModel = False
        else:
            cleanUp([os.getenv("outputModelfile"), os.getenv("outputTokenizerJsonfile"), os.getenv("outputModelCheckPointfile")])

    print("")
    gpuInit(0) # select first gpu

    print("")
    print("Available devices:")
    print(tf.config.list_physical_devices())

    # Example CSV file path (replace with your file path)
    csv_file = os.getenv("inputStagePath")
    # Read CSV file into a pandas DataFrame
    df       = pd.read_csv(csv_file)
    # Display the first few rows of the DataFrame

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment_encoded'], test_size = float(os.getenv("test_size_ratio")), random_state = 0)

    print("")
    print("Data Distribution:")
    print(f"Training Data: {X_train.size} | Label Training Data: {y_train.size}")
    print(f"Validation Training Data: {X_test.size} | Label Validation Data: {y_test.size}")

    # [ "I love TensorFlow", "TensorFlow is amazing", "Deep learning is powerful" ]
    # Tokenization in general is parsing the text and storing the tokens separated by a blank(here it is adequate for English, it must be wrong for another language). In TensorFlow, tokenization is the process that attributes to each word a number. For each word, the algorithm will give an identification. This is done by sorting the words by their occurrence in our training data.
    tokenizer = Tokenizer(num_words=max_words, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    word_index  = tokenizer.word_index # {'<OOV>': 1, 'is': 2, 'tensorflow': 3, 'i': 4, 'love': 5, 'amazing': 6, 'deep': 7, 'learning': 8, 'powerful': 9}
    vocab_size  = len(word_index) + 1

    # Sequencing In a sentence, we will replace each word with its number(remember the values in our dictionary) and store it as a list. For example: ‘I love Matrix’ will be stored as [3,5,9]. And we build a dictionary or a corpus with all the words encountered in the training data. When we see a new word, we replace it with ‘OOV’ — Out Of Vocabulary or ‘UNK’ — Unknown.
    sequences      = tokenizer.texts_to_sequences(X_train) # [[4, 5, 3], [3, 2, 6], [7, 8, 2, 9]]
    test_sequences = tokenizer.texts_to_sequences(X_test) # [[4, 5, 3], [3, 2, 6], [7, 8, 2, 9]]

    # Padding Our sentences don’t have the same length, So our matrix (list of lists of words indexes) will not have the same shape. Models don’t like that. So we need to normalize their length. And here comes the padding. We add zeroes at the end or at the beginning of our sequence to make it the same length as the longest sentence in our data.
    padded      = pad_sequences(sequences, maxlen=max_words, truncating=trunc_type) # [[4 5 3 0 0],[3 2 6 0 0],[7 8 2 9 0]]
    test_padded = pad_sequences(test_sequences, maxlen=max_words, truncating=trunc_type) # [[4 5 3 0 0],[3 2 6 0 0],[7 8 2 9 0]]

    if isNewModel == True:
        model = Sequential([
            Input(shape=(max_words,)), # input activation functions first layer input
            Embedding(vocab_size, output_dim, input_length=max_words), # Converts word indices into dense vectors of fixed size (16 dimensions).
            # Flatten(), remove flatten if LSTM is used.
            LSTM(units=32, return_sequences=True), # LSTM These layers capture patterns in the text over sequences of words.
            LSTM(units=16),
            Dense(units=8, # Reduces the dimensionality of the features.
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01), # Adds penalties to the loss function for large weights
                activation='relu' # relu (Rectified Linear Unit). ReLU is commonly used in hidden layers as it introduces non-linearity, helping the model learn complex patterns. It outputs the input directly if it’s positive or zero otherwise.
            ),
            Dropout(rate=0.5), # Randomly drops units during training to reduce overfitting
            Dense(units=1, activation='sigmoid') # sigmoid activation for bi`nary classification ( 0 | 1 ) | softmax for mulitple output
        ])
    else:
        # Load existing model
        base_model = load_model(os.getenv("outputModelfile"))

        # # This line sets the number of layers from the base model that will be fine-tuned during the training process.
        # fine_tune_at = -1

        # # Modify the model (optional)
        # for layer in base_model.layers[:fine_tune_at]:
        #     layer.trainable = False

        # Freeze the weights of the pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Create a new model on top of the pre-trained base model
        model = Sequential([
            base_model,
            Embedding(vocab_size, output_dim, input_length=max_words), # Converts word indices into dense vectors of fixed size (16 dimensions).
            # Flatten(), remove flatten if LSTM is used.
            LSTM(units=32, return_sequences=True), # LSTM These layers capture patterns in the text over sequences of words.
            LSTM(units=16),
            Dense(units=8, # Reduces the dimensionality of the features.
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01), # Adds penalties to the loss function for large weights
                activation='relu' # relu (Rectified Linear Unit). ReLU is commonly used in hidden layers as it introduces non-linearity, helping the model learn complex patterns. It outputs the input directly if it’s positive or zero otherwise.
            ),
            Dropout(rate=0.5), # Randomly drops units during training to reduce overfitting
            Dense(units=1, activation='sigmoid') # sigmoid activation for bi`nary classification ( 0 | 1 ) | softmax for mulitple output
        ])

    # Optimizer Choose Adam optimizer with a specific learning rate
    # Feature	         Adam	                            SGD
    # Learning Rate	     Adaptive per parameter	            Fixed (unless decay is used)
    # Convergence Speed	 Faster, works well out-of-the-box	Slower, needs fine-tuning
    # Momentum	         Implicitly included	            Requires explicit setting
    # Use Case	         General-purpose	                Works well with large datasets
    # learning_rate (float, default = 0.001): Higher values can lead to faster convergence, but also to instability if too large. Lower values can lead to slower convergence, but more stable updates
    # beta_1 (float, default = 0.9): Typical values: Close to 1 (usually around 0.9), meaning that recent gradients are weighted more heavily. If set to 1, it effectively means no decay, using only the current gradient.
    # beta_2 (float, default = 0.999): Typical values: Close to 1 (usually around 0.999), meaning that it uses long-term information about the gradient variance. If set to 1, it would ignore the squared gradients entirely.
    # epsilon A small value added to the denominator to prevent division by zero. This is useful in cases where the squared gradients are very small.
    optimizer = Adam(
        learning_rate=0.001, # adjust acoordingly when adjust to larget batch_size
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=0.0000001
    )

    # Compile the model
    model.compile(
        optimizer=optimizer, # The optimizer determines how the model is updated during training. Adam (short for Adaptive Moment Estimation) is an efficient optimization algorithm that combines the benefits of both Adagrad and RMSprop. It adjusts the learning rate based on the average of recent gradient updates and is widely used in deep learning tasks.
        loss='binary_crossentropy', # categorical_crossentropy’ for multiple classification | binary_crossentropy for binary)
        metrics=['accuracy'] # The accuracy metric is used to track how often the model’s predictions match the true labels. It will output the percentage of correct predictions.
    )

    # Model summary
    model.summary()

    # Stop training when the validation loss stops improving.
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor (can be 'val_accuracy', 'val_loss', etc.)
        patience=3,   # Number of epochs with no improvement to wait before stopping
        verbose=1,           # Print messages when early stopping is triggered
        restore_best_weights=True # Restore model weights from the best epoch
    )

    # Create a callback that saves the model's weights every 5 epochs
    model_checkpoint = ModelCheckpoint(
        filepath=os.getenv("outputModelCheckPointfile"), 
        verbose=1,
        save_best_only=True,
        save_freq=5*batch_size
    )

    print("")
    if ask_yes_or_no(f"Proceed {'with new model' if isNewModel else 'with existing model'}?") != True:
        print("Aborted.")
        sys.exit(0)
    
    # Train the model
    history = model.fit(
        padded, np.array(y_train), # Training data
        batch_size=batch_size, # The batch size refers to the number of samples that will be processed before the model’s internal parameters (weights) are updated. In other words, the training dataset is divided into batches, and the model updates its weights after each batch, rather than after the entire dataset.
        epochs=epochs, # An epoch refers to one complete pass through the entire training dataset during model training. After each epoch, the model updates its weights based on the gradients computed from the training data. The number of epochs determines how many times the model will see the entire training set.
        callbacks=[early_stopping, model_checkpoint], # Include the early stopping callback
        validation_data=(test_padded, y_test)
    )

    # Making Predictions
    _makePrediction(tokenizer, ["if you like original gut wrenching laughter you will like this movie if you are young or old then you will love this movie hell even my mom liked it great camp", "protocol is an implausible movie whose only saving grace is that it stars goldie hawn along with a good cast of supporting actors the story revolves around a ditzy cocktail waitress who becomes famous after inadvertently saving the life of an arab dignitary the story goes downhill halfway through the movie and goldies charm just doesnt save this movie unless you are a goldie hawn fan dont go out of your way to see this film"], model)

    # Save the model to a file in keras format
    model.save(f'{os.getenv("outputModelfile")}')

    # Save tokenizer configuration to json
    tokenizer_json = tokenizer.to_json()
    with open(f'{os.getenv("outputTokenizerJsonfile")}', 'w') as json_file:
        json.dump(tokenizer_json, json_file)

    # Visualizing Model Performance
    _visualize(history)

def _visualize(history):
    # Visualizing the training history
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    # Plot training & validation accuracy values
    # Underfitting: The model performs poorly on both training and validation datasets, meaning it hasn’t learned the underlying patterns in the data
    # Overfitting: The model performs well on the training dataset but poorly on the validation dataset, meaning it has learned patterns specific to the training data, including noise

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting accuracy on the first y-axis (left)
    ax1.plot(epochs, acc, 'bo', label='Training accuracy')
    ax1.plot(epochs, val_acc, 'b', label='Validation accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Training and Validation Accuracy & Loss')

    # Create a second y-axis for loss
    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis

    # Plotting loss on the second y-axis (right)
    ax2.plot(epochs, loss, 'ro', label='Training loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot as an image
    plt.savefig(os.getenv('inputStageStatsTrainingPath'), dpi=300, bbox_inches='tight')  # Save with 300 DPI and no extra whitespace
    plt.close()  # Close the current plot to free up memory
    
def _makePrediction(tokenizer, new_sentences, model):
    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    padded        = pad_sequences(new_sequences, maxlen=max_words)
    predictions   = model.predict(padded)
    print("")
    for i in range(0,len(new_sentences)):
        print('Review: ' + new_sentences[i] + ' | ' + 'Sentiment Score: ' + str(predictions[i]) + '\n')

_trainRegression()