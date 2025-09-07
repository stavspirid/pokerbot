"""
This module includes functions responsible for building, training, and using a card recognition model
to classify playing card ranks (Jack, Queen, King, Ace). It leverages a convolutional neural network (CNN) 
to process card images and recognize their rank.

Dependencies:
- data_sets module: Provides functions for loading training and testing data.
"""

from data_sets import *

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import saving
from datetime import datetime


def build_model():
    """
    Prepare the model.

    Returns
    -------
    model : model class from any toolbox you choose to use.
        Model definition (untrained).
    """

    model = Sequential()

    #first conv layer w/relu + maxpooling
    model.add(Conv2D(32, (3, 3), activation='relu',strides=1, padding='same',input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #second conv layer w/relu + maxpooling, bigger filter
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #third conv layer w/relu + maxpooling, bigger filter
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #flatten output before fully connected
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))

    #dropout for regularization after fully connected
    model.add(Dropout(0.5))

    #output Layer - 4 classes (J, Q, K, A)
    model.add(Dense(4, activation='softmax'))


    return model

def train_model(model, n_validation, write_to_file=False):
    """
    Fit the model on the training data set.

    Arguments
    ---------
    model : model class
        Model structure to fit, as defined by build_model().
    n_validation : int
        Number of training examples used for cross-validation.
    write_to_file : bool
        Write model to file; can later be loaded through load_model().

    Returns
    -------
    model : model class
        The trained model.
    """

    #load training and validation data
    training_images, training_labels, validation_images, validation_labels = load_training_data('dataset/dataset_train_4_15000_08.npz')

    #compile model for training
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    #setup early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
    
    #model training
    history = model.fit(
        training_images, 
        training_labels,
        batch_size=64,
        epochs=20,
        verbose = 2,  
        validation_data=(validation_images, validation_labels),
        callbacks=[early_stopping]
    )

    # Save model to file if requested
    if write_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(f'./models/model_{timestamp}.keras')

    return model

def load_model(filepath = './models/model_example.keras'):
    """
    Load a model from file.

    Returns
    -------
    model : model class
        Previously trained model.
    """

    return saving.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)

def evaluate_model(model):
    """
    Evaluate model on the test set.

    Arguments
    ---------
    model : model class
        Trained model.

    Returns
    -------
    score : float
        A measure of model performance.
    """

    test_images, test_labels, _, _ = load_training_data('dataset_testing/dataset_testing.npz')
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    return test_accuracy

def identify(raw_image, model, is_4_rank_game=False):
    """
    Use model to classify a single card image and return the class prediction vector.

    Arguments
    ---------
    raw_image : 32x32 Image
        Raw image to classify.
    model : model class
        Trained model.
    is_4_rank_game: Bool
        Information about which game variation (3 vs 4 rank type)

    Returns
    -------
    prediction_vector : np.array
        string indicating predicted class 'A','J','K' or 'Q'
    """
    # Rank labels corresponding to model output
    ranks = ['A','J','K','Q']

    # Normalize the input image
    image = normalize_image(raw_image)

    # Expand dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)  # Shape should be (1, height, width, channels)

    # Use the model to predict the class probabilities
    prediction_vector = model.predict(image).flatten()

    # Get the index of the class with the highest confidence
    predicted_index = np.argmax(prediction_vector)

    print(f'Prediction [A J K Q] = {np.round(prediction_vector.flatten(),3)}')

    # case in witch 'A' was predicted in 3 card game chose next most probable rank
    if is_4_rank_game == False and ranks[predicted_index] == 'A':
        sorted_indicies = np.argsort(prediction_vector)[::-1]
        for idx in sorted_indicies:
            if ranks[idx] != 'A':
                return ranks[idx]

    return ranks[predicted_index]

