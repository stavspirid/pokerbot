"""
This module provides functions for generating, processing, and loading a dataset of card images used 
for training and testing a card recognition model. It includes tools to normalize images, apply 
one-hot encoding, generate noisy images, and manage datasets by loading from or saving to files.
"""

import os
import random
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random 
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
TRAINING_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "training_images")  # Directory for storing training images
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "test_images")  # Directory for storing test images
LABELS = ['J', 'Q', 'K', 'A']  # Possible card labels
IMAGE_SIZE = 32 
ROTATE_MAX_ANGLE = 15

FONTS = [
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'medium')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'medium')),
]  # True type system fonts


def normalize_image(raw_image: Image):
    """
    Normalize a raw image to serve as input to the image classifier.

    Arguments
    ---------
    raw_image : Image
        Raw image to normalize.

    Returns
    -------
    image/np.max(image): numpy array of floats between zero and one
        Normalized image that can be used by the image classifier.
    """

    image = np.array(raw_image)

    return image/np.max(image) #vector with values between zero and one

def one_hot_encoding(vector):
    """
    Do one-hoe encoding on a vector of labels.

    Arguments
    ---------

    vector : numpy array
        Array to one-hot encode
    
    Returns
    -------
    encoded_df.tonumpy() : one hot-encoded numpy vector 
        Vector with 0 and 1 that encodes a label into a numeric array
    """

    encoded_df = pd.get_dummies(vector) #use pandas to hanlde strings

    return encoded_df.to_numpy() #return a numpy vector

def load_data_set(data_dir, split = np.inf):
    """
    Normalize the images in data_dir and divide in a training and validation set.

    Parameters
    ----------
    data_dir : str
        Directory of images to load
    split : float
        Percentage of training data in relation with all the data
    """

    # Extract png files
    files = os.listdir(data_dir)
    png_files = []
    for file in files:
        if file.split('.')[-1] == "png":
            png_files.append(file)

    random.shuffle(png_files)  # Shuffled list of the png-file names that are stored in data_dir

    # Load the training and validation set and prepare the images and labels. Use normalize_image()
    # to normalize raw images (you can load an image with Image.open()) to be processed by your
    # image classifier. You can extract the original label from the image filename.
    

    total_images = len(png_files)
    train_test_split = split*total_images #number of images that will be in training data
    counter = 0 #Needed to see when to save files to test dataset
    

    training_images =[]
    training_labels = []
    validation_images = []
    validation_labels =  []

    for image in png_files:
        file_dir = data_dir + '/' + image
        im = Image.open(file_dir)
        im_vector = normalize_image(im) #normalize image before saving it

        if  counter < train_test_split: #save in training data if instance is lower than the threshold 
            training_images.append(im_vector)
            training_labels.append(image[0])

        else:
            validation_images.append(im_vector)
            validation_labels.append(image[0])

        counter += 1
    
    #encode labels before returning them
    training_labels = one_hot_encoding(np.array(training_labels))
    validation_labels = one_hot_encoding(np.array(validation_labels))


    return np.array(training_images), training_labels, np.array(validation_images), validation_labels

def dataset_to_npz(dataset_dir,filename):
    """
    Load dataset from .png, pre-process it, divide it into training and validation data and save it to .npz

    Parameters
    ----------
    data_dir : str
        Directory of images to load
    filename : str
        Name for .npz file
    """

    training_images, training_labels, validation_images, validation_labels = load_data_set(dataset_dir)

    np.savez(filename, 
         training_images=training_images, 
         training_labels=training_labels, 
         validation_images=validation_images, 
         validation_labels=validation_labels)

def load_training_data(file_directory):
    """
    Load dataset from .npz.
    If loading testing data, ignore the two latter outputs

    Parameters
    ----------
    file_directory : str
        Directory of .npz file to load

    """

    loaded_data = np.load(file_directory)

    # Access individual arrays
    training_images = loaded_data['training_images']
    training_labels = loaded_data['training_labels']
    validation_images = loaded_data['validation_images']
    validation_labels = loaded_data['validation_labels']

    return training_images, training_labels, validation_images, validation_labels

def generate_data_set(n_samples, data_dir):
    """
    Generate n_samples noisy images by using generate_noisy_image(), and store them in data_dir.

    Arguments
    ---------
    n_samples : int
        Number of train/test examples to generate
    data_dir : str in [TRAINING_IMAGE_DIR, TEST_IMAGE_DIR]
        Directory for storing images
    """

    noise_possibilities = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    # noise_possibilities = [0.1,0.2,0.3,0.4,0.5]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Generate a directory for data set storage, if not already present

    for i in range(n_samples):
        # Pick a random rank and convert it to a noisy image through generate_noisy_image().
        
        rank = random.choice(LABELS)
        noise_level = random.choice(noise_possibilities)
        img = generate_noisy_image(rank,noise_level)

        img.save(f"./{data_dir}/{rank}_{i}.png")  # The filename encodes the original label for training/testing

def generate_noisy_image(rank, noise_level):
    """
    Generate a noisy image with a given noise corruption. This implementation mirrors how the server generates the
    raw images. However the exact server settings for noise_level and ROTATE_MAX_ANGLE are unknown.
    For the PokerBot assignment you won't need to update this function, but remember to test it.

    Arguments
    ---------
    rank : str in ['J', 'Q', 'K']
        Original card rank.
    noise_level : float between zero and one
        Probability with which a given pixel is randomized.

    Returns
    -------
    noisy_img : Image
        A noisy image representation of the card rank.
    """

    if not 0 <= noise_level <= 1:
        raise ValueError(f"Invalid noise level: {noise_level}, value must be between zero and one")
    if rank not in LABELS:
        raise ValueError(f"Invalid card rank: {rank}")

    # Create rank image from text
    font = ImageFont.truetype(random.choice(FONTS), size = IMAGE_SIZE - 6)  # Pick a random font
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color = 255)
    draw = ImageDraw.Draw(img)
    (text_width, text_height) = draw.textsize(rank, font = font)  # Extract text size
    draw.text(((IMAGE_SIZE - text_width) / 2, (IMAGE_SIZE - text_height) / 2 - 4), rank, fill = 0, font = font)

    # Random rotate transformation
    img = img.rotate(random.uniform(-ROTATE_MAX_ANGLE, ROTATE_MAX_ANGLE), expand = False, fillcolor = '#FFFFFF')
    pixels = list(img.getdata())  # Extract image pixels

    # Introduce random noise
    for (i, _) in enumerate(pixels):
        if random.random() <= noise_level:
            pixels[i] = random.randint(0, 255)  # Replace a chosen pixel with a random intensity

    # Save noisy image
    noisy_img = Image.new('L', img.size)
    noisy_img.putdata(pixels)

    return noisy_img
