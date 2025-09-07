from data_sets import *
import pytest
import os
import warnings

TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory
TRAINING_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "training_images")
TEST_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "test_images")

class TestDataSets:

    def test_normalize_images(self):

        #We only need an image example to test the function
        training_img_dir = TRAINING_IMAGE_TEST_DIR+'/J_1.png'

        normalized_img_train = normalize_image(Image.open(training_img_dir))

        #Verify output is a numpy array
        assert type(normalized_img_train) == np.ndarray

        #Verify if normalization was done
        assert (np.max(normalized_img_train) <= 1) and (np.min(normalized_img_train)>=0)

    def test_load_data_set(self):
        training_images, training_labels, validation_images, validation_labels = load_data_set(TRAINING_IMAGE_TEST_DIR)

        #check if outputs are numpy array, if training is so is test
        assert type(training_images) == np.ndarray
        assert type(training_labels) == np.ndarray
        assert type(validation_images) == np.ndarray
        assert type(validation_labels) == np.ndarray

        #check if training and validation vectors have the same number of instances
        assert training_images.shape[0] == training_labels.shape[0]
        assert training_images.shape[0] == training_labels.shape[0]

    def test_generate_noisy_image(self):
        # Ignore warnings on code we shouldn't edit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

            # Verify if outputs an image, noise factor doesn't matter
            assert type(generate_noisy_image('J', 0.1)) == Image.Image
            assert type(generate_noisy_image('Q', 0.1)) == Image.Image
            assert type(generate_noisy_image('K', 0.1)) == Image.Image

            # Verify if images are 32x32 pxs
            assert generate_noisy_image('J', 0.1).size == (32, 32)
            assert generate_noisy_image('Q', 0.1).size == (32, 32)
            assert generate_noisy_image('K', 0.1).size == (32, 32)

            # Invalid modes of behavior
            with pytest.raises(ValueError):
                generate_noisy_image('X', 0.1)  # wrong rank

            # Wrong noise values
            with pytest.raises(ValueError):
                generate_noisy_image('J', -1)

            with pytest.raises(ValueError):
                generate_noisy_image('J', 2) 

    def test_one_hot_encoding(self):

        labels_example = np.array(['Q','Q','J','K','J','J','K'])

        #verify if output is an array
        assert type(one_hot_encoding(labels_example)) == np.ndarray

        #verify if number of different ranks is equal to the number of columns in output
        assert one_hot_encoding(labels_example).shape[1] == len(np.unique(labels_example))

    def test_generate_data_set(self):
        # ignore warnings on code we shouldn't edit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

            generate_data_set(10, 'test_data')
            generated_files = os.listdir('test_data')

            # Verify that 10 images were generated
            assert len(generated_files) == 10

            # Check if files have correct naming
            for file in generated_files:
                assert file.endswith('.png')

            # Clean up generated test data
            for file in generated_files:
                os.remove(os.path.join('test_data', file))
            os.rmdir('test_data')

    def test_dataset_to_npz(self):
        # Ignore warnings on code we shouldn't edit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            generate_data_set(10, 'test_data_for_npz')
            dataset_to_npz('test_data_for_npz', 'test_dataset.npz')

            # Use a context manager to ensure the file is properly closed after reading
            with np.load('test_dataset.npz') as loaded_data:
                # Check if the loaded data contains the expected keys
                assert 'training_images' in loaded_data
                assert 'training_labels' in loaded_data
                assert 'validation_images' in loaded_data
                assert 'validation_labels' in loaded_data

            # Clean up
            os.remove('test_dataset.npz')
            for file in os.listdir('test_data_for_npz'):
                os.remove(os.path.join('test_data_for_npz', file))
            os.rmdir('test_data_for_npz')