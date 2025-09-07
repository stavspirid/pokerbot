from model import *
import pytest
import os
import numpy as np
from keras.models import Sequential


TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory
TRAINING_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "training_images")
TEST_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "test_images")


class TestModel:

    def test_build_model(self):
        """Test if the build_model() function creates a valid Keras model."""
        model = build_model()
        
        # Ensure that the model is of type Sequential
        assert isinstance(model, Sequential), "The model should be a Keras Sequential model"
        
        # Ensure that the model has the correct number of layers (5 Conv/Dense + 2 MaxPooling + Flatten + Dropout)
        assert len(model.layers) == 10, f"Expected 10 layers, but got {len(model.layers)}"
        
        # Ensure the last layer is softmax with 4 outputs (for the 4 classes)
        assert model.layers[-1].output_shape == (None, 4), "The output layer should have 4 units"

    def test_load_model(self):
        """Test if the load_model() function successfully loads a model."""
        model = build_model()  # Build a sample model
        model.save('test_model.h5')  # Save it to a file for loading later
        
        loaded_model = load_model('test_model.h5')  # Load the saved model
        
        # Check if the loaded model matches the built model structure
        assert isinstance(loaded_model, Sequential), "Loaded model should be a Keras Sequential model"
        assert len(loaded_model.layers) == len(model.layers), "Loaded model should have the same layers as the original model"
        
        # Clean up saved model
        os.remove('test_model.h5')

    def test_identify(self):
        """Test if the identify() function correctly identifies a card rank."""
        
        model = build_model()

        # Mock image data (32x32 grayscale image)
        raw_image = np.random.rand(32, 32)

        def mock_predict_4_rank(image):
            return np.array([[0.1, 0.7, 0.1, 0.1]])  # Pretend it predicts "J" with 70% confidence

        # Mock model prediction for a 3-rank game where "A" is excluded
        def mock_predict_3_rank(image):
            return np.array([[0.9, 0.05, 0.03, 0.02]])  # Pretend it predicts "A" with 90% confidence


        model.predict = mock_predict_4_rank
        prediction_4_rank = identify(raw_image, model, is_4_rank_game=True)
        assert prediction_4_rank == 'J', f"Expected 'J' as the predicted rank, but got {prediction_4_rank}"

        # Test for the 3-rank game where "A" should be excluded
        model.predict = mock_predict_3_rank
        prediction_3_rank = identify(raw_image, model, is_4_rank_game=False)
        assert prediction_3_rank != 'A', f"Expected any rank except 'A', but got {prediction_3_rank}"
        assert prediction_3_rank in ['J', 'K', 'Q'], f"Expected valid rank ('J', 'K', 'Q'), but got {prediction_3_rank}"


