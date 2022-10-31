"""Predict image with saved Keras model."""
import numpy as np
import tensorflow as tf
import json
from PIL import Image

class image_predictor:
    """
    An image predictor.
    """

    def __init__(self, image_size=224):
        """Create an image predcitor.

        :param image_size: int, size of image (both width and height), default 224.
        """
        self.image_size = image_size

    def process_image(self, image):
        """Resize and normalize image to fullfill the requirements of model.

        :param image: np.array, image data.
        :return: np.array, processed image.
        """
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        image /=255
        return image.numpy()

    def load_map(self, path_map):
        """Load a JSON file to get a dictionary mapping label to name.

        :param path_map: JSON file.
        :return: a dictionary mapping label to names.
        """
        with open(path_map, 'r') as f:
            class_names = json.load(f)
        return class_names

    def predict_image(self, image_path, model, top_k, path_map):
        """Predict the class name in the image.

        :param image_path: path to the image to be predicted.
        :param model: saved keras model.
        :param top_k: int
        :path_map: JSON file mapping label to class names
        :return: (top_k_probs , top_k_classes, top_k_names),
                 top_k most likely probabilities, classes and names.
        """
        # Open and preprocess the image
        im = Image.open(image_path)
        test_image = np.asarray(im)
        processed_test_image = self.process_image(test_image)

        # Predict with Keras model
        probs_predict = model.predict(np.expand_dims(processed_test_image, axis=0))
        # Sort prbabilities from least to most
        probs_ascend = np.sort(probs_predict)
        # Get top_k probabilities and convet to list format
        top_k_probs = probs_ascend.squeeze()[::-1][:top_k].tolist()

        # Get class labels [0-101]
        top_k_class_indices = probs_predict.argsort().squeeze()[::-1][:top_k]
        # Convert to class names in mapping [1-102]
        top_k_classes = [str(i+1) for i in top_k_class_indices]

        # Load mapping from label to class name
        class_names = self.load_map(path_map)
        # Get top_k class names
        top_k_names = [class_names[label] for label in top_k_classes]

        return top_k_probs , top_k_classes, top_k_names
