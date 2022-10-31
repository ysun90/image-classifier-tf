"""Predict an image with saved Keras model.

See `README.md` for a detailed discussion of this project.

This script can be invoked from the command line::

    $ python predict.py /path/to/image saved_model
    $ python predict.py /path/to/image saved_model --top_k KK
    $ python predict.py /path/to/image saved_model --category_names map.json

Examples:

    $ python predict.py ./test_images/orchid.jpg my_model.h5
    $ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
    $ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
"""
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from image_predictor import image_predictor

def make_parser():
    """Create an ArgumentParser for this script.

    :return: a parser.
    """
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('image_path')
    parser.add_argument('saved_model')
    parser.add_argument('--top_k', dest="top_k", type=int, default="5")
    parser.add_argument('--category_names', dest="category_names",
                        default="label_map.json")
    return parser

def main():
    """Run the main script."""
    parser = make_parser()
    args = parser.parse_args()

    # Load saved Keras model
    model = tf.keras.models.load_model(args.saved_model,
                                       custom_objects={'KerasLayer':hub.KerasLayer},
                                       compile=False)

    # Create a image predictor
    predictor = image_predictor(image_size=224)

    # Predict the image
    top_k_probs , top_k_classes, top_k_names = \
        predictor.predict_image(args.image_path, model,
                                args.top_k, args.category_names)

    # Show results
    print(top_k_probs)
    print(top_k_classes)
    print(top_k_names)

if __name__ == '__main__':
    main()
