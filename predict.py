import argparse
import tensorflow as tf
import tensorflow_hub as hub
from func_utils import predict_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('image_path')
    parser.add_argument('saved_model')
    parser.add_argument('--top_k', dest="top_k", type=int, default="5")
    parser.add_argument('--category_names', dest="category_names", default="label_map.json")
    args = parser.parse_args()
    
    model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)

    top_k_probs , top_k_classes, top_k_names = predict_image(args.image_path, model, args.top_k, args.category_names)

    print(top_k_probs)
    print(top_k_classes)
    print(top_k_names)
