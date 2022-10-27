import tensorflow as tf
import json
from PIL import Image

def process_image(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /=255
    return image.numpy()

def load_map(path_map):
    with open(path_map, 'r') as f:
        class_names = json.load(f)

def predict_image(image_path, model, top_k, path_map):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)

    prob_predict = model.predict(np.expand_dims(processed_test_image, axis=0))
    probs = np.sort(prob_predict)

    top_k_prob = probs[0][-top_k:][::-1].tolist()
    top_k_classes = prob_predict.argsort()[0][-top_k:][::-1]
    top_k_classes = [str(i) for i in top_k_classes]

    class_names = load_map(path_map)
    top_k_class_names = [class_names[str(label + 1)] for label in top_k_classes]

    return top_k_prob , top_k_classes, tok_k_class_names
