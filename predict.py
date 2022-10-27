import argparse

from func_utils import process_image, predict_image

def make_parser():
    parser = argparse.ArgumentParser(description="Predict an Image.")

    parser.add_argument('dir', type=str)
    parser.add_argument('model', type=str)

    parser.add_argument('--top_k', type=int, default="5",
                        help="Return the top KK most likely classes.")
    parser.add_argument('--category_names', type=str, default="label_map.json",
                        help="Path to a JSON file mapping labels to flower names")

    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    top_k_prob , top_k_classes, tok_k_class_names = \
        predict_image(args.dir, args.model, args.top_k, args.category_names)

    print(top_k_prob)
    print(top_k_classes)
    print(top_k_class_names)

if __name__ == '__main__':
    main()
