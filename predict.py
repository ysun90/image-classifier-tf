import argparse

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

    print(f'{args.dir}')
    print(f'{args.model}')
    print(f'{args.top_k}')
    print(f'{args.category_names}')

if __name__ == '__main__':
    main()
