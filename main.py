import argparse
from src.captioning.eval import evaluate_image as caption_image
from src.classification.predict import predict as classify_image
from src.generation.generator import generate_image

def main():
    parser = argparse.ArgumentParser(description="VisionVerse: Image Captioning, Classification, and Generation")
    parser.add_argument('--task', choices=['caption', 'classify', 'generate'], required=True, help='Task to perform')
    parser.add_argument('--input', required=True, help='Input image path or text prompt')
    parser.add_argument('--model', required=True, help='Path to the model checkpoint')
    parser.add_argument('--vocab', help='Path to vocabulary file (for captioning)')
    parser.add_argument('--categories', help='Path to category names file (for classification)')
    args = parser.parse_args()

    if args.task == 'caption':
        result = caption_image(args.input, args.model, args.vocab)
        print(f"Caption: {result}")
    elif args.task == 'classify':
        probs, classes = classify_image(args.input, args.model, category_names=args.categories)
        for prob, cls in zip(probs, classes):
            print(f"{cls}: {prob:.4f}")
    elif args.task == 'generate':
        image = generate_image(args.input)
        image.save('generated_image.png')
        print("Image generated and saved as 'generated_image.png'")

if __name__ == "__main__":
    main()
