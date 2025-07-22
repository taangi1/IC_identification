from ultralytics import YOLO
import argparse
import os
import sys
from pathlib import Path

def train_yolo_model():
    """
    Parses command-line arguments and initiates the YOLO model training process.
    This script is designed to be flexible, allowing users to specify their
    custom dataset, model architecture, pre-trained weights, and training
    hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Train a YOLO model on custom data.")

    # --- Core Training Arguments ---
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to the dataset YAML file (e.g., data.yaml). '
             'This file defines the paths to your training/validation images '
             'and labels, and class names.'
    )
    parser.add_argument(
        '--model', type=str, default='models/160p500es-seg.pt', # Default to a common pre-trained model
        help='Path to the pre-trained model (e.g. 160p500es-seg.pt)'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Total batch size for training (per GPU if multiple GPUs are used).'
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='Input image size (pixels) for training and validation.'
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help='Device to use for training (e.g., "0" for GPU 0, "0,1" for GPU 0 and 1, "cpu"). '
             'Automatically defaults to GPU 0 if available, otherwise CPU.'
    )

    # --- Advanced/Optional Arguments ---
    parser.add_argument(
        '--name', type=str, default='yolo_custom_train',
        help='Name of the training run. Results will be saved in runs/detect/name.'
    )
    parser.add_argument(
        '--project', type=str, default='runs/segment',
        help='Directory to save results (e.g., runs/segment).'
    )
    parser.add_argument(
        '--weights', type=str, default='', # Optional: explicit path to initial weights
        help='Path to initial weights to load. Overrides --model if both are specified '
             'and --model points to a config file. If --model is a .pt file, this is redundant.'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from the last checkpoint.'
    )
    parser.add_argument(
        '--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'AdamW', 'auto'],
        help='Optimizer to use for training.'
    )
    parser.add_argument(
        '--lr0', type=float, default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--lrf', type=float, default=0.01,
        help='Final learning rate (as a ratio of lr0).'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of Dataloader workers (for data loading parallelism).'
    )
    parser.add_argument(
        '--cache', type=str, default='ram', choices=['ram', 'disk', ''],
        help='Cache images for faster training: "ram" (default) or "disk". '
             'Leave empty for no caching.'
    )
    parser.add_argument(
        '--patience', type=int, default=50,
        help='Epochs to wait for no improvement in validation metric before early stopping.'
    )
    parser.add_argument(
        '--val', action='store_true',
        help='Perform validation during training.'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility.'
    )


    args = parser.parse_args()

    # --- Path Validation ---
    data_path = Path(args.data)
    model_path = Path(args.model)

    if not data_path.is_file():
        print(f"Error: Data YAML file not found at '{data_path}'")
        sys.exit(1)

    if not model_path.exists(): # Check if it's a file or a valid model name
        print(f"Error: Model file or name '{model_path}' not found or invalid.")
        print("Please provide a valid path to a .yaml config or .pt weights file, or a valid model name (e.g., 'yolov8n.pt').")
        sys.exit(1)

    # --- Initialize YOLO Model ---
    print(f"\nInitializing YOLO model with: {args.model}")
    model = YOLO(args.model) # Loads a pre-trained model or creates from a config

    # If explicit weights are provided, load them (this can override the model argument if it was a config)
    if args.weights and Path(args.weights).is_file():
        print(f"Loading additional weights from: {args.weights}")
        model.load(args.weights)
    elif args.weights:
        print(f"Warning: Specified weights file '{args.weights}' not found. Continuing without loading.")


    # --- Start Training ---
    print("\n--- Starting YOLO Model Training ---")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Run Name: {args.name}")
    print(f"Project Directory: {args.project}")
    print("-" * 30)

    try:
        # The 'train' method of the YOLO model handles the entire training loop.
        # It automatically saves checkpoints and results.
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            name=args.name,
            project=args.project,
            device=args.device,
            resume=args.resume,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            workers=args.workers,
            cache=args.cache,
            patience=args.patience,
            val=args.val,
            seed=args.seed,
            # Add any other Ultralytics training arguments here as needed
            # For a full list, refer to the Ultralytics documentation:
            # https://docs.ultralytics.com/modes/train/
        )
        print("\n--- Training Completed Successfully! ---")
        print(f"Results saved to: {Path(args.project) / args.name}")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_yolo_model()