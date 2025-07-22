from ultralytics import YOLO
import cv2
import argparse
import time
import os

def main():
    parser = argparse.ArgumentParser(description="Run YOLO predictions on camera feed or a single image.")

    # Predict form ethier a picture, or a camera, not both.
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-c", "--camera", type=str, help="Show predictions in real-time from specified camera.")
    group.add_argument("-p", "--picture", type=str, help="Path to an image file to save prediction as 'prediction-*timestamp*.jpg' in 'predictions' folder.")

    # Path to the model
    parser.add_argument("-m", "--model", type=str, default="models/160p500es-seg.pt", help="Path to the trained model, default is models/160p500es-seg.pt")
    
    args = parser.parse_args()

    # Load the trained model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure '160p500es-seg.pt' is in the same directory or provide a valid path to your trained model.")
        return

    if args.camera:
        try:
            cap = cv2.VideoCapture(int(args.camera)) 
            if not cap.isOpened():
                raise IOError("Cannot open webcam")
        except Exception as e:
            print(f"Error opening webcam: {e}")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            # Perform inference
            conf = 0.6
            results = model.predict(frame, verbose=False, conf=conf)

            # Annotate the frame with predictions
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Real-time Predictions", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif args.picture:
        print(f"Processing image: {args.picture}")
        try:
            image = cv2.imread(args.picture)
            if image is None:
                print(f"Error: Could not load image from {args.picture}")
                return

            # Perform inference
            results = model(image, verbose=False)

            # Annotate the image
            annotated_image = results[0].plot()

            # Save the annotated image with timestamp into predictions directory
            os.makedirs('predictions', exist_ok=True)
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
            output_filename = os.path.join('predictions', f"prediction-{timestamp}.jpg")
            cv2.imwrite(output_filename, annotated_image)
            print(f"Prediction saved to {output_filename}")

        except Exception as e:
            print(f"Error processing picture: {e}")

    else:
        print("Please specify either --camera for real-time predictions or --picture <path_to_image> for single image prediction.")
        parser.print_help()

if __name__ == "__main__":
    main()