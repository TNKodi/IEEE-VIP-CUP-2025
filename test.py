from ultralytics import YOLO
import cv2
import torch
# Load your best model
model = YOLO('bestm.pt')  # adjust path if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Check if CUDA is available and print the device being used  

devNumber= torch.cuda.current_device() 
print(f"Device number: {devNumber}")  # Print the device number being used
denName= torch.cuda.get_device_name(devNumber)
print(f"Gpu name: {denName}")  # Print the name of the device being used
# Run predictions on test image folder
for result in model.predict(
    source='datasets\\test\\images',  # adjust path if needed
    imgsz=320,
    conf=0.25,
    save=True,
    save_txt=False,
    name='drone_test_video',
    stream=True,
    device=device,
):
    # result.orig_img is the raw frame
    # result.plot() draws the detections onto it
    frame = result.plot()

    # Show the annotated frame
    cv2.imshow("YOLOv8 Video", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
