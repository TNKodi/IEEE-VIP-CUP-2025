
from ultralytics import YOLO
from tqdm import tqdm 
import torch
import shutil
import time
import multiprocessing
import os
import numpy as np


def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Check if CUDA is available and print the device being used  

    devNumber= torch.cuda.current_device() 
    print(f"Device number: {devNumber}")  # Print the device number being used
    denName= torch.cuda.get_device_name(devNumber)
    print(f"Gpu name: {denName}")

    # Load model
    model = YOLO("runs/detect/drone_colab_exp/weights/last.pt")  # Load the best model from training

    # Training loop

    model.train(
    data='drone.yaml',
    epochs=5,
    imgsz=320,
    batch=16,
    device=device,
    half=True,
    name='drone_colab_exp',
    
)
  
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Critical for Windows
    main()