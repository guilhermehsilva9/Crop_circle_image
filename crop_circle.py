"""
# Atak Sistemas Technical Test: Circular Image Cropping

Author: Guilherme Henrique da Silva
Email: guilherme.utf@gmail.com
Date: January 27, 2026

This module provides functionality to crop a circular region of interest (ROI) from an image.
It includes a demonstration script that allows users to specify the image path, circle center coordinates, and radius 
via command-line arguments. The cropped circular region is saved as a new image file. 

## Dependencies:
- Python 3.x
- OpenCV
- NumPy

## Usage Examples
1. Default parameters:
    python crop_circle.py

2. Grayscale image:
    python crop_circle.py --image hannibal_lecter_gray.png

3. Custom coordinates and image:
    python crop_circle.py --image <path_to_image> --x <x_coordinate> --y <y_coordinate> --r <circle_radius>

4. Help:
    python crop_circle.py --help
"""

import cv2
import numpy as np
import argparse
import os

def crop_circular_region(image: np.ndarray, x: int, y: int, r: int):
    """
    Extracts a circular region of interest (ROI) from an image.
    
    The function calculates the bounding box of the circle, handles edge 
    constraints, and uses vectorized NumPy operations to mask the exterior 
    pixels with black.
    
    Args:
        image (numpy.ndarray): Input image (BGR or Grayscale).
        x (int): Horizontal coordinate of the circle center (column).
        y (int): Vertical coordinate of the circle center (row).
        r (int): Radius of the circle in pixels.
    
    Returns:
        numpy.ndarray: The smallest square image containing the circular crop.
    """
    # Validate inputs
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if r <= 0:
        raise ValueError("Radius (r) must be a positive integer.")
    
    h, w = image.shape[:2]

    # Calculate bounding box coordinates, ensuring they stay within image bounds
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r), min(h, y + r)

    # Ensure the resulting coordinates define a valid area within the image
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"The specified circle (center: {x}, {y}) is completely outside "
                         f"the image boundaries. Try adjusting the center coordinates. Image size: ({w}, {h}), ")

    # Extract the region of interest (ROI) based on the bounding box
    roi = image[y1:y2, x1:x2]

    # Generate a grid of (x, y) coordinates within the ROI
    yy, xx = np.indices(roi.shape[:2])
    
    # Convert grid coordinates to global image coordinates
    global_xx = xx + x1
    global_yy = yy + y1

    # Apply the circle equation (x - x0)^2 + (y - y0)^2 <= r^2 to create a mask
    distances_squared = (global_xx - x)**2 + (global_yy - y)**2

    # Create a mask for pixels inside the circle
    mask = distances_squared <= r**2

    # Initialize output image with black pixels
    output = np.zeros_like(roi)
    
    # Apply the mask: where mask is True, copy pixels from ROI
    output[mask] = roi[mask]

    return output

# --- Demonstration ---
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Crop a circular region from an image.")
    argparser.add_argument('--image', type=str, default='hannibal_lecter.png', help='Path to the input image.')
    argparser.add_argument('--x', type=int, default=390, help='X coordinate of the circle center.')
    argparser.add_argument('--y', type=int, default=70, help='Y coordinate of the circle center.')
    argparser.add_argument('--r', type=int, default=40, help='Radius of the circle in pixels.')
    args = argparser.parse_args()
    
    # Check if the image file exists
    if not os.path.isfile(args.image):
        print(f"[ERROR] File '{args.image}' does not exist. Check the image path and try again.")
    else:
        # Load the image
        img = cv2.imread(args.image)

        # Check if image loading was successful
        if img is None:
            print(f"[ERROR] Unable to load image '{args.image}'. Check the file format and try again.")
        else:
            try:
                # Parameters: center coordinates and radius
                x, y, r = args.x, args.y, args.r
                print(f"Croping circular region at ({x}, {y}) with radius {r}.")
                
                result = crop_circular_region(img, x, y, r)
                resized_result = cv2.resize(result, (200, 200))  # Resize for better visibility

                # Save the resulting image
                cv2.imwrite("circular_crop_result.png", result)   
                cv2.imshow("Circular ROI", resized_result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                print("Circular crop saved as 'circular_crop_result.png'.")
                print("Process completed.")
            
            # Catch and print any errors encountered during processing
            except Exception as e:
                print(f"[ERROR] {e}")