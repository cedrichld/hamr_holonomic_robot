import cv2
import numpy as np

# Load an image
image = cv2.imread('terrain_assets/heightmaps/terrain_512.png') 

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    # Apply Gaussian blur
    # Parameters:
    # 1. src: The input image.
    # 2. ksize: Gaussian kernel size. It should be a tuple (width, height) where both width and height are positive and odd.
    # 3. sigmaX: Gaussian kernel standard deviation in X direction.
    # 4. sigmaY: Gaussian kernel standard deviation in Y direction. If 0, it is taken as sigmaX.
    # 5. borderType: Pixel extrapolation method (optional, default is BORDER_DEFAULT).
    blurred_image = cv2.GaussianBlur(image, (1, 1), 0)

    # Display the original and blurred images (optional)
    cv2.imshow('Original Image', image)
    cv2.imshow('Gaussian Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the blurred image (optional)
    cv2.imwrite('terrain_assets/heightmaps/terrain_512_new_blurr.png', blurred_image)