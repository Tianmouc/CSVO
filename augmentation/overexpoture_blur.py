import cv2
import numpy as np

def blur_and_overexpose(image_path, blur_kernel=(15, 15), exposure_factor=1.5):
    """
    Apply blur and overexposure effects to an image.

    Parameters:
        image_path (str): Path to the input image.
        blur_kernel (tuple): Kernel size for the blur effect (width, height).
        exposure_factor (float): Factor to increase the exposure (e.g., 1.5 for 50% brighter).

    Returns:
        processed_image (numpy.ndarray): The processed image with blur and overexposure.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, blur_kernel, 0)

    # Apply overexposure by scaling pixel values
    overexposed_image = cv2.convertScaleAbs(blurred_image, alpha=exposure_factor, beta=0)

    return overexposed_image

# Example usage
if __name__ == "__main__":
    input_path = "/data/zzx/DPVO_E2E/datasets/TartanAirNew/amusement/Easy/P001/image_left/000000_left.png"  # Replace with your image path
    output_path = "output.jpg"

    try:
        processed = blur_and_overexpose(input_path, blur_kernel=(15, 15), exposure_factor=2)
        cv2.imwrite(output_path, processed)
        print(f"Processed image saved to {output_path}")
    except ValueError as e:
        print(e)
