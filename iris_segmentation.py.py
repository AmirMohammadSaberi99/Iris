import cv2
import numpy as np
import argparse


def segment_iris(image_path, output_mask_path=None, output_overlay_path=None):
    """
    Segment the iris from an eye image using thresholding and contour detection.

    Parameters:
        image_path (str): Path to the input eye image.
        output_mask_path (str, optional): Path to save the binary iris mask.
        output_overlay_path (str, optional): Path to save the iris overlay on the original image.

    Returns:
        mask (numpy.ndarray): Binary mask of the segmented iris.
        overlay (numpy.ndarray): Original image masked by the iris region.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold using Otsu's method (invert to get dark iris on white)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in the thresholded image.")

    # Keep the largest contour assuming it's the iris
    max_contour = max(contours, key=cv2.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=-1)

    # Optional: Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Generate overlay of the iris on the original image
    overlay = cv2.bitwise_and(img, img, mask=mask)

    # Save outputs if paths are provided
    if output_mask_path:
        cv2.imwrite(output_mask_path, mask)
    if output_overlay_path:
        cv2.imwrite(output_overlay_path, overlay)

    return mask, overlay


def main():
    parser = argparse.ArgumentParser(
        description="Segment the iris from an eye image using thresholding and contour detection."
    )
    parser.add_argument(
        "image", help="Path to the input eye image (e.g., eye.jpg)"
    )
    parser.add_argument(
        "--mask", help="Path to save the binary iris mask", default="iris_mask.png"
    )
    parser.add_argument(
        "--overlay", help="Path to save the iris overlay image", default="iris_overlay.png"
    )
    args = parser.parse_args()

    mask, overlay = segment_iris(args.image, args.mask, args.overlay)
    print(f"Iris mask saved to {args.mask}")
    print(f"Iris overlay saved to {args.overlay}")


if __name__ == "__main__":
    main()

# Usage:
# python iris_segmentation.py eye.jpg --mask iris_mask.png --overlay iris_overlay.png
# Requirements: pip install opencv-python numpy
