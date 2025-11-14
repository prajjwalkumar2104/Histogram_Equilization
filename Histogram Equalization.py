import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, title="Histogram"):
    """Helper function to plot image histogram"""
    if len(image.shape) == 2:  # Grayscale
        plt.hist(image.flatten(), 256, [0,256], color='black')
        plt.title(title)
    else:  # Color
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            plt.hist(image[:,:,i].flatten(), 256, [0,256], color=col, alpha=0.5)
        plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
    plt.show()

def equalize_histogram_grayscale(image):
    """Histogram Equalization for Grayscale"""
    return cv2.equalizeHist(image)

def equalize_histogram_color(image):
    """Histogram Equalization for Color (HSV Intensity Channel)"""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    eq_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return eq_img

def main():
    # Load your image
    image_path = 'Test Flower.jpg'  # Change this to your image file path
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image!")
        return

    print("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    plot_histogram(image, "Original Image Histogram")
    
    # Grayscale Equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_gray = equalize_histogram_grayscale(gray)
    print("Grayscale Histogram Equalization")
    images = [gray, eq_gray]
    titles = ["Original Grayscale", "Histogram Equalized"]
    plt.figure(figsize=(10,5))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1,2,i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()
    
    plot_histogram(gray, "Grayscale Histogram Before Equalization")
    plot_histogram(eq_gray, "Grayscale Histogram After Equalization")
    
    # Color Equalization (HSV channel processing)
    eq_color = equalize_histogram_color(image)
    print("Color Image Histogram Equalization (HSV)")
    
    images = [image, eq_color]
    titles = ["Original Color", "Histogram Equalized (HSV Intensity)"]
    plt.figure(figsize=(10,5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1,2,i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()
    
    plot_histogram(image, "Color Histogram Before Equalization")
    plot_histogram(eq_color, "Color Histogram After Equalization")
    
    # Save results
    cv2.imwrite('equalized_grayscale.jpg', eq_gray)
    cv2.imwrite('equalized_color.jpg', eq_color)
    print("Equalized images saved to current directory!")

if __name__ == "__main__":
    main()
