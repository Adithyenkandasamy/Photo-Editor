import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from rembg import remove
import requests
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline
import mediapipe as mp
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import warnings
warnings.filterwarnings('ignore')

# Initialize CLIP model for segmentation
def init_clip_model():
    print("Initializing CLIP model...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    return processor, model

def enhance_image(image):
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Enhance details
    enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    return enhanced

def remove_background(image_path):
    print("Removing background...")
    with open(image_path, 'rb') as i:
        input_image = i.read()
        output = remove(input_image)
        
    output_path = os.path.splitext(image_path)[0] + "_nobg.png"
    with open(output_path, 'wb') as o:
        o.write(output)
    return output_path

def change_clothing(image_path, target_clothing="business suit"):
    print(f"Changing clothing to {target_clothing}...")
    
    # Load CLIP model
    processor, model = init_clip_model()
    
    # Load and preprocess image
    image = Image.open(image_path)
    inputs = processor(images=image, text=[target_clothing], padding=True, return_tensors="pt")
    
    # Get segmentation mask
    with torch.no_grad():
        outputs = model(**inputs)
        mask = torch.sigmoid(outputs.logits)
        mask = mask.squeeze().numpy()
        
    # Convert mask to binary
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Create colored overlay for clothing
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Dark blue color for business suit
    colored_mask[mask > 127] = [50, 50, 150]
    
    # Convert original image to numpy array
    original = cv2.imread(image_path)
    original = cv2.resize(original, (mask.shape[1], mask.shape[0]))
    
    # Blend original image with colored mask
    alpha = 0.7
    result = cv2.addWeighted(original, 1-alpha, colored_mask, alpha, 0)
    
    # Save result
    output_path = os.path.splitext(image_path)[0] + "_suited.png"
    cv2.imwrite(output_path, result)
    return output_path

def add_tech_background(image_path):
    print("Adding technological background...")
    foreground = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    height, width = foreground.shape[:2]
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create more complex tech background
    for i in range(width):
        for j in range(height):
            # Create a grid pattern
            grid = (i // 20 + j // 20) % 2
            color = int((i / width) * 255)
            if grid:
                background[j, i] = [color, color//2, color]
            else:
                background[j, i] = [color//2, color//3, color]
    
    # Add some random "tech" circles
    for _ in range(20):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(5, 30)
        color = (0, np.random.randint(150, 255), np.random.randint(150, 255))
        cv2.circle(background, (x, y), radius, color, 2)
    
    # Combine foreground and background
    if foreground.shape[-1] == 4:  # If image has alpha channel
        alpha = foreground[:, :, 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        foreground = foreground[:, :, :3]
        result = background * (1 - alpha) + foreground * alpha
    else:
        result = foreground
    
    output_path = os.path.splitext(image_path)[0] + "_tech_bg.png"
    cv2.imwrite(output_path, result)
    return output_path

def process_image(image_path):
    try:
        print(f"Processing image: {image_path}")
        
        # Step 1: Enhance the image
        print("Enhancing image quality...")
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image")
        
        enhanced_image = enhance_image(image)
        enhanced_path = os.path.splitext(image_path)[0] + "_enhanced.jpg"
        cv2.imwrite(enhanced_path, enhanced_image)
        print(f"Enhanced image saved as: {enhanced_path}")
        
        # Step 2: Change clothing to suit
        suited_path = change_clothing(enhanced_path)
        print(f"Image with suit overlay saved as: {suited_path}")
        
        # Step 3: Remove background
        no_bg_path = remove_background(suited_path)
        print(f"Image with removed background saved as: {no_bg_path}")
        
        # Step 4: Add technological background
        final_path = add_tech_background(no_bg_path)
        print(f"Final image with tech background saved as: {final_path}")
        
        print("\nProcessing completed!")
        return final_path
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    print("Image Processing Options:")
    print("1. Process image (enhance, change clothing, add tech background)")
    print("2. Only enhance image")
    print("3. Only change clothing")
    print("4. Only change background")
    
    choice = input("\nEnter your choice (1-4): ")
    image_path = input("Enter the path to your image: ")
    
    try:
        if choice == "1":
            result_path = process_image(image_path)
        elif choice == "2":
            image = cv2.imread(image_path)
            enhanced = enhance_image(image)
            result_path = os.path.splitext(image_path)[0] + "_enhanced.jpg"
            cv2.imwrite(result_path, enhanced)
        elif choice == "3":
            result_path = change_clothing(image_path)
        elif choice == "4":
            no_bg = remove_background(image_path)
            result_path = add_tech_background(no_bg)
        else:
            print("Invalid choice!")
            return
        
        if result_path:
            print(f"\nProcessing completed successfully!")
            print(f"Final image saved as: {result_path}")
        else:
            print("\nProcessing failed. Please check the error message above.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()