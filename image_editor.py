import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
from rembg import remove
import os

class ImageEditor:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("AI Image Editor")
        self.window.geometry("1200x800")
        
        # Variables
        self.current_image = None
        self.original_image = None
        self.filename = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create left panel for buttons
        self.left_panel = ctk.CTkFrame(self.main_frame, width=200)
        self.left_panel.pack(side="left", fill="y", padx=5, pady=5)
        
        # Create buttons
        ctk.CTkButton(self.left_panel, text="Open Image", command=self.open_image).pack(pady=5, padx=5)
        ctk.CTkButton(self.left_panel, text="Remove Background", command=self.remove_background).pack(pady=5, padx=5)
        ctk.CTkButton(self.left_panel, text="Enhance Lighting", command=self.enhance_lighting).pack(pady=5, padx=5)
        ctk.CTkButton(self.left_panel, text="Save Image", command=self.save_image).pack(pady=5, padx=5)
        
        # Create brightness slider
        self.brightness_label = ctk.CTkLabel(self.left_panel, text="Brightness")
        self.brightness_label.pack(pady=(20,0))
        self.brightness_slider = ctk.CTkSlider(self.left_panel, from_=0, to=2, command=self.adjust_brightness)
        self.brightness_slider.set(1)
        self.brightness_slider.pack(pady=5, padx=5)
        
        # Create contrast slider
        self.contrast_label = ctk.CTkLabel(self.left_panel, text="Contrast")
        self.contrast_label.pack(pady=(20,0))
        self.contrast_slider = ctk.CTkSlider(self.left_panel, from_=0, to=2, command=self.adjust_contrast)
        self.contrast_slider.set(1)
        self.contrast_slider.pack(pady=5, padx=5)
        
        # Create image display area
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="No image loaded")
        self.image_label.pack(fill="both", expand=True)
        
    def open_image(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if self.filename:
            self.original_image = Image.open(self.filename)
            self.current_image = self.original_image.copy()
            self.display_image()
    
    def display_image(self):
        if self.current_image:
            # Resize image to fit the display area while maintaining aspect ratio
            display_size = (800, 600)
            self.current_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(self.current_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
    
    def remove_background(self):
        if self.current_image:
            try:
                # Convert PIL Image to bytes
                img_byte_arr = self.current_image.convert('RGB')
                # Remove background
                output = remove(img_byte_arr)
                self.current_image = output
                self.display_image()
                messagebox.showinfo("Success", "Background removed successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error removing background: {str(e)}")
    
    def enhance_lighting(self):
        if self.current_image:
            try:
                # Convert to numpy array for OpenCV processing
                img_array = np.array(self.current_image)
                
                # Convert to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                
                # Merge channels
                limg = cv2.merge((cl,a,b))
                
                # Convert back to RGB
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                
                # Convert back to PIL Image
                self.current_image = Image.fromarray(enhanced)
                self.display_image()
                messagebox.showinfo("Success", "Lighting enhanced successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error enhancing lighting: {str(e)}")
    
    def adjust_brightness(self, value):
        if self.original_image:
            enhancer = ImageEnhance.Brightness(self.original_image)
            self.current_image = enhancer.enhance(float(value))
            self.display_image()
    
    def adjust_contrast(self, value):
        if self.current_image:
            enhancer = ImageEnhance.Contrast(self.current_image)
            self.current_image = enhancer.enhance(float(value))
            self.display_image()
    
    def save_image(self):
        if self.current_image:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if save_path:
                self.current_image.save(save_path)
                messagebox.showinfo("Success", "Image saved successfully!")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ImageEditor()
    app.run()
