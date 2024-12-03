from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
import requests
from PIL import Image
from io import BytesIO

# Function to remove the background
def remove_background(image_path, api_key):
    url = "https://api.remove.bg/v1.0/removebg"
    with open(image_path, 'rb') as img_file:
        response = requests.post(
            url,
            files={"image_file": img_file},
            headers={"X-Api-Key": api_key},
        )
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGBA")
    else:
        raise Exception(f"Error in background removal: {response.text}")

# Function to use Stable Diffusion inpainting for dress change
def change_dress(pipe, image, mask, prompt):
    # Ensure dimensions are compatible
    if mask.size != image.size:
        print("Resizing mask to match image dimensions...")
        mask = mask.resize(image.size, Image.Resampling.BILINEAR)
    
    width, height = image.size
    new_width = (width // 64) * 64
    new_height = (height // 64) * 64
    if (width, height) != (new_width, new_height):
        print("Resizing input image and mask to match Stable Diffusion requirements...")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        mask = mask.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Pass the adjusted inputs to the pipeline
    result = pipe(prompt=prompt, image=image, mask_image=mask)
    return result.images[0]

# Function to generate a new background using Stable Diffusion
def generate_background(pipe, prompt):
    result = pipe(prompt=prompt)
    return result.images[0].convert("RGBA")

# Main program
def main():
    # File paths and API keys
    input_image_path = "Poove-anna.jpeg"  # Input image path
    remove_bg_api_key = "WMmqHrkHzpou9ayGMVv5w7Jd"  # Replace with your Remove.bg API key
    dress_prompt = "A red evening gown, elegant, high-quality"  # Prompt to change dress
    background_prompt = "A beautiful sunset beach, photorealistic"  # Prompt for new background

    # Load the Stable Diffusion pipeline
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting"
    ).to("cpu")  # Use "cuda" if GPU is available

    try:
        # Step 1: Remove the background
        print("Removing background...")
        subject_image = remove_background(input_image_path, remove_bg_api_key)
        subject_image.save("subject.png")

        # Step 2: Prepare a mask for the dress
        print("Preparing dress mask...")
        mask_image = Image.open("mask.png").convert("RGB")  # Provide a valid mask

        # Step 3: Change the dress
        print("Changing the     dress...")
        updated_subject = change_dress(pipe_inpaint, subject_image, mask_image, dress_prompt)
        updated_subject.save("updated_subject.png")

        # Step 4: Generate a new background
        print("Generating new background...")
        pipe_background = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        pipe_background.load_lora_weights("anthienlong/flux_image_enhancer")
        background_image = generate_background(pipe_background, background_prompt)
        background_image.save("background.png")

        # Step 5: Combine the subject and the new background
        print("Combining subject and background...")
        final_image = Image.alpha_composite(background_image, updated_subject)
        final_image.save("final_image.png")
        print("Enhanced image saved as 'final_image.png'.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
