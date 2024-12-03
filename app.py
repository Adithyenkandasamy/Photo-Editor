    import requests
    from PIL import Image
    from io import BytesIO
    from diffusers import StableDiffusionInpaintPipeline

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
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(f"Error in background removal: {response.text}")

    # Function to use Stable Diffusion inpainting for dress change
    def change_dress(pipe, image, mask, prompt):
        result = pipe(prompt=prompt, image=image, mask_image=mask)
        return result.images[0]

    # Function to generate a new background using Stable Diffusion
    def generate_background(pipe, prompt):
        result = pipe(prompt=prompt)
        return result.images[0]

    # Main program
    def main():
        # File paths and API keys
        input_image_path = "/home/yellowflash/Adithyenrepose/Photo-Editor/Poove-anna.jpeg"  # Input image path
        remove_bg_api_key = "WMmqHrkHzpou9ayGMVv5w7Jd"  # Replace with your Remove.bg API key
        dress_prompt = "A red evening gown, elegant, high-quality"  # Prompt to change dress
        background_prompt = "A beautiful sunset beach, photorealistic"  # Prompt for new background

        # Load the Stable Diffusion pipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting"
        ).to("cuda")  # Use GPU if available

        try:
            # Step 1: Remove the background
            print("Removing background...")
            subject_image = remove_background(input_image_path, remove_bg_api_key)
            subject_image.save("subject.png")  # Save for debugging

            # Step 2: Prepare a mask for the dress
            # Assuming you create a "mask.png" where white areas correspond to the dress
            mask_image = Image.open("mask.png").convert("RGB")

            # Step 3: Change the dress
            print("Changing the dress...")
            updated_subject = change_dress(pipe, subject_image, mask_image, dress_prompt)
            updated_subject.save("updated_subject.png")

            # Step 4: Generate a new background
            print("Generating new background...")
            background_image = generate_background(pipe, background_prompt)
            background_image.save("background.png")

            # Step 5: Combine the subject and the new background
            print("Combining subject and background...")
            background_image.paste(updated_subject, (0, 0), updated_subject)
            background_image.save("final_image.png")
            print("Enhanced image saved as 'final_image.png'.")

        except Exception as e:
            print(f"Error: {e}")

    if __name__ == "__main__":
        main()
