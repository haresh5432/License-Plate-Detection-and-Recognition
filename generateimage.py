
import os
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------
# Parameters
# ------------------------------
img_size = 32                # output image size
num_images_per_char = 100    # how many images per character
font_path = "/Users/hareshshokeen/Desktop/CV/LicensePlate.ttf" # path to your license plate TTF font

# Output folders
output_letters = "dataset_letters"
output_digits = "dataset_digits"
os.makedirs(output_letters, exist_ok=True)
os.makedirs(output_digits, exist_ok=True)

# ------------------------------
# Helper function to create character image
# ------------------------------
def create_char_image(char, font, size=img_size):
    img = Image.new('L', (size, size), color=255)  # white background
    draw = ImageDraw.Draw(img)
    
    # Use textbbox to get size of character
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # center text
    x = (size - w) // 2
    y = (size - h) // 2
    draw.text((x, y), char, font=font, fill=0)  # black text
    
    return np.array(img)

# ------------------------------
# Load font
# ------------------------------
font = ImageFont.truetype(font_path, 28)  # font size 28

# ------------------------------
# Generate letters A-Z
# ------------------------------
for char in string.ascii_uppercase:
    char_dir = os.path.join(output_letters, char)
    os.makedirs(char_dir, exist_ok=True)
    for i in range(num_images_per_char):
        img_array = create_char_image(char, font)
        # Save image
        img = Image.fromarray(img_array)
        img.save(os.path.join(char_dir, f"{char}_{i}.png"))

# ------------------------------
# Generate digits 0-9
# ------------------------------
for char in string.digits:
    char_dir = os.path.join(output_digits, char)
    os.makedirs(char_dir, exist_ok=True)
    for i in range(num_images_per_char):
        img_array = create_char_image(char, font)
        img = Image.fromarray(img_array)
        img.save(os.path.join(char_dir, f"{char}_{i}.png"))

print("✅ Dataset generated successfully!")
