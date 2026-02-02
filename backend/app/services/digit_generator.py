import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

class DigitGenerator:
    def generate(self, digit: int, count: int = 5):
        images = []

        for _ in range(count):
            img = Image.new("L", (280, 280), color=0)
            draw = ImageDraw.Draw(img)

            # Random font size & position
            font_size = random.randint(140, 200)
            x_offset = random.randint(40, 80)
            y_offset = random.randint(20, 80)

            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/Arial.ttf",
                    font_size
                )
            except:
                font = ImageFont.load_default()

            draw.text(
                (x_offset, y_offset),
                str(digit),
                fill=255,
                font=font
            )

            # Random blur + noise
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

            noise = np.random.randint(0, 25, (280, 280), dtype="uint8")
            img = Image.fromarray(
                np.clip(np.array(img) + noise, 0, 255).astype("uint8")
            )

            images.append(img)

        return images
