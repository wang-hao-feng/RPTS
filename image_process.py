import base64
from io import BytesIO
from PIL import Image, ImageOps

def base64_images(images:list[Image.Image]) -> list[str]:
    def pil2base64(image:Image.Image):
        img_buffer = BytesIO()
        image.save(img_buffer, format='png')
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        return base64_str
    return [pil2base64(image) for image in images]

def padding_images(images:list[Image.Image]) -> list[Image.Image]:
    if len(images) < 2:
        return images
    max_w = max([image.size[0] for image in images])
    max_h = max([image.size[1] for image in images])
    return [ImageOps.pad(image, (max_w, max_h)) for image in images]

def concat_images(images:list[Image.Image]) -> list[Image.Image]:
    if len(images) < 2:
        return images
    total_w = sum([image.size[0] for image in images])
    max_h = max([image.size[1] for image in images])
    new_image = Image.new('RGB', (total_w, max_h))
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]
    return [new_image]