import numpy as np
from PIL import Image
import torch

def torch2png(data,filename):
    #data: [c,h,w]
    data=data.cpu()
    data_stretch = truncated_linear_stretch(data.numpy().astype(np.float32), 2)
    im = Image.fromarray(np.transpose(data_stretch,(1,2,0)))
    im.save(filename)

def truncated_linear_stretch(image, truncated_value=2, max_out=255, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if (max_out <= 255):
            gray = np.uint8(gray)
        elif (max_out <= 65535):
            gray = np.uint16(gray)
        return gray

    #  如果是多波段
    if (len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch

# img=torch.randn(3,512,512)
# filename='try2.png'
# torch2png(img,filename)
