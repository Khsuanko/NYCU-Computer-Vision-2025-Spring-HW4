import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from model import EnhancedUNet


def load_image(path):
    image = Image.open(path).convert('RGB')
    transform = T.ToTensor()
    return transform(image)


def restore_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor).squeeze(0).cpu().clamp(0, 1)
    return output


def visualize_comparison(degraded, restored, clean, title):
    images = [degraded, restored, clean]
    titles = ['Degraded', 'Restored', 'Clean']

    # Convert tensors to numpy arrays (H, W, C)
    images = [img.numpy().transpose(1, 2, 0) for img in images]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=16)
        ax.axis('off')

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedUNet().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))

    # Images to visualize
    images_to_visualize = ['rain-1.png', 'snow-1.png']

    for img_name in images_to_visualize:
        degraded_path = os.path.join('data/train/degraded', img_name)

        # Clean image names are like: rain_clean-1.png and snow_clean-1.png
        if img_name.startswith('rain'):
            clean_img_name = img_name.replace('rain-', 'rain_clean-')
        elif img_name.startswith('snow'):
            clean_img_name = img_name.replace('snow-', 'snow_clean-')
        else:
            raise ValueError(f"Unexpected image name: {img_name}")

        clean_path = os.path.join('data/train/clean', clean_img_name)

        # Load images
        degraded_tensor = load_image(degraded_path)
        clean_tensor = load_image(clean_path)

        # Restore
        restored_tensor = restore_image(model, degraded_tensor, device)

        # Visualize
        visualize_comparison(degraded_tensor, restored_tensor, clean_tensor, title=img_name)



if __name__ == '__main__':
    main()
