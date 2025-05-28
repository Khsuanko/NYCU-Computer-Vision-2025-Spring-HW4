import os
import torch
import numpy as np
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

def run_inference(model_path, test_dir, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    image_dict = {}
    filenames = sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0]))

    for filename in filenames:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(test_dir, filename)
        image_tensor = load_image(path)
        restored_tensor = restore_image(model, image_tensor, device)

        # Convert to uint8 (3, H, W)
        restored_np = (restored_tensor.numpy() * 255).astype(np.uint8)
        image_dict[filename] = restored_np

    np.savez(output_path, **image_dict)
    print(f"Saved {len(image_dict)} restored images to {output_path}")

# === Run ===
if __name__ == '__main__':
    run_inference(
        model_path='best_model.pth',
        test_dir='data/test/degraded',
        output_path='pred.npz'
    )
