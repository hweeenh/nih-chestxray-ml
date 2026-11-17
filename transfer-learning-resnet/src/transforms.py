from torchvision import transforms

# ImageNet normalisation parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_preprocessing_transform():
    """
    Returns deterministic preprocessing for chest X-ray images.
    
    Steps:
    1. Resize images to 224 x 224 pixels.
    2. Convert grayscale images to 3 channels.
    3. Convert images to PyTorch tensors.
    4. Apply ImageNet normalisation.

    Returns:
        torchvision.transforms.Compose: A composition of the preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])