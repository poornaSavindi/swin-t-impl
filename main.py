# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from swin_transformer import SwinTransformer, swin_t, swin_s, swin_b, swin_l
from PIL import Image
from torchvision import transforms
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # img = Image.open("1.jpg")
    # transform = transforms.ToTensor()
    # tensor_image = transform(img)

    # dummy_x = torch.randn(1, 3, 224, 224)

    image_path = "1.jpg"  # Replace with the path to your image file
    image = Image.open(image_path)

    # Define the transformation to convert the image to a PyTorch tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the desired size
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    # Apply the transformation
    tensor_image = transform(image)

    # Add an extra batch dimension
    tensor_image = torch.unsqueeze(tensor_image, 0)


    swin_t = swin_t()
    print(swin_t(tensor_image))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
