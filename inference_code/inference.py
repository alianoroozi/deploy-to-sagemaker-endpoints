import torch
from torchvision import transforms
import numpy as np
from six import BytesIO
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # move the model to GPU if available
    model = model.to(device)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/x-npy'
    image_npy = np.load(BytesIO(request_body))
    image = Image.fromarray(image_npy)
    input_object = pre_process(image)
    return input_object


def predict_fn(input_object, model):
    with torch.no_grad():
        output = model(input_object)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probs = torch.nn.functional.softmax(output[0], dim=0)
    return probs


def pre_process(image):
    """ Preprocess image and add a batch dimension
    Args:
        image: PIL Image
    """    
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = data_transforms(image)
    # add a batch dimension
    input_batch = input_tensor.unsqueeze(0) 
    # move the input to GPU if available
    input_batch = input_batch.to(device)
    return input_batch
