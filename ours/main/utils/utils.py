import numpy as np
import random
import os
import torch
import yaml
from easydict import EasyDict
import torch.optim as optim
import torch.nn.functional as F

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def compute_minimal_perturbation(model, loader, device='cuda'):
    """
    Compute the minimal perturbation in the right-bottom 6x6 region to change the model's output.

    Args:
    - model (nn.Module): The PyTorch model.
    - loader (DataLoader): DataLoader for the input images.
    - device (str): Device to run the model on, default is 'cuda'.

    Returns:
    - perturbation (torch.Tensor): The minimal perturbation applied to the right-bottom 6x6 region.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Get the first batch of images and labels from the DataLoader
    num = 0
    trigger_effective_radius = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if num > 100: 
            break
        
        # Ensure the input is a 4D tensor (batch size, channels, height, width)
        if images.ndimension() == 3:
            images = images.unsqueeze(0)
        
        for idx in range(images.size(0)):
            image = images[idx].unsqueeze(0)

            # Save the original output for comparison
            original_output = model(image)
            original_label = torch.argmax(original_output, dim=1) 
            if original_label.item() != 0:
                continue

            # Define the region for perturbation (right-bottom 6x6 area)
            height, width = image.shape[2], image.shape[3]
            perturbation_mask = torch.zeros_like(image)
            perturbation_mask[:, :, height-6:, width-6:] = 1  # Right-bottom 6x6 mask

            # Initialize the perturbation tensor with small random values
            perturbation = torch.zeros_like(image, requires_grad=True).to(device)

            # Define an optimizer for the perturbation
            optimizer = optim.Adam([perturbation], lr=0.01)

            # Start optimizing the perturbation
            output_label = original_label
            count = 0
            while original_label == output_label:
                count += 1
                if count > 200:  # some image's origin label is 0, and after bd is 0 too
                    break
                optimizer.zero_grad()

                # Add the perturbation to the image
                perturbed_image = image + perturbation * perturbation_mask                

                # Pass the perturbed image through the model
                perturbed_output = model(perturbed_image)
                output_label = torch.argmax(perturbed_output, dim=1)

                # Calculate the loss as the difference between original and perturbed output
                loss = - F.cross_entropy(perturbed_output, original_label)

                # Backpropagate the gradients
                loss.backward()

                # Update the perturbation
                optimizer.step()

            # print(torch.sum(perturbation * perturbation_mask).item())
            if count < 200:
                trigger_effective_radius += torch.sum(torch.abs(perturbed_image - image)).item()
                num += 1
    return trigger_effective_radius / num