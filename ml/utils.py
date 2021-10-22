import torch


def set_device(cuda: bool = True):
    """
    Set the device to cuda and default tensor types to FloatTensor on the device
    """
    # Set device
    device = torch.device("cuda" if (
        torch.cuda.is_available() and cuda) else "cpu")
    # Set default tensor types
    # torch.set_default_tensor_type("torch.FloatTensor")
    # if device.type == "cuda":
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device
