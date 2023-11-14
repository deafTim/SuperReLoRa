import torch


def smooth_restart_weights(model, alpha=0.1):
    """
    Applies a smooth restart to the weights of the given model.

    Parameters:
    model (torch.nn.Module): The neural network model whose weights are to be restarted.
    alpha (float): The blending factor for old and new weights. Default is 0.1.
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Generate new random weights
                new_weights = torch.randn_like(param)

                # Blend the old weights with new weights
                param.data = (1 - alpha) * param.data + alpha * new_weights
