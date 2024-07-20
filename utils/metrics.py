import torch

def mean_squared_error(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def mean_absolute_error(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def percentage_of_correct_keypoints(predictions, targets, threshold=0.05):
    distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))
    correct_keypoints = distances < threshold
    return torch.mean(correct_keypoints.float())

if __name__ == '__main__':
    # Test if the evaluation metric functions are working correctly
    pred = torch.tensor([[0.1, 0.1], [0.4, 0.4], [0.8, 0.8]])
    target = torch.tensor([[0.1, 0.2], [0.3, 0.3], [0.9, 0.9]])

    mse = mean_squared_error(pred, target)
    mae = mean_absolute_error(pred, target)
    pck = percentage_of_correct_keypoints(pred, target)

    print("MSE:", mse.item())
    print("MAE:", mae.item())
    print("PCK:", pck.item())