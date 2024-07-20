import os
import torch
from torch import nn

# General helper functions
def save_model(model, optimizer, epoch, loss, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    },path)

def load_model(model, optimizer, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

if __name__ == "__main__":
    # Test
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epoch = 5
    loss = 0.25
    save_path = "checkpoint/test_model.pth"

    save_model(model, optimizer, epoch, loss, save_path)
    print(f"Model loaded from {save_path}")

    loaded_model = DummyModel()
    loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.01)
    loaded_model, loaded_optimizer, loaded_epoch, loaded_loss = load_model(loaded_model, loaded_optimizer, save_path)

    print(f"Model loaded from {save_path}")
    print(f"Epoch: {loaded_epoch}, Loss: {loaded_loss}")