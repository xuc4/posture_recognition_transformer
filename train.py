import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.pose_transformer import PoseTransformer
from utils.data_loader import get_data_loaders
from utils.metrics import mean_squared_error, mean_absolute_error, percentage_of_correct_keypoints
from utils.utils import save_model, load_model

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        keypoints = batch['keypoints'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(keypoints)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * keypoints.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, device):
    model.eval()
    mse, mae, pck = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            keypoints = batch['keypoints'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(keypoints)
            mse += mean_squared_error(outputs, labels).item() * keypoints.size(0)
            mae += mean_absolute_error(outputs, labels).item() * keypoints.size(0)
            pck += percentage_of_correct_keypoints(outputs, labels).item() * keypoints.size(0)

    mse /= len(dataloader.dataset)
    mae /= len(dataloader.dataset)
    pck /= len(dataloader.dataset)
    return mse, mae, pck

def main():
    dataset_dir = 'dataset'
    dataset_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    if not dataset_names:
        raise ValueError("No dataset found in the 'dataset' directory")

    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        data_path = os.path.join('data', f'{dataset_name}_processed_data.pkl')
        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        train_loader = get_data_loaders(dataset_name, batch_size=32, shuffle=True, num_workers=4)
        val_loader = get_data_loaders(dataset_name, batch_size=32, shuffle=False, num_workers=4)

        # Initialize the model
        model = PoseTransformer().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 25
        best_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_mse, val_mae, val_pck = evaluate(model, val_loader, device)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}, Val PCK: {val_pck:.4f}')

            if val_mse < best_loss:
                best_loss = val_mse
                save_model(model, optimizer, epoch, best_loss, os.path.join(checkpoint_dir, f'best_model_{dataset_name}.pth'))

        print(f'Training complete for dataset: {dataset_name}')

if __name__ == '__main__':
    main()
