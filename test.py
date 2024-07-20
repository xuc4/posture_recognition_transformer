import os
import torch
from models.pose_transformer import PoseTransformer
from utils.data_loader import get_data_loaders
from utils.metrics import mean_squared_error, mean_absolute_error, percentage_of_correct_keypoints
from utils.utils import load_model

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
        raise ValueError("No datasets found in the 'dataset' directory.")

    for dataset_name in dataset_names:
        print(f"Evaluating dataset: {dataset_name}")
        data_path = os.path.join('data', f'{dataset_name}_processed_data.pkl')
        checkpoint_path = os.path.join('checkpoints', f'best_model_{dataset_name}.pth')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        test_loader = get_data_loaders(dataset_name, batch_size=32, shuffle=False, num_workers=4)

        # Initial model
        model = PoseTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Load model
        model, optimizer, epoch, loss = load_model(model, optimizer, checkpoint_path)

        # Evaluate model
        mse, mae, pck = evaluate(model, test_loader, device)

        print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, PCK: {pck:.4f}')

if __name__ == '__main__':
    main()
