import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_pytorch.simple_vit import SimpleViT
import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def main(args):
    num_models = args.num_models
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_steps = args.num_steps
    image_size = args.image_size
    patch_size = args.patch_size
    num_classes = args.num_classes
    dim = args.dim
    dim_head = args.dim_head
    depth = args.depth
    heads = args.heads
    mlp_dim = args.mlp_dim 

    start = args.start
    end = args.end if args.end is not None else start + num_models

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    save_dir = "./data/cifar10_vit/vit_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    stat_path = "./data/cifar10_vit/stat"
    if not os.path.exists(stat_path):
        os.makedirs(stat_path)
    
    vit_hyperparameters = {
        "image_size": image_size,
        "patch_size": patch_size,
        "num_classes": num_classes,
        "dim": dim,
        "dim_head": dim_head,
        "depth": depth,
        "heads": heads,
        "mlp_dim": mlp_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_steps": num_steps
    }

    hyperparams_path = os.path.join(stat_path, f"vit_hyperparameters_{start}_{end}.json")
    with open(hyperparams_path, "w") as f:
        json.dump(vit_hyperparameters, f, indent=4)
    with open(os.path.join(stat_path, "vit_hyperparameters.json"), "w") as f:
        json.dump(vit_hyperparameters, f, indent=4)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root="./data/pytorch_ds", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="./data/pytorch_ds", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()

    def test(model, loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        return 100. * correct / total

    results = []
    for i in range(start, end):
        model = SimpleViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            dim_head=dim_head,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=3
        ).to(device)

        if i == 0:
            print(model)

        print(f"Training model {i + 1}/{end}...")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        selected_epochs = [0, 1, 2, 3, 5, 10, 13, 18, 23]
        end_epoch = selected_epochs[i%len(selected_epochs)]

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()


            if epoch == end_epoch:
                train_acc = test(model, train_loader, device)
                test_acc = test(model, test_loader, device)
                model_path = os.path.join(save_dir, f"model_{i + 1}_{epoch}.pth")
                torch.save(model.state_dict(), model_path)
                results.append({"model_id": i + 1, "epoch": epoch, "train_accuracy": train_acc, "test_accuracy": test_acc})
                print(f"[Model {i + 1} epoch {epoch}], Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")
                
                break

        del model
        torch.cuda.empty_cache()

    results_path = os.path.join(stat_path, f"results_{start}_{end}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(stat_path, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    model_ids = [item["model_id"] for item in results]
    train_end = int(0.8 * len(model_ids))
    val_end = int(0.9 * len(model_ids))
    splits = {
        "train": model_ids[:train_end],
        "val": model_ids[train_end:val_end],
        "test": model_ids[val_end:],
    }
    with open(os.path.join(stat_path, "splits.json"), "w") as f:
        json.dump(splits, f, indent=4)

    print(f"All models trained and saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train multiple ViT models on CIFAR-10")
    parser.add_argument("--num-models", "--num_models", dest="num_models", type=int, default=1000, help="Number of models to train")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128, help="Batch size for training and testing")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=25, help="Number of epochs per model")
    parser.add_argument("--num-steps", "--num_steps", dest="num_steps", type=int, default=0, help="Number of steps per model")
    parser.add_argument('--device', type=str, default='0')

    parser.add_argument("--image-size", "--image_size", dest="image_size", type=int, default=32, help="Resized image size")
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=4, help="Patch size for the ViT model")
    parser.add_argument("--num-classes", "--num_classes", dest="num_classes", type=int, default=10, help="Number of classes (default is CIFAR-10)")
    parser.add_argument("--dim", type=int, default=32, help="Dimensionality of the embedding in ViT")
    parser.add_argument("--dim-head", "--dim_head", dest="dim_head", type=int, default=16, help="Dimensionality of the head in ViT")
    parser.add_argument("--depth", type=int, default=2, help="Number of transformer blocks in ViT")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads in ViT")
    parser.add_argument("--mlp-dim", "--mlp_dim", dest="mlp_dim", type=int, default=64, help="Dimension of the MLP in ViT")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    
    args = parser.parse_args()
    main(args)
