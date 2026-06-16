import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from PIL import Image
import base64
import io

TRAINING_DIR = "training"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset_records(path):
    with open(path, "r") as f:
        dataset = json.load(f)
    records = []
    for index, item in enumerate(dataset):
        records.append({
            "image": item["image"],
            "concentration": float(item["concentration"]),
            "filename": item.get("filename", f"sample_{index}.jpg"),
            "source_id": item.get("source_id", item.get("filename", f"sample_{index}.jpg")),
        })
    return records


def decode_image(image_data):
    """Decode base64 image to PIL Image."""
    if isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raise ValueError("Unsupported image format")


class ConcentrationDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = decode_image(record["image"])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(record["concentration"], dtype=torch.float32)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip()
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model(backbone="efficientnet_b0", freeze_backbone=True):
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

    elif backbone == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "r2": float(r2_score(all_labels, all_preds)),
        "rmse": float(mean_squared_error(all_labels, all_preds) ** 0.5),
        "mae": float(mean_absolute_error(all_labels, all_preds)),
    }


def train_cnn(records, backbone="efficientnet_b0", epochs_frozen=10, epochs_unfrozen=20,
              batch_size=16, lr_frozen=1e-3, lr_unfrozen=1e-4):

    # Split by source_id to avoid leakage
    source_ids = list(set(r["source_id"] for r in records))
    train_ids, holdout_ids = train_test_split(source_ids, test_size=0.2, random_state=42)
    train_ids = set(train_ids)
    holdout_ids = set(holdout_ids)

    train_records = [r for r in records if r["source_id"] in train_ids]
    holdout_records = [r for r in records if r["source_id"] in holdout_ids]

    train_transform, val_transform = get_transforms()
    train_dataset = ConcentrationDataset(train_records, transform=train_transform)
    holdout_dataset = ConcentrationDataset(holdout_records, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size)

    print(f"\nTraining samples: {len(train_records)}, Holdout samples: {len(holdout_records)}")
    print(f"Device: {DEVICE}")

    # Phase 1 — frozen backbone, train head only
    print(f"\n--- Phase 1: Frozen backbone ({epochs_frozen} epochs) ---")
    model = build_model(backbone=backbone, freeze_backbone=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen)
    criterion = nn.MSELoss()

    for epoch in range(epochs_frozen):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images).squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, holdout_loader)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} | "
                  f"holdout R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")

    # Phase 2 — unfreeze and fine-tune
    print(f"\n--- Phase 2: Full fine-tuning ({epochs_unfrozen} epochs) ---")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_unfrozen)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_unfrozen)

    for epoch in range(epochs_unfrozen):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images).squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            train_metrics = evaluate(model, train_loader)
            holdout_metrics = evaluate(model, holdout_loader)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} | "
                  f"train R²={train_metrics['r2']:.4f} | "
                  f"holdout R²={holdout_metrics['r2']:.4f}, RMSE={holdout_metrics['rmse']:.4f}")

    # Final evaluation
    train_metrics = evaluate(model, train_loader)
    holdout_metrics = evaluate(model, holdout_loader)
    r2_gap = train_metrics["r2"] - holdout_metrics["r2"]

    print(f"\n{'='*60}")
    print(f"Final Results — {backbone}")
    print(f"  Train R²:   {train_metrics['r2']:.4f}")
    print(f"  Holdout R²: {holdout_metrics['r2']:.4f}")
    print(f"  Holdout RMSE: {holdout_metrics['rmse']:.4f}")
    print(f"  Holdout MAE:  {holdout_metrics['mae']:.4f}")
    print(f"  R² gap (overfit): {r2_gap:.4f}")

    return model, train_metrics, holdout_metrics


if __name__ == "__main__":
    import sys
    candidates = [
        os.path.join(TRAINING_DIR, "new_dataset.json"),
        os.path.join(TRAINING_DIR, "processed_image_data.json"),
    ]
    input_file = next((p for p in candidates if os.path.exists(p)), None)
    if not input_file:
        print("No dataset found.")
        sys.exit(1)

    print(f"Using: {input_file}")
    records = load_dataset_records(input_file)

    # Try both backbones
    for backbone in ["efficientnet_b0", "resnet18"]:
        train_cnn(records, backbone=backbone)