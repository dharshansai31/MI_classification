import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from model_resnet1d import HighAccuracyECGNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3
PATIENCE = 12

DATA_DIR = "data/processed_cv"
OUT_DIR = "outputs_cv"

os.makedirs(OUT_DIR, exist_ok=True)


class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def add_noise(self, x, noise_std=0.01):
        noise = np.random.normal(0, noise_std, size=x.shape).astype(np.float32)
        return x + noise

    def scale_amplitude(self, x, low=0.9, high=1.1):
        scale = np.random.uniform(low, high)
        return x * scale

    def random_shift(self, x, max_shift=50):
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, shift=shift, axis=1)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.augment:
            if np.random.rand() < 0.5:
                x = self.add_noise(x)
            if np.random.rand() < 0.5:
                x = self.scale_amplitude(x)
            if np.random.rand() < 0.5:
                x = self.random_shift(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def build_loaders(X, y, folds, test_fold):
    val_fold = 1 if test_fold == 10 else test_fold + 1

    test_mask = folds == test_fold
    val_mask = folds == val_fold
    train_mask = ~(test_mask | val_mask)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_folds = [f for f in range(1, 11) if f not in [test_fold, val_fold]]
    print(f"\nFold setup: train={train_folds} val={val_fold} test={test_fold}")
    print("Train shape:", X_train.shape, "class counts:", np.bincount(y_train))
    print("Val shape  :", X_val.shape, "class counts:", np.bincount(y_val))
    print("Test shape :", X_test.shape, "class counts:", np.bincount(y_test))

    train_ds = ECGDataset(X_train, y_train, augment=True)
    val_ds = ECGDataset(X_val, y_val, augment=False)
    test_ds = ECGDataset(X_test, y_test, augment=False)

    class_sample_count = np.bincount(y_train)
    weights = 1.0 / class_sample_count
    sample_weights = weights[y_train]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion):
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()

            p = torch.argmax(out, dim=1)
            preds.extend(p.cpu().numpy())
            targets.extend(y.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    acc = accuracy_score(targets, preds)
    overall_f1 = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )[2]

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=[1], average=None, zero_division=0
    )

    avg_loss = total_loss / max(1, len(loader))
    return {
        "loss": avg_loss,
        "acc": acc,
        "overall_f1": float(overall_f1),
        "mi_precision": float(precision[0]),
        "mi_recall": float(recall[0]),
        "mi_f1": float(f1[0]),
        "cm": confusion_matrix(targets, preds).tolist()
    }


def train_one_fold(X, y, folds, test_fold):
    train_loader, val_loader, test_loader = build_loaders(X, y, folds, test_fold)

    model = HighAccuracyECGNet(in_channels=12, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_mi_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, criterion)

        scheduler.step(val_metrics["mi_f1"])

        print(
            f"Fold {test_fold} | Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val Acc={val_metrics['acc']:.4f} | "
            f"MI Prec={val_metrics['mi_precision']:.4f} | "
            f"MI Rec={val_metrics['mi_recall']:.4f} | "
            f"MI F1={val_metrics['mi_f1']:.4f}"
        )

        if val_metrics["mi_f1"] > best_mi_f1:
            best_mi_f1 = val_metrics["mi_f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(OUT_DIR, f"best_model_fold_{test_fold}.pth"))
            patience_counter = 0
            print(f"Saved best model for fold {test_fold}.")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping fold {test_fold}.")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion)
    return test_metrics


def main():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    folds = np.load(os.path.join(DATA_DIR, "folds.npy"))

    results = []

    for test_fold in range(1, 11):
        print("\n" + "=" * 80)
        print(f"STARTING FOLD {test_fold}")
        print("=" * 80)

        fold_result = train_one_fold(X, y, folds, test_fold)
        fold_result["test_fold"] = test_fold
        results.append(fold_result)

        print(f"\nFold {test_fold} TEST METRICS:")
        print(json.dumps(fold_result, indent=2))

    accs = np.array([r["acc"] for r in results])
    mi_precs = np.array([r["mi_precision"] for r in results])
    mi_recs = np.array([r["mi_recall"] for r in results])
    mi_f1s = np.array([r["mi_f1"] for r in results])

    summary = {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "mi_precision_mean": float(mi_precs.mean()),
        "mi_precision_std": float(mi_precs.std()),
        "mi_recall_mean": float(mi_recs.mean()),
        "mi_recall_std": float(mi_recs.std()),
        "mi_f1_mean": float(mi_f1s.mean()),
        "mi_f1_std": float(mi_f1s.std()),
        "per_fold_results": results
    }

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    with open(os.path.join(OUT_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_DIR, "cv_summary.txt"), "w") as f:
        f.write("CROSS-VALIDATION SUMMARY\n")
        f.write(f"Accuracy Mean ± Std     : {summary['accuracy_mean']:.6f} ± {summary['accuracy_std']:.6f}\n")
        f.write(f"MI Precision Mean ± Std : {summary['mi_precision_mean']:.6f} ± {summary['mi_precision_std']:.6f}\n")
        f.write(f"MI Recall Mean ± Std    : {summary['mi_recall_mean']:.6f} ± {summary['mi_recall_std']:.6f}\n")
        f.write(f"MI F1 Mean ± Std        : {summary['mi_f1_mean']:.6f} ± {summary['mi_f1_std']:.6f}\n")

    print(f"\nSaved CV summary to {os.path.join(OUT_DIR, 'cv_summary.json')}")


if __name__ == "__main__":
    main()
