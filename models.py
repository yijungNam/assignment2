
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm

# CIFAR-10 클래스 레이블
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_cifar10_loaders(batch_size=128):
    # 학습용 transform: 데이터 증강 포함
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # 테스트용 transform: 증강 없음
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


def build_resnet50_cifar10():

    model = resnet50(weights=None)  

    # CIFAR-10의 작은 이미지(32x32)에 맞게 첫 레이어 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_model(model, train_loader, test_loader, epochs=30, lr=0.1, weight_decay=5e-4,
                model_name="model", save_path=None):

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    print(f"\n=== {model_name} 학습 시작 (epochs={epochs}, lr={lr}) ===")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        acc = evaluate_model(model, test_loader)
        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}, test_acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved: {save_path} (acc={best_acc:.2f}%)")

    print(f"=== {model_name} 학습 완료. Best acc: {best_acc:.2f}% ===\n")
    return model


def evaluate_model(model, loader):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def load_or_train_models(model_a_path='model_a.pth', model_b_path='model_b.pth',
                         force_retrain=False, quick_train=False):

    train_loader, test_loader = get_cifar10_loaders()
    epochs = 5 if quick_train else 30 

    model_a = build_resnet50_cifar10()
    model_b = build_resnet50_cifar10()

    if os.path.exists(model_a_path) and not force_retrain:
        print(f"Model A 로드: {model_a_path}")
        model_a.load_state_dict(torch.load(model_a_path, map_location=DEVICE))
        model_a = model_a.to(DEVICE)
    else:
        print("Model A 학습 시작 (lr=0.1, wd=5e-4)...")
        model_a = train_model(
            model_a, train_loader, test_loader,
            epochs=epochs, lr=0.1, weight_decay=5e-4,
            model_name="Model A", save_path=model_a_path
        )

    if os.path.exists(model_b_path) and not force_retrain:
        print(f"Model B 로드: {model_b_path}")
        model_b.load_state_dict(torch.load(model_b_path, map_location=DEVICE))
        model_b = model_b.to(DEVICE)
    else:
        print("Model B 학습 시작 (lr=0.05, wd=1e-3)...")
        model_b = train_model(
            model_b, train_loader, test_loader,
            epochs=epochs, lr=0.05, weight_decay=1e-3,
            model_name="Model B", save_path=model_b_path
        )

    model_a.eval()
    model_b.eval()
    return model_a, model_b
