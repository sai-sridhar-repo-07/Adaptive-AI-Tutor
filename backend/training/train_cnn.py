import torch
import torch.nn as nn
import torch.optim as optim

from backend.app.models.cnn_model import DigitCNN
from backend.training.dataset import get_dataloaders


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / len(train_loader)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:.2f}%)\n"
    )

    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders(batch_size=64)

    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            epoch
        )
        test(model, device, test_loader, criterion)

    # Save trained model
    torch.save(model.state_dict(), "digit_cnn.pth")
    print("✅ Model saved as digit_cnn.pth")


if __name__ == "__main__":
    main()
