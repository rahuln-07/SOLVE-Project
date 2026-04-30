# PURE CNN
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import classification_report, accuracy_score

# 1. Define the End-to-End Model
class EndToEndGeoResNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super(EndToEndGeoResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer for 6 bands
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(in_channels, original_conv1.out_channels,
                                      kernel_size=original_conv1.kernel_size,
                                      stride=original_conv1.stride,
                                      padding=original_conv1.padding, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))

        # Modify the final layer to predict exactly 2 classes
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 2. Training Loop
def train_pure_cnn(X_train, y_train, X_test, y_test, epochs=25):
    print("--- Training Pure End-to-End ResNet ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndGeoResNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=32, shuffle=True)

    model.train()
    for ep in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
        print(f"Epoch {ep+1:02d} | Train Acc: {100*correct/total:.2f}%")

    # Evaluation
    print("\n--- Evaluating Pure CNN ---")
    model.eval()
    test_tensor = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        test_preds = model(test_tensor).argmax(1).cpu().numpy()

    print(classification_report(y_test, test_preds, target_names=['Not Suitable', 'Suitable']))
    print(f"Pure CNN Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    return model

# Run it! (Make sure X_p_train and y_train are already in memory from your previous script)
pure_cnn_model = train_pure_cnn(X_p_train, y_train, X_p_test, y_test)