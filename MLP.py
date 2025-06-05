import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

num_train = len(trainset)
indices = list(range(num_train))
np.random.seed(42)
np.random.shuffle(indices)
train_split = int(0.8 * num_train)
train_idx, valid_idx = indices[:train_split], indices[train_split:]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(train_idx), num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(valid_idx), num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * running_correct / running_total
    return avg_loss, accuracy

def main():
    output_dir = 'mlp'
    os.makedirs(output_dir, exist_ok=True)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    train_losses, valid_losses, test_losses = [], [], []
    train_accuracies, valid_accuracies, test_accuracies = [], [], []
    examples_seen, valid_examples_seen, test_examples_seen = [], [], []
    cumulative_examples = 0
    batch_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0
        running_train_total = 0
        num_batches = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_train_total += labels.size(0)
            running_train_correct += (predicted == labels).sum().item()
            num_batches += 1
            batch_counter += 1
            cumulative_examples += images.size(0)
            train_losses.append(loss.item())
            train_accuracies.append(100 * running_train_correct / running_train_total)
            examples_seen.append(cumulative_examples)

            if batch_counter % 100 == 0:
                valid_loss, valid_acc = evaluate(model, validloader, criterion, device)
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_acc)
                valid_examples_seen.append(cumulative_examples)
                test_loss, test_acc = evaluate(model, testloader, criterion, device)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                test_examples_seen.append(cumulative_examples)

        avg_train_loss = running_train_loss / num_batches
        avg_train_accuracy = 100 * running_train_correct / running_train_total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.2f}%')

    window_size = 20
    smoothed_train_losses = moving_average(train_losses, window_size)
    smoothed_train_accuracies = moving_average(train_accuracies, window_size)
    smoothed_examples_seen = examples_seen[window_size - 1:]

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_examples_seen, smoothed_train_losses, label='Train Loss (Smoothed)', color='blue', linewidth=1.5)
    plt.plot(valid_examples_seen, valid_losses, 'go-', label='Validation Loss', markersize=5, linewidth=1.0)
    plt.plot(test_examples_seen, test_losses, 'ro-', label='Test Loss', markersize=5, linewidth=1.0)
    plt.title('Learning Curves: Loss')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Negative Log-Likelihood Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_examples_seen, smoothed_train_accuracies, label='Train Accuracy (Smoothed)', color='blue', linewidth=1.5)
    plt.plot(valid_examples_seen, valid_accuracies, 'go-', label='Validation Accuracy', markersize=5, linewidth=1.0)
    plt.plot(test_examples_seen, test_accuracies, 'ro-', label='Test Accuracy', markersize=5, linewidth=1.0)
    plt.title('Learning Curves: Accuracy')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_accuracy.png'))
    plt.close()

    model.eval()
    with torch.no_grad():
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print("\nSample Test Predictions (First 5 Images):")
        for i in range(5):
            print(f"Image {i + 1}: Predicted: {classes[predicted[i]]}, Actual: {classes[labels[i]]}")
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.247, 0.243, 0.261]) + np.array([0.4914, 0.4822, 0.4465])).clip(0, 1)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((128, 128), Image.LANCZOS)
            pil_img.save(os.path.join(output_dir, f'pred_{i}.png'))

        imgs = [Image.open(os.path.join(output_dir, f'pred_{i}.png')) for i in range(5)]
        widths, heights = zip(*(i.size for i in imgs))
        total_width = sum(widths)
        max_height = max(heights)
        new_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in imgs:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        new_img.save(os.path.join(output_dir, 'predicted_images.png'))

    for name, loader in [('train', trainloader), ('valid', validloader), ('test', testloader)]:
        loss, acc = evaluate(model, loader, criterion, device)
        print(f'{name.capitalize()} Loss: {loss:.4f}, {name.capitalize()} Accuracy: {acc:.2f}%')
        preds, labels_list = [], []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        cm = confusion_matrix(labels_list, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix on {name.capitalize()} Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png'))
        plt.close()

if __name__ == '__main__':
    main()