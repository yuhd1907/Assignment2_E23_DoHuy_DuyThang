import torch # Thư viện PyTorch để xây dựng và huấn luyện mô hình học sâu
import torch.nn as nn # Chứa các lớp module cho mạng nơ-ron
import torch.optim as optim # Chứa các thuật toán tối ưu hóa (ví dụ: SGD, Adam)
import torchvision # Chứa các tập dữ liệu và mô hình phổ biến cho thị giác máy tính
import torchvision.transforms as transforms # Chứa các phép biến đổi dữ liệu
import matplotlib.pyplot as plt # Thư viện để tạo biểu đồ và đồ thị
from torch.utils.data import SubsetRandomSampler # Dùng để lấy mẫu ngẫu nhiên một tập con của dữ liệu
import numpy as np # Thư viện cho các phép toán số học
from sklearn.metrics import confusion_matrix # Dùng để tạo ma trận nhầm lẫn
import seaborn as sns # Thư viện để tạo biểu đồ thống kê đẹp mắt, dựa trên matplotlib
import os # Thư viện để tương tác với hệ điều hành (ví dụ: tạo thư mục)
from PIL import Image # Thư viện Python Imaging Library để xử lý ảnh

# Định nghĩa các lớp (nhãn) trong tập dữ liệu CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Xác định thiết bị sẽ sử dụng để huấn luyện (GPU nếu có, nếu không thì CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Định nghĩa các phép biến đổi sẽ áp dụng cho ảnh
transform = transforms.Compose([
    transforms.ToTensor(), # Chuyển đổi ảnh PIL Image hoặc numpy.ndarray sang torch.FloatTensor và chia tỷ lệ các giá trị pixel trong khoảng [0.0, 1.0]
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # Chuẩn hóa các kênh màu (R, G, B) với giá trị trung bình và độ lệch chuẩn
])

# Tải tập dữ liệu CIFAR-10
# root='./data': thư mục nơi lưu trữ dữ liệu
# train=True: tải tập huấn luyện
# download=True: tải dữ liệu nếu chưa có
# transform=transform: áp dụng các phép biến đổi đã định nghĩa
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Chia tập huấn luyện thành tập huấn luyện và tập xác thực
num_train = len(trainset) # Tổng số mẫu trong tập huấn luyện
indices = list(range(num_train)) # Tạo danh sách các chỉ số
np.random.seed(42) # Đặt seed cho bộ tạo số ngẫu nhiên để đảm bảo kết quả có thể tái lập
np.random.shuffle(indices) # Xáo trộn ngẫu nhiên các chỉ số
train_split = int(0.8 * num_train) # 80% cho tập huấn luyện
train_idx, valid_idx = indices[:train_split], indices[train_split:] # Chia các chỉ số

# Tạo DataLoader cho tập huấn luyện, xác thực và kiểm tra
# batch_size: số lượng mẫu trong mỗi batch
# sampler: dùng SubsetRandomSampler để lấy mẫu từ các chỉ số đã chia
# num_workers: số lượng quy trình con để tải dữ liệu
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(train_idx), num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(valid_idx), num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2) # shuffle=False vì không cần xáo trộn khi kiểm tra

# Định nghĩa mô hình Multilayer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__() # Gọi hàm khởi tạo của lớp cha (nn.Module)
        self.flatten = nn.Flatten() # Lớp làm phẳng đầu vào (biến ảnh 32x32x3 thành một vector dài)
        self.network = nn.Sequential( # Định nghĩa chuỗi các lớp trong mạng
            nn.Linear(32 * 32 * 3, 512), # Lớp kết nối đầy đủ đầu tiên: 32*32*3 (đầu vào ảnh) -> 512 nơ-ron
            nn.ReLU(), # Hàm kích hoạt ReLU (Rectified Linear Unit)
            nn.Linear(512, 10) # Lớp kết nối đầy đủ thứ hai: 512 nơ-ron -> 10 nơ-ron (cho 10 lớp đầu ra)
        )

    def forward(self, x):
        x = self.flatten(x) # Làm phẳng đầu vào
        x = self.network(x) # Chuyển đầu vào qua mạng
        return x # Trả về đầu ra của mạng

# Hàm tính trung bình động để làm mịn dữ liệu (ví dụ: loss, accuracy)
def moving_average(data, window_size):
    # np.convolve thực hiện tích chập, 'valid' nghĩa là chỉ các điểm mà cửa sổ hoàn toàn nằm trong dữ liệu
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Hàm đánh giá mô hình trên một tập dữ liệu
def evaluate(model, dataloader, criterion, device):
    model.eval() # Đặt mô hình ở chế độ đánh giá (tắt dropout, batchnorm, v.v.)
    running_loss = 0.0 # Tổng loss
    running_correct = 0 # Tổng số dự đoán đúng
    running_total = 0 # Tổng số mẫu
    with torch.no_grad(): # Không tính toán gradient trong quá trình đánh giá để tiết kiệm bộ nhớ và tăng tốc
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device) # Chuyển ảnh và nhãn đến thiết bị (GPU/CPU)
            outputs = model(images) # Dự đoán đầu ra
            loss = criterion(outputs, labels) # Tính toán loss
            running_loss += loss.item() # Cộng dồn loss của batch
            _, predicted = torch.max(outputs, 1) # Lấy lớp có xác suất cao nhất
            running_total += labels.size(0) # Cộng dồn số lượng mẫu
            running_correct += (predicted == labels).sum().item() # Cộng dồn số dự đoán đúng
    avg_loss = running_loss / len(dataloader) # Tính loss trung bình
    accuracy = 100 * running_correct / running_total # Tính độ chính xác
    return avg_loss, accuracy # Trả về loss trung bình và độ chính xác

# Hàm chính để chạy quá trình huấn luyện và đánh giá
def main():
    output_dir = 'mlp' # Thư mục để lưu kết quả
    os.makedirs(output_dir, exist_ok=True) # Tạo thư mục nếu nó chưa tồn tại

    model = MLP().to(device) # Khởi tạo mô hình MLP và chuyển nó đến thiết bị
    criterion = nn.CrossEntropyLoss() # Định nghĩa hàm mất mát (Cross-Entropy Loss cho bài toán phân loại)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Định nghĩa thuật toán tối ưu hóa SGD (Stochastic Gradient Descent)
                                                                    # lr: learning rate, momentum: động lượng

    num_epochs = 10 # Số lượng epoch để huấn luyện
    train_losses, valid_losses, test_losses = [], [], [] # Danh sách lưu trữ loss cho từng tập
    train_accuracies, valid_accuracies, test_accuracies = [], [], [] # Danh sách lưu trữ độ chính xác cho từng tập
    examples_seen, valid_examples_seen, test_examples_seen = [], [], [] # Danh sách lưu trữ số lượng ví dụ đã thấy
    cumulative_examples = 0 # Biến đếm tổng số ví dụ đã thấy
    batch_counter = 0 # Biến đếm số batch đã xử lý

    for epoch in range(num_epochs): # Lặp qua từng epoch
        model.train() # Đặt mô hình ở chế độ huấn luyện
        running_train_loss = 0.0 # Loss tích lũy trong epoch
        running_train_correct = 0 # Số dự đoán đúng tích lũy trong epoch
        running_train_total = 0 # Tổng số mẫu tích lũy trong epoch
        num_batches = 0 # Số lượng batch trong epoch

        for images, labels in trainloader: # Lặp qua từng batch trong trainloader
            images, labels = images.to(device), labels.to(device) # Chuyển dữ liệu đến thiết bị
            optimizer.zero_grad() # Đặt gradient về 0 cho mỗi batch
            outputs = model(images) # Thực hiện forward pass
            loss = criterion(outputs, labels) # Tính toán loss
            loss.backward() # Thực hiện backward pass (tính toán gradient)
            optimizer.step() # Cập nhật trọng số của mô hình

            running_train_loss += loss.item() # Cộng dồn loss của batch
            _, predicted = torch.max(outputs, 1) # Lấy lớp dự đoán
            running_train_total += labels.size(0) # Cộng dồn số mẫu
            running_train_correct += (predicted == labels).sum().item() # Cộng dồn số dự đoán đúng
            num_batches += 1 # Tăng số batch
            batch_counter += 1 # Tăng tổng số batch đã xử lý
            cumulative_examples += images.size(0) # Cộng dồn số ví dụ đã thấy
            train_losses.append(loss.item()) # Lưu loss của batch huấn luyện
            train_accuracies.append(100 * running_train_correct / running_train_total) # Lưu độ chính xác của batch huấn luyện
            examples_seen.append(cumulative_examples) # Lưu số ví dụ đã thấy

            # Đánh giá trên tập xác thực và kiểm tra sau mỗi 100 batch
            if batch_counter % 100 == 0:
                valid_loss, valid_acc = evaluate(model, validloader, criterion, device)
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_acc)
                valid_examples_seen.append(cumulative_examples) # Lưu số ví dụ đã thấy khi đánh giá
                test_loss, test_acc = evaluate(model, testloader, criterion, device)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                test_examples_seen.append(cumulative_examples) # Lưu số ví dụ đã thấy khi đánh giá

        # In kết quả trung bình của epoch
        avg_train_loss = running_train_loss / num_batches
        avg_train_accuracy = 100 * running_train_correct / running_train_total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.2f}%')

    # Vẽ biểu đồ Learning Curves: Loss
    window_size = 20 # Kích thước cửa sổ cho trung bình động
    smoothed_train_losses = moving_average(train_losses, window_size) # Làm mịn loss huấn luyện
    smoothed_train_accuracies = moving_average(train_accuracies, window_size) # Làm mịn độ chính xác huấn luyện
    smoothed_examples_seen = examples_seen[window_size - 1:] # Điều chỉnh số ví dụ đã thấy cho dữ liệu đã làm mịn

    plt.figure(figsize=(10, 6)) # Tạo figure mới
    plt.plot(smoothed_examples_seen, smoothed_train_losses, label='Train Loss (Smoothed)', color='blue', linewidth=1.5) # Vẽ loss huấn luyện đã làm mịn
    plt.plot(valid_examples_seen, valid_losses, 'go-', label='Validation Loss', markersize=5, linewidth=1.0) # Vẽ loss xác thực
    plt.plot(test_examples_seen, test_losses, 'ro-', label='Test Loss', markersize=5, linewidth=1.0) # Vẽ loss kiểm tra
    plt.title('Learning Curves: Loss') # Tiêu đề biểu đồ
    plt.xlabel('Number of Training Examples Seen') # Nhãn trục X
    plt.ylabel('Negative Log-Likelihood Loss') # Nhãn trục Y
    plt.legend() # Hiển thị chú giải
    plt.grid(True, linestyle='--', alpha=0.7) # Hiển thị lưới
    plt.tight_layout() # Tự động điều chỉnh khoảng cách giữa các phần tử để tránh chồng chéo
    plt.savefig(os.path.join(output_dir, 'learning_curves_loss.png')) # Lưu biểu đồ
    plt.close() # Đóng biểu đồ

    # Vẽ biểu đồ Learning Curves: Accuracy
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

    # Hiển thị và lưu một số ví dụ dự đoán
    model.eval() # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad(): # Tắt tính toán gradient
        dataiter = iter(testloader) # Tạo một iterator từ testloader
        images, labels = next(dataiter) # Lấy một batch ảnh và nhãn
        images, labels = images.to(device), labels.to(device) # Chuyển dữ liệu đến thiết bị
        outputs = model(images) # Dự đoán đầu ra
        _, predicted = torch.max(outputs, 1) # Lấy lớp dự đoán
        print("\nSample Test Predictions (First 5 Images):") # In tiêu đề
        for i in range(5): # Lặp qua 5 ảnh đầu tiên
            print(f"Image {i + 1}: Predicted: {classes[predicted[i]]}, Actual: {classes[labels[i]]}") # In dự đoán và nhãn thực tế
            # Chuyển đổi tensor ảnh về định dạng hiển thị được
            img = images[i].cpu().numpy().transpose(1, 2, 0) # Chuyển từ (C, H, W) sang (H, W, C)
            # Hoàn tác chuẩn hóa để hiển thị ảnh gốc
            img = (img * np.array([0.247, 0.243, 0.261]) + np.array([0.4914, 0.4822, 0.4465])).clip(0, 1)
            img = (img * 255).astype(np.uint8) # Chuyển về 0-255 và kiểu uint8
            pil_img = Image.fromarray(img) # Tạo ảnh PIL từ numpy array
            pil_img = pil_img.resize((128, 128), Image.LANCZOS) # Thay đổi kích thước ảnh để dễ nhìn hơn
            pil_img.save(os.path.join(output_dir, f'pred_{i}.png')) # Lưu ảnh dự đoán

        # Ghép 5 ảnh dự đoán lại thành một ảnh duy nhất
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

    # Tính toán và vẽ ma trận nhầm lẫn cho tập huấn luyện, xác thực và kiểm tra
    for name, loader in [('train', trainloader), ('valid', validloader), ('test', testloader)]:
        loss, acc = evaluate(model, loader, criterion, device) # Đánh giá mô hình
        print(f'{name.capitalize()} Loss: {loss:.4f}, {name.capitalize()} Accuracy: {acc:.2f}%') # In loss và accuracy

        preds, labels_list = [], [] # Danh sách lưu trữ dự đoán và nhãn thực tế
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy()) # Chuyển dự đoán về CPU và lưu
                labels_list.extend(labels.cpu().numpy()) # Chuyển nhãn về CPU và lưu
        cm = confusion_matrix(labels_list, preds) # Tính toán ma trận nhầm lẫn
        plt.figure(figsize=(10, 8)) # Tạo figure mới
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes) # Vẽ heatmap của ma trận nhầm lẫn
        plt.title(f'Confusion Matrix on {name.capitalize()} Set') # Tiêu đề biểu đồ
        plt.xlabel('Predicted') # Nhãn trục X
        plt.ylabel('Actual') # Nhãn trục Y
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png')) # Lưu ma trận nhầm lẫn
        plt.close() # Đóng biểu đồ

if __name__ == '__main__':
    main() # Gọi hàm main khi script được chạy