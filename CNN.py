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

# Định nghĩa tên các lớp cho tập dữ liệu CIFAR-10.
# Danh sách này sẽ được sử dụng để hiển thị nhãn trong các biểu đồ và dự đoán.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Thiết lập thiết bị để huấn luyện và suy luận.
# Mã này kiểm tra xem có GPU hỗ trợ CUDA nào khả dụng không; nếu không, nó sẽ mặc định sử dụng CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}") # In ra thiết bị đang được sử dụng

# Định nghĩa các phép biến đổi dữ liệu sẽ áp dụng cho ảnh CIFAR-10.
# transforms.ToTensor() chuyển đổi ảnh PIL hoặc numpy.ndarray thành PyTorch FloatTensor
# và chia tỷ lệ giá trị pixel về khoảng [0.0, 1.0].
# transforms.Normalize() chuẩn hóa tensor với giá trị trung bình và độ lệch chuẩn
# cho mỗi kênh. Các giá trị này là đặc trưng cho tập dữ liệu CIFAR-10.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Tải tập dữ liệu CIFAR-10.
# root='./data' chỉ định thư mục nơi tập dữ liệu sẽ được tải xuống.
# train=True tải tập huấn luyện, train=False tải tập kiểm tra.
# download=True tải tập dữ liệu nếu nó chưa có sẵn.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Chia tập huấn luyện thành tập huấn luyện và tập xác thực (validation).
# Điều này rất quan trọng để theo dõi hiệu suất của mô hình trên dữ liệu chưa thấy trong quá trình huấn luyện
# và để điều chỉnh siêu tham số.
num_train = len(trainset)
indices = list(range(num_train)) # Tạo danh sách các chỉ số
np.random.seed(42) # Đặt một seed ngẫu nhiên để có thể tái tạo kết quả
np.random.shuffle(indices) # Xáo trộn các chỉ số
train_split = int(0.8 * num_train) # 80% cho huấn luyện
train_idx, valid_idx = indices[:train_split], indices[train_split:] # Chia các chỉ số

# Định nghĩa các bộ tải dữ liệu (data loaders).
# Các bộ tải dữ liệu là các đối tượng có thể lặp qua tập dữ liệu và hỗ trợ chia lô (batching), xáo trộn (shuffling), và
# tải dữ liệu đa tiến trình.
# batch_size: Số lượng mẫu trong mỗi lô.
# sampler: Định nghĩa chiến lược để lấy mẫu từ tập dữ liệu.
# SubsetRandomSampler được sử dụng ở đây để lấy mẫu từ các chỉ số train_idx và valid_idx cụ thể.
# num_workers: Số lượng tiến trình con để tải dữ liệu. 2 là một giá trị phổ biến.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(train_idx), num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(valid_idx), num_workers=2)
# Đối với tập kiểm tra, shuffle được đặt thành False vì thứ tự không quan trọng cho việc đánh giá,
# và đó là một thực hành tốt để giữ cho nó nhất quán.
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# Định nghĩa lớp Mạng nơ-ron tích chập (CNN).
# Mô hình này là một CNN đơn giản với 3 lớp tích chập, theo sau là lớp gộp (pooling),
# và sau đó là hai lớp kết nối đầy đủ (fully connected) để phân loại.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Lớp tích chập 1: Đầu vào 3 kênh (RGB), đầu ra 16 kênh.
        # kernel_size=3 nghĩa là bộ lọc 3x3. padding=1 giữ nguyên kích thước không gian.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Lớp tích chập 2: Đầu vào 16 kênh, đầu ra 32 kênh.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Lớp tích chập 3: Đầu vào 32 kênh, đầu ra 64 kênh.
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Lớp Max Pooling: Giảm kích thước không gian đi 2x2.
        # kernel_size=2, stride=2 nghĩa là cửa sổ 2x2 với bước nhảy là 2.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Lớp kết nối đầy đủ 1: Kích thước đầu vào được tính dựa trên đầu ra của
        # lớp gộp cuối cùng (64 kênh * kích thước không gian 4x4).
        # Kích thước đầu ra 512.
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        # Lớp kết nối đầy đủ 2 (Lớp đầu ra): Đầu vào 512, đầu ra 10 (cho 10 lớp).
        self.fc2 = nn.Linear(512, 10)
        # Hàm kích hoạt ReLU (Rectified Linear Unit).
        self.relu = nn.ReLU()

    # Định nghĩa quá trình forward của mạng.
    # Điều này chỉ định cách dữ liệu chảy qua các lớp.
    def forward(self, x):
        # Conv1 -> ReLU -> Pool. Đầu vào (32x32) -> Đầu ra (16x16)
        x = self.pool(self.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool. Đầu vào (16x16) -> Đầu ra (8x8)
        x = self.pool(self.relu(self.conv2(x)))
        # Conv3 -> ReLU -> Pool. Đầu vào (8x8) -> Đầu ra (4x4)
        x = self.pool(self.relu(self.conv3(x)))
        # Làm phẳng (flatten) đầu ra của các lớp tích chập cho các lớp kết nối đầy đủ.
        # x.view(-1, ...) định hình lại tensor; -1 suy ra kích thước lô (batch size).
        x = x.view(-1, 64 * 4 * 4) # (batch_size, 1024)
        # Lớp kết nối đầy đủ 1 -> ReLU
        x = self.relu(self.fc1(x))
        # Lớp kết nối đầy đủ 2 (đầu ra)
        x = self.fc2(x)
        return x

# Hàm tính trung bình động để làm mịn các đường cong.
# Điều này giúp trực quan hóa xu hướng trong dữ liệu nhiễu như mất mát/độ chính xác của huấn luyện.
def moving_average(data, window_size):
    # np.convolve thực hiện tích chập. Chế độ 'valid' nghĩa là chỉ xuất ra kết quả
    # khi phép tích chập đầy đủ.
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Hàm để đánh giá hiệu suất của mô hình trên một bộ tải dữ liệu đã cho.
def evaluate(model, dataloader, criterion, device):
    model.eval() # Đặt mô hình ở chế độ đánh giá (tắt dropout, cập nhật batch normalization, v.v.)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    # torch.no_grad() tắt tính toán gradient, điều này không cần thiết cho đánh giá
    # và tiết kiệm bộ nhớ cũng như tính toán.
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device) # Chuyển dữ liệu đến thiết bị đã chỉ định
            outputs = model(images) # Quá trình forward
            loss = criterion(outputs, labels) # Tính toán mất mát
            running_loss += loss.item() # Tích lũy mất mát
            _, predicted = torch.max(outputs, 1) # Lấy lớp có xác suất cao nhất
            running_total += labels.size(0) # Tổng số mẫu
            running_correct += (predicted == labels).sum().item() # Số dự đoán đúng
    avg_loss = running_loss / len(dataloader) # Mất mát trung bình trên mỗi lô
    accuracy = 100 * running_correct / running_total # Độ chính xác theo phần trăm
    return avg_loss, accuracy

# Hàm chính để chạy quá trình huấn luyện và đánh giá.
def main():
    # Tạo một thư mục để lưu các tệp đầu ra (biểu đồ, ảnh).
    output_dir = 'cnn'
    os.makedirs(output_dir, exist_ok=True) # Tạo thư mục nếu nó chưa tồn tại

    # Khởi tạo mô hình CNN và chuyển nó đến thiết bị đã chọn.
    model = CNN().to(device)
    # Định nghĩa hàm mất mát (Cross-Entropy Loss cho phân loại đa lớp).
    criterion = nn.CrossEntropyLoss()
    # Định nghĩa bộ tối ưu hóa (Stochastic Gradient Descent với tốc độ học và động lượng).
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10 # Số epoch huấn luyện
    # Các danh sách để lưu trữ giá trị mất mát và độ chính xác để vẽ các đường cong học tập.
    train_losses, valid_losses, test_losses = [], [], []
    train_accuracies, valid_accuracies, test_accuracies = [], [], []
    # Các danh sách để lưu trữ số lượng mẫu đã thấy, để vẽ trên trục x.
    examples_seen, valid_examples_seen, test_examples_seen = [], [], []
    cumulative_examples = 0 # Bộ đếm tổng số mẫu đã thấy
    batch_counter = 0 # Bộ đếm tổng số lô đã xử lý

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        model.train() # Đặt mô hình ở chế độ huấn luyện
        running_train_loss = 0.0
        running_train_correct = 0
        running_train_total = 0
        num_batches = 0

        # Lặp qua các lô trong dữ liệu huấn luyện
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device) # Chuyển dữ liệu đến thiết bị
            optimizer.zero_grad() # Đặt lại gradient về 0 từ lần lặp trước
            outputs = model(images) # Quá trình forward
            loss = criterion(outputs, labels) # Tính toán mất mát
            loss.backward() # Lan truyền ngược: tính toán gradient
            optimizer.step() # Cập nhật các tham số của mô hình
            running_train_loss += loss.item() # Tích lũy mất mát huấn luyện
            _, predicted = torch.max(outputs, 1) # Lấy dự đoán
            running_train_total += labels.size(0) # Cập nhật tổng số mẫu
            running_train_correct += (predicted == labels).sum().item() # Cập nhật số dự đoán đúng
            num_batches += 1
            batch_counter += 1
            cumulative_examples += images.size(0) # Tăng số mẫu tích lũy
            train_losses.append(loss.item()) # Thêm mất mát lô hiện tại
            # Thêm độ chính xác huấn luyện hiện tại (được tính trên tổng số đúng/tổng số mẫu tích lũy theo lô)
            train_accuracies.append(100 * running_train_correct / running_train_total)
            examples_seen.append(cumulative_examples) # Ghi lại số mẫu đã thấy

            # Đánh giá trên tập xác thực và tập kiểm tra sau mỗi 100 lô.
            # Điều này cung cấp cái nhìn chi tiết hơn về hiệu suất của mô hình trong quá trình huấn luyện.
            if batch_counter % 100 == 0:
                valid_loss, valid_acc = evaluate(model, validloader, criterion, device)
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_acc)
                valid_examples_seen.append(cumulative_examples) # Ghi lại số mẫu tích lũy cho xác thực
                test_loss, test_acc = evaluate(model, testloader, criterion, device)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                test_examples_seen.append(cumulative_examples) # Ghi lại số mẫu tích lũy cho kiểm tra

        # Tính toán và in ra mất mát huấn luyện trung bình và độ chính xác cho epoch hiện tại.
        avg_train_loss = running_train_loss / num_batches
        avg_train_accuracy = 100 * running_train_correct / running_train_total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Mất mát huấn luyện trung bình: {avg_train_loss:.4f}, Độ chính xác huấn luyện: {avg_train_accuracy:.2f}%')

    # Áp dụng trung bình động để làm mịn các đường cong mất mát và độ chính xác của huấn luyện.
    window_size = 20 # Kích thước cửa sổ làm mịn
    smoothed_train_losses = moving_average(train_losses, window_size)
    smoothed_train_accuracies = moving_average(train_accuracies, window_size)
    # Điều chỉnh examples_seen để khớp với độ dài của dữ liệu đã làm mịn.
    smoothed_examples_seen = examples_seen[window_size - 1:]


    ## Vẽ Biểu đồ Học tập: Mất mát

    plt.figure(figsize=(10, 6)) # Tạo một hình mới
    plt.plot(smoothed_examples_seen, smoothed_train_losses, label='Mất mát huấn luyện (Đã làm mịn)', color='blue', linewidth=1.5)
    plt.plot(valid_examples_seen, valid_losses, 'go-', label='Mất mát xác thực', markersize=5, linewidth=1.0)
    plt.plot(test_examples_seen, test_losses, 'ro-', label='Mất mát kiểm tra', markersize=5, linewidth=1.0)
    plt.title('Đường cong học tập: Mất mát')
    plt.xlabel('Số lượng mẫu huấn luyện đã thấy')
    plt.ylabel('Mất mát Negative Log-Likelihood')
    plt.legend() # Hiển thị chú giải
    plt.grid(True, linestyle='--', alpha=0.7) # Thêm lưới để dễ đọc
    plt.tight_layout() # Điều chỉnh biểu đồ để tránh các nhãn chồng chéo
    plt.savefig(os.path.join(output_dir, 'learning_curves_loss.png')) # Lưu biểu đồ
    plt.close() # Đóng biểu đồ để giải phóng bộ nhớ


    ## Vẽ Biểu đồ Học tập: Độ chính xác

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_examples_seen, smoothed_train_accuracies, label='Độ chính xác huấn luyện (Đã làm mịn)', color='blue', linewidth=1.5)
    plt.plot(valid_examples_seen, valid_accuracies, 'go-', label='Độ chính xác xác thực', markersize=5, linewidth=1.0)
    plt.plot(test_examples_seen, test_accuracies, 'ro-', label='Độ chính xác kiểm tra', markersize=5, linewidth=1.0)
    plt.title('Đường cong học tập: Độ chính xác')
    plt.xlabel('Số lượng mẫu huấn luyện đã thấy')
    plt.ylabel('Độ chính xác (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_accuracy.png'))
    plt.close()


    ## Đánh giá và lưu ảnh dự đoán

    model.eval() # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad():
        dataiter = iter(testloader) # Lấy một iterator cho dữ liệu kiểm tra
        images, labels = next(dataiter) # Lấy lô ảnh và nhãn đầu tiên
        images, labels = images.to(device), labels.to(device) # Chuyển đến thiết bị
        outputs = model(images) # Lấy đầu ra của mô hình
        _, predicted = torch.max(outputs, 1) # Lấy lớp dự đoán cho mỗi ảnh

        print("\nCác dự đoán mẫu trên tập kiểm tra (5 ảnh đầu tiên):")
        for i in range(5): # Lặp qua 5 ảnh đầu tiên trong lô
            # In nhãn dự đoán so với nhãn thực tế
            print(f"Ảnh {i + 1}: Dự đoán: {classes[predicted[i]]}, Thực tế: {classes[labels[i]]}")
            # Chuyển đổi tensor ảnh PyTorch trở lại thành mảng NumPy để vẽ biểu đồ
            # Hoàn tác chuẩn hóa: (ảnh * std) + mean
            # Chuyển đổi kích thước từ (Kênh, Chiều cao, Chiều rộng) thành (Chiều cao, Chiều rộng, Kênh) cho matplotlib/PIL
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.247, 0.243, 0.261]) + np.array([0.4914, 0.4822, 0.4465])).clip(0, 1)
            img = (img * 255).astype(np.uint8) # Thay đổi tỷ lệ về 0-255 và chuyển đổi sang uint8
            pil_img = Image.fromarray(img) # Chuyển đổi mảng NumPy thành ảnh PIL
            pil_img = pil_img.resize((128, 128), Image.LANCZOS) # Thay đổi kích thước để dễ hình dung hơn
            pil_img.save(os.path.join(output_dir, f'pred_{i}.png')) # Lưu từng ảnh dự đoán riêng lẻ

        # Kết hợp 5 ảnh dự đoán thành một ảnh duy nhất theo chiều ngang.
        imgs = [Image.open(os.path.join(output_dir, f'pred_{i}.png')) for i in range(5)]
        widths, heights = zip(*(i.size for i in imgs)) # Lấy chiều rộng và chiều cao của các ảnh
        total_width = sum(widths) # Tính tổng chiều rộng cho ảnh mới
        max_height = max(heights) # Chiều cao tối đa cho ảnh mới
        new_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in imgs:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        new_img.save(os.path.join(output_dir, 'predicted_images.png')) # Lưu ảnh tổng hợp

    ## Đánh giá và vẽ ma trận nhầm lẫn

    # Lặp qua các bộ dữ liệu huấn luyện, xác thực và kiểm tra để tạo ma trận nhầm lẫn.
    for name, loader in [('train', trainloader), ('valid', validloader), ('test', testloader)]:
        loss, acc = evaluate(model, loader, criterion, device)
        print(f'{name.capitalize()} Mất mát: {loss:.4f}, {name.capitalize()} Độ chính xác: {acc:.2f}%')
        preds, labels_list = [], []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy()) # Thu thập các dự đoán
                labels_list.extend(labels.cpu().numpy()) # Thu thập các nhãn thực tế
        cm = confusion_matrix(labels_list, preds) # Tạo ma trận nhầm lẫn
        plt.figure(figsize=(10, 8)) # Tạo một hình mới cho ma trận nhầm lẫn
        # sns.heatmap vẽ biểu đồ nhiệt của ma trận nhầm lẫn
        # annot=True hiển thị giá trị trong các ô
        # fmt='d' định dạng số nguyên
        # cmap='Blues' sử dụng bảng màu xanh
        # xticklabels/yticklabels đặt nhãn trục x/y là tên các lớp
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Ma trận nhầm lẫn trên tập {name.capitalize()}')
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png')) # Lưu ma trận nhầm lẫn
        plt.close()

if __name__ == '__main__':
    main()