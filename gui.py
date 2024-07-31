import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PyQt5.QtGui import QPixmap, QImage
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load('model.pth'))  
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Load CIFAR-10 dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Transformation for displaying the image (no normalization)
display_transform = transforms.Compose([
    transforms.ToTensor()
])


class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Image Classification with PyQt and CIFAR-10'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        layout = QVBoxLayout()
        
        self.imageLabel = QLabel(self)
        self.imageLabel.setFixedSize(256, 256)
        self.imageLabel.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.imageLabel)
        
        self.originalLabel = QLabel(self)
        layout.addWidget(self.originalLabel)
        
        self.predictionLabel = QLabel(self)
        layout.addWidget(self.predictionLabel)
        
        hbox = QHBoxLayout()
        
        self.imageSelect = QComboBox(self)
        self.imageSelect.addItems([f"Image {i}" for i in range(len(testset))])
        self.imageSelect.currentIndexChanged.connect(self.load_image)
        hbox.addWidget(self.imageSelect)
        
        self.predictButton = QPushButton('Predict', self)
        self.predictButton.clicked.connect(self.predict)
        hbox.addWidget(self.predictButton)
        
        layout.addLayout(hbox)
        
        self.setLayout(layout)
        self.show()
        
        self.load_image(0)
    
    def load_image(self, index):
        # Get the image and label from the dataset
        image, label = testset[index]
        image = image.permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Rescale image for display
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        
        self.imagePath = image
        self.originalLabelText = class_names[label]
        
        # Display the image
        qimage = QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), aspectRatioMode=True))
        self.originalLabel.setText(f'Original Label: {self.originalLabelText}')
        self.predictionLabel.setText('')
    
    def predict(self):
        # Preprocess the image
        image = self.imagePath
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        image = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = class_names[predicted.item()]
            self.predictionLabel.setText(f'Prediction: {prediction}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
