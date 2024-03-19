import torch
import torch.nn as nn
import torchvision.models as models

class AbstractModel(nn.Module):
    def __init__(self, input_shape):
        super(AbstractModel, self).__init__()
        self.input_shape = input_shape

    def forward(self, x):
        pass

class ModelV0(AbstractModel):
    def __init__(self, input_shape):
        super(ModelV0, self).__init__(input_shape)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 32 * 12 * 12)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        return x

class ModelV1(AbstractModel):
    def __init__(self, input_shape):
        super(ModelV1, self).__init__(input_shape)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 12 * 12)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        return x

class ModelV2(AbstractModel):
    def __init__(self, input_shape):
        super(ModelV2, self).__init__(input_shape)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 10 * 10)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        return x

class ModelV3(AbstractModel):
    def __init__(self, input_shape):
        super(ModelV3, self).__init__(input_shape)
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128 * 7 * 7, 3072)
        self.fc2 = nn.Linear(3072, 2048)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        return x

class ModelV4(AbstractModel):
    def __init__(self, input_shape):
        super(ModelV4, self).__init__(input_shape)
        self.conv1 = nn.Conv2d(input_shape[0], 96, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool3(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        return x

class ModelV21(AbstractModel):
    def __init__(self, input_shape):
        super(ModelV21, self).__init__(input_shape)
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        return x.view(x.size(0), -1)

class ModelV22(AbstractModel):
    def __init__(self, input_shape):
        super(ModelV22, self).__init__(input_shape)
        self.base_model = models.nasnet.NASNetLarge(pretrained=True)
        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        return x.view(x.size(0), -1)
