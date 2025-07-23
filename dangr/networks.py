import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedNoiseGenerator(nn.Module):
    def __init__(self, input_channels=3):
        super(ImprovedNoiseGenerator, self).__init__()
        # 인코더
        self.enc_conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # 디코더
        self.dec_conv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.dec_conv1 = nn.Conv2d(128, input_channels, 3, stride=1, padding=1)
        
        # 정규화 및 활성화
        self.bn_enc1 = nn.BatchNorm2d(64)
        self.bn_enc2 = nn.BatchNorm2d(128)
        self.bn_enc3 = nn.BatchNorm2d(256)
        self.bn_dec3 = nn.BatchNorm2d(128)
        self.bn_dec2 = nn.BatchNorm2d(64)
        
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # 인코더
        e1 = self.relu(self.bn_enc1(self.enc_conv1(x)))
        e2 = self.relu(self.bn_enc2(self.enc_conv2(e1)))
        e3 = self.relu(self.bn_enc3(self.enc_conv3(e2)))
        
        # 디코더 (스킵 커넥션 포함)
        d3 = self.relu(self.bn_dec3(self.dec_conv3(e3)))
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.relu(self.bn_dec2(self.dec_conv2(d3)))
        d2 = torch.cat([d2, e1], dim=1)
        
        # 최종 출력은 작은 노이즈를 생성하도록 tanh에 스케일링 적용
        delta = self.tanh(self.dec_conv1(d2)) * 0.1
        return delta

class DomainClassifier(nn.Module):
    def __init__(self, num_domains, input_dim):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_domains)
        )
    
    def forward(self, x):
        return self.classifier(x.view(x.size(0), -1))

class ImprovedClassClassifier(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(ImprovedClassClassifier, self).__init__()
        
        # 배치 정규화 추가
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
        # 잔차 연결(Residual connection)을 위한 추가 레이어
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 평탄화
        main_path = self.classifier(x)
        shortcut_path = self.shortcut(x)
        
        # 잔차 연결 적용
        return main_path + shortcut_path

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
