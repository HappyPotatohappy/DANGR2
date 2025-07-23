import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from DomainBed.domainbed import algorithms
    from DomainBed.domainbed.lib import misc
    from DomainBed.domainbed import networks
except ImportError:
    print("Warning: DomainBed not found. Running in standalone mode.")
    algorithms = None

from .networks import ImprovedNoiseGenerator, DomainClassifier, ImprovedClassClassifier, LabelSmoothingLoss

class DANGR(algorithms.Algorithm if algorithms else object):
    """
    Domain-Adversarial Noise Generation for Robust classification (DANGR)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        if algorithms:
            super(DANGR, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 하이퍼파라미터
        self.lambda_domain = hparams.get('lambda_domain', 0.5)
        self.lambda_class = hparams.get('lambda_class', 1.0)
        self.lambda_noise = hparams.get('lambda_noise', 0.1)
        self.lr = hparams.get('lr', 1e-4)
        self.lr_g = hparams.get('lr_g', 1e-4)
        self.weight_decay = hparams.get('weight_decay', 1e-5)
        
        # Feature extractor
        if algorithms and hasattr(networks, 'Featurizer'):
            self.featurizer = networks.Featurizer(input_shape, self.hparams)
            self.n_outputs = self.featurizer.n_outputs
        else:
            # Standalone mode: ResNet50 사용
            from torchvision import models
            resnet = models.resnet50(pretrained=True)
            self.featurizer = nn.Sequential(*list(resnet.children())[:-1])
            self.n_outputs = 2048
        
        # 네트워크 구성
        self.noise_generator = ImprovedNoiseGenerator(input_shape[0]).to(self.device)
        self.domain_classifier = DomainClassifier(num_domains, self.n_outputs).to(self.device)
        self.classifier = ImprovedClassClassifier(num_classes, self.n_outputs).to(self.device)
        
        # 손실 함수
        self.criterion_domain = nn.CrossEntropyLoss()
        self.criterion_class = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
        
        # 옵티마이저 설정
        self.optimizer_G = optim.Adam(
            self.noise_generator.parameters(),
            lr=self.lr_g,
            weight_decay=self.weight_decay
        )
        self.optimizer_D = optim.Adam(
            self.domain_classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.optimizer_C = optim.Adam(
            list(self.classifier.parameters()) + list(self.featurizer.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 스케줄러
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=50)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=50)
        self.scheduler_C = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_C, T_max=50)
    
    def update(self, minibatches, unlabeled=None):
        """DomainBed 호환 업데이트 함수"""
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.long, device=self.device)
            for i, (x, y) in enumerate(minibatches)
        ])
        
        # 노이즈 생성
        delta = self.noise_generator(all_x)
        x_tilde = torch.clamp(all_x + delta, 0, 1)
        
        # (1) 도메인 분류기 업데이트
        self.optimizer_D.zero_grad()
        features = self.featurizer(x_tilde.detach())
        features = features.view(features.size(0), -1)
        domain_preds = self.domain_classifier(features)
        loss_D = self.criterion_domain(domain_preds, all_d)
        loss_D.backward()
        self.optimizer_D.step()
        
        # (2) 클래스 분류기 업데이트
        self.optimizer_C.zero_grad()
        features = self.featurizer(x_tilde.detach())
        features = features.view(features.size(0), -1)
        class_preds = self.classifier(features)
        loss_C = self.criterion_class(class_preds, all_y)
        loss_C.backward()
        self.optimizer_C.step()
        
        # (3) 노이즈 생성기 업데이트
        self.optimizer_G.zero_grad()
        features_G = self.featurizer(x_tilde)
        features_G = features_G.view(features_G.size(0), -1)
        
        # 도메인 적대적 손실
        domain_preds_G = self.domain_classifier(features_G)
        loss_domain_G = self.criterion_domain(domain_preds_G, all_d)
        
        # 클래스 손실
        class_preds_G = self.classifier(features_G)
        loss_class_G = self.criterion_class(class_preds_G, all_y)
        
        # 노이즈 정규화
        loss_noise = self.lambda_noise * torch.mean(
            torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
        )
        
        # 전체 생성기 손실
        loss_G = -self.lambda_domain * loss_domain_G + \
                  self.lambda_class * loss_class_G + \
                  loss_noise
        
        loss_G.backward()
        self.optimizer_G.step()
        
        return {
            'loss': loss_C.item(),
            'loss_domain': loss_D.item(),
            'loss_generator': loss_G.item(),
            'loss_noise': loss_noise.item()
        }
    
    def predict(self, x):
        """예측 함수"""
        # 테스트 시에도 노이즈 생성
        delta = self.noise_generator(x)
        x_tilde = torch.clamp(x + delta, 0, 1)
        
        features = self.featurizer(x_tilde)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def step_schedulers(self):
        """학습률 스케줄러 스텝"""
        self.scheduler_G.step()
        self.scheduler_D.step()
        self.scheduler_C.step()
    
    def get_state_dict(self):
        """모델 상태 반환"""
        return {
            'featurizer': self.featurizer.state_dict(),
            'noise_generator': self.noise_generator.state_dict(),
            'domain_classifier': self.domain_classifier.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_C': self.optimizer_C.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """모델 상태 로드"""
        self.featurizer.load_state_dict(state_dict['featurizer'])
        self.noise_generator.load_state_dict(state_dict['noise_generator'])
        self.domain_classifier.load_state_dict(state_dict['domain_classifier'])
        self.classifier.load_state_dict(state_dict['classifier'])
        if 'optimizer_G' in state_dict:
            self.optimizer_G.load_state_dict(state_dict['optimizer_G'])
            self.optimizer_D.load_state_dict(state_dict['optimizer_D'])
            self.optimizer_C.load_state_dict(state_dict['optimizer_C'])

# DomainBed에 알고리즘 등록
if algorithms:
    algorithms.ALGORITHMS['DANGR'] = DANGR
