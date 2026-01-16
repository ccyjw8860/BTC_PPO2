import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 1. 커스텀 1D CNN 클래스 정의
class Custom1DCNN(BaseFeaturesExtractor):
    """
    Time Series용 1D CNN Feature Extractor
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 입력 차원: (Batch, Seq_Len, N_Features) -> (Batch, N_Features, Seq_Len)으로 변환해 처리
        n_input_channels = observation_space.shape[1] # 59개 Feature
        
        self.cnn = nn.Sequential(
            # Layer 1: 세밀한 특징 추출
            nn.Conv1d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 100 -> 50
            
            # Layer 2: 조금 더 큰 패턴 추출
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 50 -> 25
            
            # Layer 3: 추세 등 거시적 특징 추출
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 25 -> 12
            
            nn.Flatten(),
        )

        # CNN 통과 후 차원 계산
        with th.no_grad():
            # 더미 데이터 생성 (Batch=1, Seq=100, Feat=59)
            sample = th.as_tensor(observation_space.sample()[None]).float()
            # PyTorch Conv1d는 (Batch, Channel, Length) 순서를 원함 -> Permute
            sample = sample.permute(0, 2, 1)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # PPO 입력 (Batch, Seq, Feat) -> (Batch, Feat, Seq)로 순서 변경
        x = observations.permute(0, 2, 1)
        x = self.cnn(x)
        return self.linear(x)