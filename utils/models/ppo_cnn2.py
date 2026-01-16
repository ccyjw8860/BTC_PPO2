import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Custom1DCNN(BaseFeaturesExtractor):
    """
    Sequence Length 1000에 최적화된 1D CNN
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[1] 
        
        self.cnn = nn.Sequential(
            # Layer 1 (1000 -> 500)
            nn.Conv1d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            
            # Layer 2 (500 -> 250)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            
            # Layer 3 (250 -> 125)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 

            # 🟢 [추가됨] Layer 4 (125 -> 62) : 1000개니까 한 번 더 압축!
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
        )

        # (이하 동일) 차원 계산 로직이 자동으로 처리하므로 그대로 두시면 됩니다.
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample = sample.permute(0, 2, 1)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    # forward 함수는 그대로
    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.permute(0, 2, 1)
        x = self.cnn(x)
        return self.linear(x)