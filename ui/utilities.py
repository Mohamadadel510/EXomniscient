## THIS FILE IS INTENDED TO MAKE THE APP FILE MORE CLEARER BY MAKING STATIC FUNCTION HERE
import torch
import torch.nn as nn
import torch.optim as optim


class ExoplanetCNN(nn.Module):
    def __init__(self, global_length=2001, local_length=201):
        super(ExoplanetCNN, self).__init__()

        # Global branch
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
        )

        # Local branch
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Compute FC input size dynamically
        with torch.no_grad():
            # Create dummy inputs with correct shapes
            g_dummy = torch.zeros(1, 1, global_length)
            l_dummy = torch.zeros(1, 1, local_length)

            g_out = self.global_conv(g_dummy)
            l_out = self.local_conv(l_dummy)

            g_feat = g_out.view(1, -1).size(1)
            l_feat = l_out.view(1, -1).size(1)
            in_features = g_feat + l_feat

        print(f"Global conv output features: {g_feat}")
        print(f"Local conv output features: {l_feat}")
        print(f"Total FC input features: {in_features}")

        # Fully connected layers (reduced complexity)
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, global_x, local_x):
        g = self.global_conv(global_x)
        l = self.local_conv(local_x)

        g = g.view(g.size(0), -1)
        l = l.view(l.size(0), -1)

        x = torch.cat((g, l), dim=1)
        return self.fc(x)


def add_noise(x, noise_level=0.01):
    """Add Gaussian noise to input tensor"""
    noise = torch.randn_like(x) * noise_level
    return x + noise

