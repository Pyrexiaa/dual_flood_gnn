import torch.nn as nn
import torch

class WaterLevelMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class WaterLevelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.gru(x)
        return self.head(out[:, -1])  # (B, 1)

class JointGRU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.head_1d = nn.Linear(64, 1)
        self.head_2d = nn.Linear(64, 1)

    def forward(self, x, node_type):
        h, _ = self.gru(x)
        h_last = h[:, -1]

        y = torch.zeros(len(x), 1, device=x.device)

        mask_1d = node_type == 0
        mask_2d = node_type == 1

        if mask_1d.any():
            y[mask_1d] = self.head_1d(h_last[mask_1d])

        if mask_2d.any():
            y[mask_2d] = self.head_2d(h_last[mask_2d])

        return y

class TwoHeadGRU(nn.Module):
    """
    Two-head architecture for node-level features with shared GRU backbone.
    
    Since 1D and 2D samples are ALREADY PADDED to the same dimension (15 features),
    we use the SAME encoder for both, but separate prediction heads.
    
    Architecture:
    1. Shared feature encoder (handles padded features)
    2. Shared GRU backbone
    3. Separate prediction heads for 1D and 2D
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        """
        Args:
            input_dim: Feature dimension (same for 1D and 2D after padding, e.g., 15)
            hidden_dim: Hidden dimension for GRU
            num_layers: Number of GRU layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Shared input encoder (works for both 1D and 2D)
        # The padding ensures both have same dimension
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Shared GRU backbone
        self.gru = nn.GRU(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Separate output heads for 1D and 2D
        self.head_1d = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.head_2d = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, X, node_type):
        """
        Args:
            X: (batch, window, features) - padded to same dimension for all samples
            node_type: (batch,) - 0 for 1D, 1 for 2D
        
        Returns:
            predictions: (batch, 1)
        """
        batch_size, window_size, _ = X.shape
        device = X.device
        
        # Separate 1D and 2D samples
        mask_1d = (node_type == 0)
        mask_2d = (node_type == 1)
        
        # Initialize output
        output = torch.zeros((batch_size, 1), device=device)
        
        # Process 1D samples
        if mask_1d.any():
            X_1d = X[mask_1d]  # (B1, window, features)
            
            # Encode: process each timestep
            B1, W, F = X_1d.shape
            X_1d_flat = X_1d.reshape(B1 * W, F)
            encoded_1d = self.encoder(X_1d_flat)
            encoded_1d = encoded_1d.reshape(B1, W, self.hidden_dim)
            
            # GRU
            _, h_1d = self.gru(encoded_1d)
            h_1d = h_1d[-1]  # Take last layer: (B1, hidden_dim)
            
            # Predict with 1D head
            output[mask_1d] = self.head_1d(h_1d)
        
        # Process 2D samples
        if mask_2d.any():
            X_2d = X[mask_2d]  # (B2, window, features)
            
            # Encode: process each timestep
            B2, W, F = X_2d.shape
            X_2d_flat = X_2d.reshape(B2 * W, F)
            encoded_2d = self.encoder(X_2d_flat)
            encoded_2d = encoded_2d.reshape(B2, W, self.hidden_dim)
            
            # GRU
            _, h_2d = self.gru(encoded_2d)
            h_2d = h_2d[-1]  # Take last layer: (B2, hidden_dim)
            
            # Predict with 2D head
            output[mask_2d] = self.head_2d(h_2d)
        
        return output