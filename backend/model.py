"""
Hybrid Deep Learning Model for Stroke Detection
Combines ResNet18 (CT Image) + GCN (EEG Time Series)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EEGGCNBlock(nn.Module):
    """Graph Convolutional Network for EEG Time Series Feature Extraction"""
    
    def __init__(self, in_channels, out_channels, num_nodes=19):
        super(EEGGCNBlock, self).__init__()
        self.num_nodes = num_nodes
        
        # Graph adjacency matrix (simplified - based on EEG electrode positions)
        self.register_buffer('adj', self._create_adjacency_matrix(num_nodes))
        
        # Graph convolution layers
        self.gc1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.gc2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.3)
        
    def _create_adjacency_matrix(self, num_nodes):
        """Create a simple adjacency matrix for EEG electrodes"""
        # Using a fully connected approach with distance-based weights
        adj = torch.ones(num_nodes, num_nodes)
        return adj
    
    def graph_conv(self, x, adj):
        """Perform graph convolution"""
        # Add self-connections
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        # Normalize adjacency matrix
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Graph convolution operation
        return torch.matmul(adj_normalized, x)
    
    def forward(self, x):
        # x shape: (batch, channels, num_nodes, time_steps)
        batch_size = x.size(0)
        
        # Reshape for graph conv: (batch, num_nodes, channels * time_steps)
        x = x.mean(-1)  # Average over time
        x = x.permute(0, 2, 1)  # (batch, num_nodes, channels)
        
        # Apply graph convolution
        x = self.graph_conv(x, self.adj)
        
        # Transpose and apply conv layers
        x = x.permute(0, 2, 1)  # (batch, channels, num_nodes)
        x = F.relu(self.bn1(self.gc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.gc2(x)))
        
        # Global pooling
        x = x.mean(-1)  # (batch, out_channels)
        
        return x


class HybridStrokeModel(nn.Module):
    """
    Hybrid Model combining:
    - ResNet18 for CT Image feature extraction
    - GCN for EEG Time Series feature extraction
    - Fusion layer for final classification
    """
    
    def __init__(self, num_classes=3, eeg_channels=19, eeg_timepoints=1000):
        super(HybridStrokeModel, self).__init__()
        
        self.num_classes = num_classes
        
        # CT Image Feature Extractor using ResNet18
        resnet = models.resnet18(pretrained=False)
        self.ct_features = nn.Sequential(*list(resnet.children())[:-1])
        ct_feature_dim = 512
        
        # EEG Feature Extractor using GCN
        self.eeg_gcn = EEGGCNBlock(in_channels=eeg_channels, out_channels=256, num_nodes=eeg_channels)
        eeg_feature_dim = 256
        
        # Fusion Layer
        fused_dim = ct_feature_dim + eeg_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification Head
        self.classifier = nn.Linear(128, num_classes)
        
        # Class names
        self.class_names = ['No Stroke', 'Hemorrhagic Stroke', 'Ischemic Stroke']
        
    def forward_ct(self, ct_image):
        """Extract features from CT image"""
        # ct_image shape: (batch, 3, 224, 224)
        features = self.ct_features(ct_image)
        features = features.view(features.size(0), -1)
        return features
    
    def forward_eeg(self, eeg_data):
        """Extract features from EEG data"""
        # eeg_data shape: (batch, num_channels, time_points)
        # Reshape to (batch, channels, num_nodes, time_steps)
        batch_size = eeg_data.size(0)
        num_channels = eeg_data.size(1)
        time_points = eeg_data.size(2)
        
        # Reshape for GCN
        eeg_input = eeg_data.unsqueeze(2)  # (batch, channels, 1, time)
        eeg_input = eeg_input.expand(-1, -1, num_channels, -1)  # (batch, channels, num_nodes, time)
        
        features = self.eeg_gcn(eeg_input)
        return features
    
    def forward(self, ct_image, eeg_data):
        """Forward pass through the hybrid model"""
        # Extract CT features
        ct_features = self.forward_ct(ct_image)
        
        # Extract EEG features
        eeg_features = self.forward_eeg(eeg_data)
        
        # Fusion
        fused_features = torch.cat([ct_features, eeg_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output
    
    def predict(self, ct_image, eeg_data):
        """Make predictions and return class, confidence"""
        self.eval()
        with torch.no_grad():
            output = self.forward(ct_image, eeg_data)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            class_name = self.class_names[predicted_class.item()]
            confidence_pct = confidence.item() * 100
            
        return class_name, confidence_pct, probabilities


def create_model(num_classes=3):
    """Create and return the hybrid stroke detection model"""
    model = HybridStrokeModel(num_classes=num_classes)
    return model


def load_trained_model(model_path=None, num_classes=3):
    """Load a trained model or create a new one with random weights"""
    model = create_model(num_classes=num_classes)
    
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded trained model from {model_path}")
        except Exception as e:
            print(f"Could not load model from {model_path}, using untrained model: {e}")
    
    model.eval()
    return model


# Mock prediction function for demonstration
def mock_predict():
    """Generate mock predictions for demonstration purposes"""
    import random
    
    classes = ['No Stroke', 'Hemorrhagic Stroke', 'Ischemic Stroke']
    selected_class = random.choice(classes)
    
    if selected_class == 'No Stroke':
        confidence = random.uniform(85, 98)
    else:
        confidence = random.uniform(88, 99)
    
    # Mock metrics
    metrics = {
        'precision': random.uniform(90, 96),
        'recall': random.uniform(88, 95),
        'f1_score': random.uniform(0.89, 0.95)
    }
    
    return selected_class, confidence, metrics


if __name__ == "__main__":
    # Test model creation
    model = create_model(num_classes=3)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass with dummy data
    batch_size = 1
    ct_image = torch.randn(batch_size, 3, 224, 224)
    eeg_data = torch.randn(batch_size, 19, 1000)
    
    output = model(ct_image, eeg_data)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

