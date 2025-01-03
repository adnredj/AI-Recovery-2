import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ..utils.logging import setup_logger

class WalletPatternRecognizer(nn.Module):
    """Neural network for wallet pattern recognition"""
    
    def __init__(self, config):
        super(WalletPatternRecognizer, self).__init__()
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Network architecture
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(self.config.pattern_classes))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 256 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)
    
    def analyze_patterns(self, wallet_data: bytes) -> Dict[str, any]:
        """Analyze wallet data for known patterns"""
        # Preprocess data
        tensor_data = self._preprocess_data(wallet_data)
        
        # Get model predictions
        with torch.no_grad():
            predictions = self(tensor_data)
            
        # Process and format results
        return self._process_predictions(predictions)
    
    def _preprocess_data(self, data: bytes) -> torch.Tensor:
        """Preprocess wallet data for the network"""
        # Convert bytes to numpy array
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        # Normalize data
        normalized = byte_array.astype(np.float32) / 255.0
        
        # Reshape for 1D convolution (batch_size, channels, length)
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.view(1, 1, -1)
        
        # Pad or truncate to expected length
        target_length = self.config.input_length
        current_length = tensor.size(-1)
        
        if current_length < target_length:
            padding = target_length - current_length
            tensor = F.pad(tensor, (0, padding))
        elif current_length > target_length:
            tensor = tensor[:, :, :target_length]
            
        return tensor
    
    def _process_predictions(self, predictions: torch.Tensor) -> Dict[str, any]:
        """Process and format model predictions"""
        # Get probabilities and classes
        probs, classes = torch.max(predictions, dim=1)
        
        # Convert to numpy for processing
        probs = probs.numpy()
        classes = classes.numpy()
        
        # Get top k predictions
        k = min(5, len(self.config.pattern_classes))
        top_k = torch.topk(predictions[0], k)
        
        results = {
            'top_patterns': [
                {
                    'pattern': self.config.pattern_classes[idx.item()],
                    'confidence': score.item()
                }
                for score, idx in zip(top_k.values, top_k.indices)
            ],
            'overall_confidence': float(probs[0]),
            'predicted_class': self.config.pattern_classes[classes[0]],
            'pattern_distribution': {
                class_name: float(predictions[0][i])
                for i, class_name in enumerate(self.config.pattern_classes)
            }
        }
        
        # Add analysis metadata
        results['analysis_metadata'] = {
            'threshold_confidence': self.config.confidence_threshold,
            'model_version': self.config.model_version,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def detect_anomalies(self, wallet_data: bytes) -> Dict[str, any]:
        """Detect anomalous patterns in wallet data"""
        # Preprocess data
        tensor_data = self._preprocess_data(wallet_data)
        
        # Get feature representations
        features = self._extract_features(tensor_data)
        
        # Perform anomaly detection
        anomalies = self._detect_feature_anomalies(features)
        
        return {
            'anomalies': anomalies,
            'anomaly_score': self._calculate_anomaly_score(features),
            'feature_analysis': self._analyze_features(features)
        }
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate feature representations"""
        # Get features from intermediate layers
        features = []
        
        # Conv1 features
        x1 = F.relu(self.conv1(x))
        features.append(x1)
        
        # Conv2 features
        x2 = F.relu(self.conv2(self.pool(x1)))
        features.append(x2)
        
        # Conv3 features
        x3 = F.relu(self.conv3(self.pool(x2)))
        features.append(x3)
        
        return features
    
    def _detect_feature_anomalies(self, features: List[torch.Tensor]) -> List[Dict[str, any]]:
        """Detect anomalies in feature representations"""
        anomalies = []
        
        for i, feature_map in enumerate(features):
            # Calculate statistics
            mean = torch.mean(feature_map)
            std = torch.std(feature_map)
            
            # Find anomalous activations
            threshold = mean + (self.config.anomaly_threshold * std)
            anomalous_activations = (feature_map > threshold).nonzero()
            
            if len(anomalous_activations) > 0:
                anomalies.append({
                    'layer': f'conv{i+1}',
                    'locations': anomalous_activations.tolist(),
                    'intensity': float(torch.max(feature_map)),
                    'mean_activation': float(mean),
                    'std_activation': float(std)
                })
                
        return anomalies
    
    def _calculate_anomaly_score(self, features: List[torch.Tensor]) -> float:
        """Calculate overall anomaly score"""
        scores = []
        
        for feature_map in features:
            mean = torch.mean(feature_map)
            std = torch.std(feature_map)
            max_activation = torch.max(feature_map)
            
            # Calculate normalized deviation from mean
            score = float((max_activation - mean) / std)
            scores.append(score)
            
        return np.mean(scores)
    
    def _analyze_features(self, features: List[torch.Tensor]) -> Dict[str, any]:
        """Analyze feature patterns"""
        return {
            f'layer_{i+1}': {
                'mean_activation': float(torch.mean(feature_map)),
                'max_activation': float(torch.max(feature_map)),
                'activation_distribution': torch.histc(feature_map, bins=10).tolist(),
                'spatial_pattern': self._analyze_spatial_pattern(feature_map)
            }
            for i, feature_map in enumerate(features)
        }
    
    def _analyze_spatial_pattern(self, feature_map: torch.Tensor) -> Dict[str, any]:
        """Analyze spatial patterns in feature maps
        
        Args:
            feature_map: Tensor containing feature map activations [B, C, L]
            
        Returns:
            Dictionary containing spatial pattern analysis results
        """
        try:
            analysis = {
                'statistics': {},
                'patterns': {},
                'correlations': {},
                'structure': {},
                'anomalies': []
            }
            
            # Basic statistics
            analysis['statistics'] = {
                'mean': float(torch.mean(feature_map).item()),
                'std': float(torch.std(feature_map).item()),
                'max': float(torch.max(feature_map).item()),
                'min': float(torch.min(feature_map).item()),
                'sparsity': float((feature_map == 0).float().mean().item()),
                'entropy': self._calculate_entropy(feature_map)
            }
            
            # Pattern detection
            analysis['patterns'] = {
                'repetitive': self._detect_repetitive_patterns(feature_map),
                'gradients': self._analyze_gradients(feature_map),
                'peaks': self._find_activation_peaks(feature_map),
                'sequences': self._detect_sequences(feature_map)
            }
            
            # Correlation analysis
            analysis['correlations'] = {
                'temporal': self._analyze_temporal_correlation(feature_map),
                'channel': self._analyze_channel_correlation(feature_map),
                'auto': self._compute_autocorrelation(feature_map)
            }
            
            # Structure analysis
            analysis['structure'] = {
                'segments': self._analyze_segments(feature_map),
                'periodicity': self._analyze_periodicity(feature_map),
                'symmetry': self._analyze_symmetry(feature_map)
            }
            
            # Anomaly detection
            analysis['anomalies'] = self._detect_pattern_anomalies(feature_map)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in spatial pattern analysis: {str(e)}")
            return {}
        
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate Shannon entropy of feature activations"""
        # Normalize to probability distribution
        p = F.softmax(tensor.flatten(), dim=0)
        entropy = -torch.sum(p * torch.log2(p + 1e-10))
        return float(entropy.item())
    
    def _detect_repetitive_patterns(self, feature_map: torch.Tensor) -> Dict[str, any]:
        """Detect repetitive patterns in feature map"""
        patterns = {}
        
        # Analyze local windows
        window_size = min(32, feature_map.size(-1) // 4)
        stride = window_size // 2
        
        windows = feature_map.unfold(-1, window_size, stride)
        
        # Calculate similarity between windows
        similarity_matrix = F.cosine_similarity(
            windows.unsqueeze(2),
            windows.unsqueeze(1),
            dim=-1
        )
        
        # Find repeating patterns
        threshold = 0.8
        repeating = (similarity_matrix > threshold).float()
        
        patterns['count'] = int(torch.sum(repeating).item())
        patterns['locations'] = torch.nonzero(repeating).tolist()
        patterns['similarity_scores'] = similarity_matrix[patterns['locations']].tolist()
        
        return patterns
    
    def _analyze_gradients(self, feature_map: torch.Tensor) -> Dict[str, float]:
        """Analyze activation gradients"""
        gradients = torch.gradient(feature_map, dim=-1)[0]
        
        return {
            'mean_gradient': float(torch.mean(torch.abs(gradients)).item()),
            'max_gradient': float(torch.max(torch.abs(gradients)).item()),
            'gradient_variance': float(torch.var(gradients).item())
        }
    
    def _find_activation_peaks(self, feature_map: torch.Tensor) -> List[Dict[str, any]]:
        """Find significant activation peaks"""
        # Calculate local maxima
        kernel_size = 3
        maxpool = F.max_pool1d(
            feature_map,
            kernel_size,
            stride=1,
            padding=kernel_size//2
        )
        
        is_peak = (feature_map == maxpool)
        
        # Filter significant peaks
        threshold = torch.mean(feature_map) + torch.std(feature_map)
        significant_peaks = is_peak & (feature_map > threshold)
        
        peak_locations = torch.nonzero(significant_peaks)
        peak_values = feature_map[significant_peaks]
        
        return [
            {
                'location': loc.tolist(),
                'value': float(val.item())
            }
            for loc, val in zip(peak_locations, peak_values)
        ]
    
    def _detect_sequences(self, feature_map: torch.Tensor) -> Dict[str, any]:
        """Detect sequential patterns"""
        # Normalize feature map
        normalized = (feature_map - torch.mean(feature_map)) / torch.std(feature_map)
        
        # Find sequences using rolling correlation
        sequence_length = min(16, feature_map.size(-1) // 8)
        correlations = F.conv1d(
            normalized.unsqueeze(1),
            normalized[..., :sequence_length].unsqueeze(1),
            padding='same'
        )
        
        return {
            'sequence_length': sequence_length,
            'correlation_scores': correlations.squeeze().tolist(),
            'significant_sequences': torch.nonzero(correlations > 0.7).tolist()
        }
    
    def _analyze_temporal_correlation(self, feature_map: torch.Tensor) -> float:
        """Analyze temporal correlation in activations"""
        # Calculate correlation between adjacent time steps
        corr = F.cosine_similarity(
            feature_map[..., :-1],
            feature_map[..., 1:],
            dim=1
        )
        return float(torch.mean(corr).item())
    
    def _analyze_channel_correlation(self, feature_map: torch.Tensor) -> float:
        """Analyze correlation between channels"""
        if feature_map.size(1) > 1:
            corr = F.cosine_similarity(
                feature_map.unsqueeze(2),
                feature_map.unsqueeze(1),
                dim=-1
            )
            return float(torch.mean(corr).item())
        return 0.0
    
    def _compute_autocorrelation(self, feature_map: torch.Tensor) -> List[float]:
        """Compute autocorrelation of feature map"""
        # Normalize feature map
        normalized = (feature_map - torch.mean(feature_map)) / torch.std(feature_map)
        
        # Calculate autocorrelation for different lags
        max_lag = min(32, feature_map.size(-1) // 4)
        autocorr = [
            float(F.cosine_similarity(
                normalized[..., :-lag] if lag > 0 else normalized,
                normalized[..., lag:] if lag > 0 else normalized,
                dim=-1
            ).mean().item())
            for lag in range(max_lag)
        ]
        
        return autocorr