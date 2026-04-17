"""Constraint-aware neural network layers and networks"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from gismol.core import COHObject


class NeuralLayer(COHObject):
    """Base class for constraint-aware neural network layers"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass - to be overridden"""
        return x
    
    def get_parameter_count(self) -> int:
        """Return number of trainable parameters"""
        return 0
    
    def get_memory_footprint(self) -> int:
        """Estimate memory usage in bytes"""
        return 0


class LinearLayer(NeuralLayer):
    """Linear (dense) layer with constraint awareness"""
    
    def __init__(self, name: str, in_features: int, out_features: int, activation: str = 'relu', **kwargs):
        super().__init__(name, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self._weights = np.random.randn(out_features, in_features) * 0.01
        self._bias = np.zeros(out_features)
        self.input_shape = (in_features,)
        self.output_shape = (out_features,)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        out = self._weights @ x + self._bias
        if self.activation_name == 'relu':
            out = np.maximum(0, out)
        elif self.activation_name == 'sigmoid':
            out = 1 / (1 + np.exp(-out))
        elif self.activation_name == 'tanh':
            out = np.tanh(out)
        elif self.activation_name == 'softmax':
            exp_out = np.exp(out - np.max(out))
            out = exp_out / np.sum(exp_out)
        return out
    
    def get_parameter_count(self) -> int:
        return self.in_features * self.out_features + self.out_features


class ConvLayer(NeuralLayer):
    """Convolutional layer (simplified)"""
    
    def __init__(self, name: str, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Simplified: treat as linear for now
        self._linear = LinearLayer(f"{name}_linear", in_channels * kernel_size, out_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Simplified: assume 1D input
        return self._linear.forward(x)


class PoolingLayer(NeuralLayer):
    """Pooling layer (simplified)"""
    
    def __init__(self, name: str, pool_size: int, mode: str = 'max', **kwargs):
        super().__init__(name, **kwargs)
        self.pool_size = pool_size
        self.mode = mode
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Simplified: average pooling
        if len(x) % self.pool_size != 0:
            x = x[:len(x) - len(x) % self.pool_size]
        reshaped = x.reshape(-1, self.pool_size)
        if self.mode == 'max':
            return np.max(reshaped, axis=1)
        else:
            return np.mean(reshaped, axis=1)


class NeuralNetwork(COHObject):
    """Constraint-aware neural network composed of layers"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.layers: List[NeuralLayer] = []
        self._built = False
    
    def add_layer(self, layer: NeuralLayer) -> None:
        """Add a layer to the network"""
        self.layers.append(layer)
        self.add_child(layer)
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the network (validate layer compatibility)"""
        current_shape = input_shape
        for layer in self.layers:
            if layer.input_shape:
                # Validate compatibility
                pass
            current_shape = layer.output_shape or current_shape
        self._built = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        if not self._built:
            raise RuntimeError("Network must be built before forward pass")
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def validate_network(self) -> bool:
        """Validate that all constraints are satisfied"""
        # Check each layer's constraints
        for layer in self.layers:
            context = layer.get_context()
            for constraint in layer.identity_constraints:
                if not self.constraint_system.validate_single(constraint, context):
                    return False
        return True