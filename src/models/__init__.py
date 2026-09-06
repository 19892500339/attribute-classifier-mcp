"""CNN models for attribute classification."""
from .cnn_trainer import train_attribute_model, train_all_attributes, get_backbone
from .inference import AttributeClassifier, ModelRegistry, InferencePipeline
