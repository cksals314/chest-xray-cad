"""Pretrained CNN backbones adapted for 14-label CXR classification."""


import torch.nn as nn

try:
    from .constants import NUM_CLASSES
except ImportError:  # Support running scripts from the src directory
    from constants import NUM_CLASSES

from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


# ==========================================
# 1. Configuration
# ==========================================

SUPPORTED_MODELS = [
    "densenet121",
    "efficientnet_b0",
]


# ==========================================
# 2. Model builder
# ==========================================

def build_model(
    model_name,
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=False,
):
    """
    Load an ImageNet-pretrained model and replace
    its final classification layer.

    Args:
        model_name:
            "densenet121" or "efficientnet_b0"

        num_classes:
            Number of output classes (default: 14)

        pretrained:
            Load ImageNet pretrained weights if True.

        freeze_backbone:
            Freeze feature extractor if True.
            Only the new classification layer is trainable.

    Returns:
        PyTorch model producing raw logits of shape
        (batch_size, num_classes).
    """

    model_name = model_name.lower()

    if model_name == "densenet121":

        weights = (
            DenseNet121_Weights.DEFAULT
            if pretrained else None
        )

        model = densenet121(weights=weights)

        # Replace the original 1000-class classifier.
        in_features = model.classifier.in_features

        model.classifier = nn.Linear(
            in_features,
            num_classes,
        )

    elif model_name == "efficientnet_b0":

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )

        model = efficientnet_b0(weights=weights)

        # Replace the original 1000-class classifier.
        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes,
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from {SUPPORTED_MODELS}."
        )

    # Optionally freeze the feature extractor.
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model


# ==========================================
# 3. Parameter statistics
# ==========================================

def count_parameters(model):
    """Return the total and trainable parameter counts for ``model``."""

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_params, trainable_params


# ==========================================
# 4. Optimizer parameter groups
# ==========================================

def get_parameter_groups(
    model,
    backbone_lr=1e-4,
    classifier_lr=5e-4,
):
    """Create optimizer parameter groups with separate backbone/head LRs.

    The classifier parameters are identified by object identity so they are
    excluded from the backbone group and are not added to the optimizer twice.
    """

    classifier_param_ids = {
        id(parameter)
        for parameter in model.classifier.parameters()
    }

    backbone_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in classifier_param_ids
    ]

    classifier_params = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]

    parameter_groups = []

    if backbone_params:
        parameter_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
        })

    if classifier_params:
        parameter_groups.append({
            "params": classifier_params,
            "lr": classifier_lr,
        })

    return parameter_groups
