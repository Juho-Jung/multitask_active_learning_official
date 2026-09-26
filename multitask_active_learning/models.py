"""
Multi-Task Learning Model Components

Provides shared encoder and task-specific heads for multi-task learning.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops


class SharedEncoder(nn.Module):
    """
    Shared encoder for multi-task learning.

    Args:
        input_channels: Number of input channels
        hidden_dim: Hidden dimension size
        num_layers: Number of encoder layers (only used if pretrained=False)
        dropout: Dropout rate
        pretrained: Whether to use ImageNet pretrained ResNet-50
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        pretrained: bool = False
    ):
        super().__init__()

        self.input_channels = input_channels
        self.pretrained = pretrained

        if pretrained:
            # Use ResNet-50 pretrained on ImageNet
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove the final fully connected layer and average pooling
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            # ResNet-50 output: [batch, 2048, H/32, W/32]
            self.output_dim = 2048
            self.hidden_dim = 2048
        else:
            # Custom CNN encoder
            self.hidden_dim = hidden_dim
            layers = []
            in_channels = input_channels

            for i in range(num_layers):
                out_channels = hidden_dim // (2 ** (num_layers - 1 - i))
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.MaxPool2d(2, 2)
                ])
                in_channels = out_channels

            self.encoder = nn.Sequential(*layers)
            self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared encoder.

        Args:
            x: Input tensor [batch_size, channels, height, width]

        Returns:
            Encoded features [batch_size, hidden_dim, h', w']
        """
        return self.encoder(x)


class TaskHead(nn.Module):
    """
    Task-specific head for multi-task learning.

    Args:
        input_dim: Input feature dimension
        task_type: Type of task ('classification', 'detection', 'segmentation')
        num_classes: Number of classes for classification/segmentation
        num_boxes: Number of boxes for detection (if applicable, used for anchor-free)
        detection_model: 'anchor-free' or 'anchor-based' (default: 'anchor-free')
        anchor_scales: List of anchor scales for anchor-based (default: [0.5, 1.0, 2.0])
        anchor_ratios: List of anchor aspect ratios for anchor-based (default: [0.5, 1.0, 2.0])
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 10,
        num_boxes: int = 5,
        detection_model: str = 'anchor-free',
        anchor_scales: List[float] = None,
        anchor_ratios: List[float] = None
    ):
        super().__init__()

        self.task_type = task_type
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.detection_model = detection_model

        if task_type == 'classification':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        elif task_type == 'detection':
            if detection_model == 'anchor-free':
                # Original anchor-free head
                self.head = nn.Sequential(
                    nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, num_boxes * (num_classes + 4), kernel_size=1)
                )
            elif detection_model == 'anchor-based':
                # Anchor-based head
                if anchor_scales is None:
                    anchor_scales = [0.5, 1.0, 2.0]
                if anchor_ratios is None:
                    anchor_ratios = [0.5, 1.0, 2.0]

                self.anchor_scales = anchor_scales
                self.anchor_ratios = anchor_ratios
                self.num_anchors = len(anchor_scales) * len(anchor_ratios)

                # Classification head: predicts class scores for each anchor
                self.cls_head = nn.Sequential(
                    nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, self.num_anchors * num_classes, kernel_size=1)
                )

                # Regression head: predicts bbox offsets for each anchor
                self.reg_head = nn.Sequential(
                    nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, self.num_anchors * 4, kernel_size=1)
                )
            else:
                raise ValueError(f"Unknown detection model: {detection_model}")

        elif task_type == 'segmentation':
            # Upsampling scale depends on encoder: custom CNN (16x) or ResNet-50 (32x)
            # We'll use adaptive upsampling or calculate based on input size
            # For now, use 32x for ResNet-50 compatibility (will be adjusted dynamically if needed)
            self.head = nn.Sequential(
                nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1)
            )
            # Store upsampling separately to adjust based on encoder
            self.upsample_scale = 16 if input_dim <= 256 else 32  # Custom CNN: 16x, ResNet-50: 32x

        elif task_type in ['depth', 'normal'] or task_type == 'regression':
            # Regression tasks: depth estimation or surface normal prediction
            # Depth: [batch, 1, H, W]
            # Normal: [batch, 3, H, W]
            output_channels = 1 if task_type == 'depth' else 3
            self.head = nn.Sequential(
                nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, output_channels, kernel_size=1)
            )
            # Store upsampling separately to adjust based on encoder
            self.upsample_scale = 16 if input_dim <= 256 else 32  # Custom CNN: 16x, ResNet-50: 32x

        elif task_type == 'instance_segmentation':
            # Instance segmentation: predict instance mask for 'thing' classes
            # Similar to semantic segmentation but outputs instance-aware predictions
            self.head = nn.Sequential(
                nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1)  # num_classes includes background
            )
            # Store upsampling separately to adjust based on encoder
            self.upsample_scale = 16 if input_dim <= 256 else 32

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _generate_anchors(self, feature_map_size: Tuple[int, int],
                          image_size: Tuple[int, int], device: torch.device = None) -> torch.Tensor:
        """Generate anchor boxes for anchor-based detection."""
        h, w = feature_map_size
        img_h, img_w = image_size

        # Calculate stride
        stride_h = img_h / h
        stride_w = img_w / w

        anchors = []
        for i in range(h):
            for j in range(w):
                # Center of anchor in image coordinates
                cy = (i + 0.5) * stride_h
                cx = (j + 0.5) * stride_w

                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        # Anchor size
                        anchor_h = scale * stride_h * (ratio ** 0.5)
                        anchor_w = scale * stride_w / (ratio ** 0.5)

                        # Convert to [x, y, w, h] normalized format
                        x_coord = (cx - anchor_w / 2) / img_w
                        y_coord = (cy - anchor_h / 2) / img_h
                        w_norm = anchor_w / img_w
                        h_norm = anchor_h / img_h

                        anchors.append([x_coord, y_coord, w_norm, h_norm])

        # Use provided device or default to CPU
        if device is None:
            device = torch.device('cpu')
        return torch.tensor(anchors, device=device, dtype=torch.float32)

    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                   max_detections: int = 100, iou_threshold: float = 0.5,
                   score_threshold: float = 0.05) -> torch.Tensor:
        """Apply Non-Maximum Suppression to filter boxes."""
        if len(boxes) == 0:
            return torch.zeros((0, 4), device=boxes.device, dtype=boxes.dtype)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        if valid_mask.sum() == 0:
            return torch.zeros((0, 4), device=boxes.device, dtype=boxes.dtype)

        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

        # Convert from [x, y, w, h] to [x1, y1, x2, y2] for NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # Apply NMS
        keep_indices = ops.nms(boxes_xyxy, scores, iou_threshold)

        # Limit to max_detections
        if len(keep_indices) > max_detections:
            # Sort by score and take top-k
            sorted_scores, sorted_indices = torch.sort(scores[keep_indices], descending=True)
            keep_indices = keep_indices[sorted_indices[:max_detections]]

        return boxes[keep_indices]

    def forward(self, x: torch.Tensor, image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass through task head.

        Args:
            x: Input features from shared encoder [batch, channels, H, W]
            image_size: Original image size (height, width) for anchor-based detection

        Returns:
            Task-specific output
        """
        if self.task_type == 'detection':
            if self.detection_model == 'anchor-free':
                # Original anchor-free forward
                output = self.head(x)
                batch_size = output.shape[0]
                output = F.adaptive_avg_pool2d(output, (1, 1))
                output = output.view(batch_size, -1)
                return output.view(batch_size, self.num_boxes, -1)

            elif self.detection_model == 'anchor-based':
                batch_size = x.shape[0]
                h, w = x.shape[2], x.shape[3]

                # Get predictions
                cls_pred = self.cls_head(x)  # [batch, num_anchors * num_classes, H, W]
                reg_pred = self.reg_head(x)  # [batch, num_anchors * 4, H, W]

                # Reshape predictions
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()  # [batch, H, W, num_anchors * num_classes]
                cls_pred = cls_pred.view(batch_size, h * w * self.num_anchors, self.num_classes)

                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()  # [batch, H, W, num_anchors * 4]
                reg_pred = reg_pred.view(batch_size, h * w * self.num_anchors, 4)

                # Get image size (default to 480x480 if not provided)
                if image_size is None:
                    image_size = (480, 480)

                # Generate anchors for this feature map
                anchors = self._generate_anchors((h, w), image_size, device=x.device)  # [H*W*num_anchors, 4]
                anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, H*W*num_anchors, 4]

                # Apply regression offsets to anchors
                # reg_pred is offsets: [dx, dy, dw, dh]
                boxes = torch.zeros_like(anchors)
                boxes[:, :, 0] = anchors[:, :, 0] + reg_pred[:, :, 0]  # x
                boxes[:, :, 1] = anchors[:, :, 1] + reg_pred[:, :, 1]  # y
                boxes[:, :, 2] = anchors[:, :, 2] * torch.exp(reg_pred[:, :, 2])  # w
                boxes[:, :, 3] = anchors[:, :, 3] * torch.exp(reg_pred[:, :, 3])  # h

                # Clip boxes to [0, 1]
                boxes = torch.clamp(boxes, 0, 1)

                # During training, return raw predictions (without NMS) to preserve gradients
                # During evaluation, apply NMS for cleaner predictions
                if self.training:
                    # Training: return top-k predictions by score (without NMS) to maintain gradients
                    cls_probs = F.softmax(cls_pred, dim=-1)  # [batch, num_anchors, num_classes]
                    max_scores, class_ids = torch.max(cls_probs, dim=-1)  # [batch, num_anchors]

                    # Select top-k anchors by score for each image
                    batch_outputs = []
                    for i in range(batch_size):
                        # Get top-k anchors by max score
                        topk_scores, topk_indices = torch.topk(max_scores[i], k=min(self.num_boxes, len(max_scores[i])))

                        # Get corresponding boxes and class predictions
                        topk_boxes = boxes[i][topk_indices]  # [k, 4]
                        topk_cls_pred = cls_pred[i][topk_indices]  # [k, num_classes]

                        # Pad to num_boxes if needed
                        if len(topk_boxes) < self.num_boxes:
                            padding_boxes = torch.zeros(
                                (self.num_boxes - len(topk_boxes), 4),
                                device=topk_boxes.device,
                                dtype=topk_boxes.dtype
                            )
                            padding_cls = torch.zeros(
                                (self.num_boxes - len(topk_boxes), self.num_classes),
                                device=topk_cls_pred.device,
                                dtype=topk_cls_pred.dtype
                            )
                            topk_boxes = torch.cat([topk_boxes, padding_boxes], dim=0)
                            topk_cls_pred = torch.cat([topk_cls_pred, padding_cls], dim=0)
                        else:
                            topk_boxes = topk_boxes[:self.num_boxes]
                            topk_cls_pred = topk_cls_pred[:self.num_boxes]

                        # Create output format: [num_boxes, num_classes + 4]
                        output_i = torch.cat([topk_cls_pred, topk_boxes], dim=-1)
                        batch_outputs.append(output_i)

                    return torch.stack(batch_outputs, dim=0)  # [batch, num_boxes, num_classes + 4]
                else:
                    # Evaluation: apply NMS for cleaner predictions
                    cls_probs = F.softmax(cls_pred, dim=-1)  # [batch, num_anchors, num_classes]
                    max_scores, class_ids = torch.max(cls_probs, dim=-1)  # [batch, num_anchors]

                    # Apply NMS per image and convert to fixed format
                    batch_outputs = []
                    for i in range(batch_size):
                        # Apply NMS
                        nms_boxes = self._apply_nms(
                            boxes[i], max_scores[i],
                            max_detections=self.num_boxes,
                            iou_threshold=0.5,
                            score_threshold=0.05
                        )

                        # Pad or truncate to num_boxes
                        if len(nms_boxes) < self.num_boxes:
                            # Pad with zeros
                            padding = torch.zeros(
                                (self.num_boxes - len(nms_boxes), 4),
                                device=nms_boxes.device,
                                dtype=nms_boxes.dtype
                            )
                            nms_boxes = torch.cat([nms_boxes, padding], dim=0)
                        else:
                            nms_boxes = nms_boxes[:self.num_boxes]

                        # Get corresponding class scores
                        # For simplicity, use max class for each box
                        # In practice, you'd track which anchors survived NMS
                        box_classes = torch.zeros(
                            (self.num_boxes, self.num_classes),
                            device=cls_probs.device,
                            dtype=cls_probs.dtype
                        )
                        box_classes[:, 0] = 1.0  # Default to background

                        # Create output format: [num_boxes, num_classes + 4]
                        output_i = torch.cat([box_classes, nms_boxes], dim=-1)
                        batch_outputs.append(output_i)

                    return torch.stack(batch_outputs, dim=0)  # [batch, num_boxes, num_classes + 4]

        elif self.task_type == 'segmentation':
            output = self.head(x)
            # Upsample to original image size
            output = F.interpolate(
                output,
                scale_factor=self.upsample_scale,
                mode='bilinear',
                align_corners=False
            )
            return output
        elif self.task_type == 'depth':
            # Depth regression: upsample to original image size
            output = self.head(x)
            output = F.interpolate(
                output,
                scale_factor=self.upsample_scale,
                mode='bilinear',
                align_corners=False
            )
            return output
        elif self.task_type == 'normal':
            # Surface normal prediction: output should be unit vectors
            output = self.head(x)
            output = F.interpolate(
                output,
                scale_factor=self.upsample_scale,
                mode='bilinear',
                align_corners=False
            )
            # Normalize to unit vectors to prevent explosion
            output = F.normalize(output, p=2, dim=1)
            return output
        elif self.task_type == 'regression':
            # Generic regression: upsample to original image size
            output = self.head(x)
            output = F.interpolate(
                output,
                scale_factor=self.upsample_scale,
                mode='bilinear',
                align_corners=False
            )
            return output
        elif self.task_type == 'instance_segmentation':
            # Instance segmentation: similar to semantic segmentation
            output = self.head(x)
            output = F.interpolate(
                output,
                scale_factor=self.upsample_scale,
                mode='bilinear',
                align_corners=False
            )
            return output
        else:
            return self.head(x)


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared encoder and task-specific heads.

    Args:
        input_channels: Number of input channels
        hidden_dim: Hidden dimension for shared encoder
        num_encoder_layers: Number of encoder layers
        task_configs: Dictionary mapping task names to their configurations
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        num_encoder_layers: int = 4,
        task_configs: Dict[str, Dict] = None,
        dropout: float = 0.1,
        pretrained: bool = False,
        detection_model: str = 'anchor-free',
        anchor_scales: List[float] = None,
        anchor_ratios: List[float] = None
    ):
        super().__init__()

        if task_configs is None:
            task_configs = {
                'classification': {'num_classes': 10},
                'segmentation': {'num_classes': 10}
            }

        self.shared_encoder = SharedEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            dropout=dropout,
            pretrained=pretrained
        )

        # Get actual output dimension from encoder (may be different if pretrained)
        encoder_output_dim = self.shared_encoder.output_dim

        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            task_type = config.get('task_type', task_name)
            num_classes = config.get('num_classes', 10)
            num_boxes = config.get('num_boxes', 5)

            # Get detection model type from config or use default
            task_detection_model = config.get('detection_model', detection_model)
            task_anchor_scales = config.get('anchor_scales', anchor_scales)
            task_anchor_ratios = config.get('anchor_ratios', anchor_ratios)

            self.task_heads[task_name] = TaskHead(
                input_dim=encoder_output_dim,
                task_type=task_type,
                num_classes=num_classes,
                num_boxes=num_boxes,
                detection_model=task_detection_model if task_type == 'detection' else 'anchor-free',
                anchor_scales=task_anchor_scales,
                anchor_ratios=task_anchor_ratios
            )

    def forward(self, x: torch.Tensor, image_size: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task model.

        Args:
            x: Input tensor [batch_size, channels, height, width]
            image_size: Original image size (height, width) for anchor-based detection

        Returns:
            Dictionary of task outputs
        """
        shared_features = self.shared_encoder(x)
        outputs = {}
        for task_name, head in self.task_heads.items():
            if head.task_type == 'detection' and head.detection_model == 'anchor-based':
                outputs[task_name] = head(shared_features, image_size=image_size)
            else:
                outputs[task_name] = head(shared_features)

        return outputs

    def get_shared_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get parameters from shared encoder.

        Returns:
            List of shared encoder parameters
        """
        return list(self.shared_encoder.parameters())

    def get_task_parameters(self, task_name: str) -> List[torch.nn.Parameter]:
        """
        Get parameters from specific task head.

        Args:
            task_name: Name of the task

        Returns:
            List of task-specific parameters
        """
        if task_name not in self.task_heads:
            raise ValueError(f"Task {task_name} not found")
        return list(self.task_heads[task_name].parameters())


def create_toy_model(
    input_channels: int = 3,
    num_classes: int = 10,
    device: torch.device = None
) -> MultiTaskModel:
    """
    Create a toy multi-task model for demonstration.

    Args:
        input_channels: Number of input channels
        num_classes: Number of classes
        device: Device to place model on

    Returns:
        Multi-task model
    """
    task_configs = {
        'classification': {
            'task_type': 'classification',
            'num_classes': num_classes
        },
        'segmentation': {
            'task_type': 'segmentation',
            'num_classes': num_classes
        }
    }

    model = MultiTaskModel(
        input_channels=input_channels,
        hidden_dim=128,
        num_encoder_layers=3,
        task_configs=task_configs
    )

    if device is not None:
        model.to(device)

    return model
