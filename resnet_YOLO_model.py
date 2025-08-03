import torch
import numpy as np
import torch.nn as nn


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate=0.1):
        super(ResidualFeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x + self.adapter(x)


class ResNetYOLODetector(nn.Module):
    def __init__(self, anchor_boxes, backbone_name="resnet50", grid_size=14, freeze_backbone_epochs=15, dropout_rate=0.1):
        super(ResNetYOLODetector, self).__init__()

        self.num_classes = 2
        self.num_anchors = len(anchor_boxes)
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0
        self.dropout_rate = dropout_rate

        self.backbone, backbone_channels = self._load_backbone(
            backbone_name, grid_size)
        self.feature_adapter = ResidualFeatureAdapter(
            backbone_channels, backbone_channels // 2, dropout_rate)

        self.prediction_head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels //
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(backbone_channels // 4, self.num_anchors *
                      (5 + self.num_classes), kernel_size=1)
        )

        self.register_buffer('anchors', torch.tensor(anchor_boxes)
                             if not isinstance(anchor_boxes, torch.Tensor) else anchor_boxes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        final_conv = self.prediction_head[-1]
        num_outputs_per_anchor = 5 + self.num_classes
        for i in range(self.num_anchors):
            obj_idx = i * num_outputs_per_anchor + 4
            nn.init.constant_(
                final_conv.bias[obj_idx], -np.log((1 - 0.01) / 0.01))

        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_backbone_layers(self, num_layers=2):
        backbone_layers = list(self.backbone.children())

        if num_layers > 0:
            layers_to_unfreeze = backbone_layers[-num_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

    def update_epoch(self, epoch, partial_unfreeze_layers=2):
        self.current_epoch = epoch
        if epoch >= self.freeze_backbone_epochs and self.backbone[0].weight.requires_grad == False:
            if partial_unfreeze_layers == -1:
                self.unfreeze_backbone()
                print(f"Epoch {epoch}: Unfroze entire backbone")
            else:
                self.unfreeze_backbone_layers(partial_unfreeze_layers)
                print(
                    f"Epoch {epoch}: Unfroze last {partial_unfreeze_layers} backbone layers")

    def _load_backbone(self, backbone_name, grid_size=7):

        backbone = torch.hub.load(
            'pytorch/vision:v0.10.0', backbone_name, pretrained=True)

        if backbone_name in ['resnet18', 'resnet34']:
            final_channels = 512
            has_bottleneck = False
        else:
            final_channels = 2048
            has_bottleneck = True

        if grid_size == 7:
            backbone_modified = nn.Sequential(*list(backbone.children())[:-2])
            output_channels = final_channels

        elif grid_size == 14:
            if has_bottleneck:
                backbone.layer4[0].conv2.stride = (1, 1)
                backbone.layer4[0].downsample[0].stride = (1, 1)
            else:
                backbone.layer4[0].conv1.stride = (1, 1)
                backbone.layer4[0].downsample[0].stride = (1, 1)

            backbone_modified = nn.Sequential(*list(backbone.children())[:-2])
            output_channels = final_channels

        elif grid_size == 28:
            if has_bottleneck:
                backbone.layer3[0].conv2.stride = (1, 1)
                backbone.layer3[0].downsample[0].stride = (1, 1)
                backbone.layer4[0].conv2.stride = (1, 1)
                backbone.layer4[0].downsample[0].stride = (1, 1)
            else:
                backbone.layer3[0].conv1.stride = (1, 1)
                backbone.layer3[0].downsample[0].stride = (1, 1)
                backbone.layer4[0].conv1.stride = (1, 1)
                backbone.layer4[0].downsample[0].stride = (1, 1)

            backbone_modified = nn.Sequential(*list(backbone.children())[:-2])
            output_channels = final_channels

        else:
            raise ValueError(
                f"Unsupported grid size: {grid_size}. Supported: [7, 14, 28]")

        print(
            f"Backbone {backbone_name} configured for {grid_size}x{grid_size} grid")
        print(f"Output channels: {output_channels}")
        print(
            f"Approximate backbone parameters: {sum(p.numel() for p in backbone_modified.parameters()) / 1e6:.1f}M")

        return backbone_modified, output_channels

    def forward(self, x):
        batch_size = x.size(0)

        features = self.backbone(x)
        adapted_features = self.feature_adapter(features)
        predictions = self.prediction_head(adapted_features)

        predictions = predictions.view(
            batch_size,
            self.num_anchors,
            5 + self.num_classes,
            predictions.size(-2),
            predictions.size(-1)
        )

        return predictions

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        adapter_params = sum(p.numel()
                             for p in self.feature_adapter.parameters())
        head_params = sum(p.numel() for p in self.prediction_head.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'adapter_parameters': adapter_params,
            'head_parameters': head_params,
            'backbone_frozen': not self.backbone[0].weight.requires_grad,
            'current_epoch': self.current_epoch
        }
