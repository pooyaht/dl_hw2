import torch
import torch.nn as nn
import numpy as np


class SimpleResNetYOLO(nn.Module):
    def __init__(self, anchor_boxes, grid_size=14, unfreeze_partial_epoch=10, unfreeze_all_epoch=20, dropout_rate=0.2):
        super().__init__()

        self.num_classes = 2
        self.num_anchors = len(anchor_boxes)
        self.unfreeze_partial_epoch = unfreeze_partial_epoch
        self.unfreeze_all_epoch = unfreeze_all_epoch
        self.current_epoch = 0
        self.freeze_state = 'frozen'  # 'frozen', 'partial', 'all'

        resnet50 = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])

        if grid_size not in [7, 14]:
            raise ValueError("grid_size must be 7 or 14")

        # if grid_size == 14:
        #     layer4 = self.backbone[-1]
        #     for block in layer4:
        #         if hasattr(block, 'conv2'):
        #             if block.conv2.stride == (2, 2):
        #                 block.conv2.stride = (1, 1)
        #                 block.conv2.dilation = (2, 2)
        #                 block.conv2.padding = (2, 2)
        #         if hasattr(block, 'downsample') and block.downsample is not None:
        #             if block.downsample[0].stride == (2, 2):
        #                 block.downsample[0].stride = (1, 1)

        backbone_channels = 2048

        print(f"ResNet50 configured for {grid_size}x{grid_size} grid")
        print(f"Output channels: {backbone_channels}")
        print(
            f"Backbone parameters: {sum(p.numel() for p in self.backbone.parameters()) / 1e6:.1f}M")

        self.prediction_head = nn.Sequential(
            nn.Conv2d(backbone_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(256, 256, kernel_size=3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(256, self.num_anchors *
                      (5 + self.num_classes), kernel_size=1)
        )

        self.register_buffer('anchors', torch.tensor(anchor_boxes)
                             if not isinstance(anchor_boxes, torch.Tensor) else anchor_boxes)

        self._initialize_weights()
        self.freeze_backbone()

    def _initialize_weights(self):
        for m in self.prediction_head.modules():
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

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone completely unfrozen")

    def unfreeze_backbone_layers(self, num_layers=2):
        backbone_layers = list(self.backbone.children())

        if num_layers > 0 and num_layers <= len(backbone_layers):
            layers_to_unfreeze = backbone_layers[-num_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {num_layers} backbone layers")

        elif num_layers > len(backbone_layers):
            self.unfreeze_backbone()

    def update_epoch(self, epoch):
        self.current_epoch = epoch

        if epoch >= self.unfreeze_all_epoch and self.freeze_state != 'all':
            self.unfreeze_backbone()
            self.freeze_state = 'all'
            print(f"Epoch {epoch}: Unfroze all backbone layers")
            return 'unfreeze_all'

        elif epoch >= self.unfreeze_partial_epoch and self.freeze_state == 'frozen':
            self.unfreeze_backbone_layers(2)
            self.freeze_state = 'partial'
            print(f"Epoch {epoch}: Unfroze last 2 backbone layers")
            return 'unfreeze_partial'

        return None

    def forward(self, x):
        batch_size = x.size(0)

        features = self.backbone(x)

        predictions = self.prediction_head(features)

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
        backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.prediction_head.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'backbone_trainable_parameters': backbone_trainable,
            'head_parameters': head_params,
            'backbone_frozen': backbone_trainable == 0,
            'current_epoch': self.current_epoch,
            'freeze_state': self.freeze_state,
            'unfreeze_partial_epoch': self.unfreeze_partial_epoch,
            'unfreeze_all_epoch': self.unfreeze_all_epoch
        }

    def print_model_info(self):
        info = self.get_model_info()
        print(f"\n{'='*50}")
        print(f"Model Information - Epoch {info['current_epoch']}")
        print(f"{'='*50}")
        print(
            f"Total parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.1f}M)")
        print(
            f"Trainable parameters: {info['trainable_parameters']:,} ({info['trainable_parameters']/1e6:.1f}M)")
        print(
            f"Backbone parameters: {info['backbone_parameters']:,} ({info['backbone_parameters']/1e6:.1f}M)")
        print(
            f"  - Trainable: {info['backbone_trainable_parameters']:,} ({info['backbone_trainable_parameters']/1e6:.1f}M)")
        print(
            f"Head parameters: {info['head_parameters']:,} ({info['head_parameters']/1e6:.1f}M)")
        print(
            f"Backbone status: {info['freeze_state']}")
        print(
            f"Will unfreeze partially at epoch: {info['unfreeze_partial_epoch']}")
        print(f"Will unfreeze all at epoch: {info['unfreeze_all_epoch']}")
