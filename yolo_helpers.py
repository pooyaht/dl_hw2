import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    bce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_weight = alpha * (1 - pt) ** gamma
    focal_loss = focal_weight * bce_loss

    if reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'mean':
        return focal_loss.mean()
    else:
        return focal_loss


def create_yolo_targets(targets, anchors, grid_size=14, image_size=224, num_classes=2):
    batch_size = len(targets)
    num_anchors = len(anchors)

    yolo_targets = torch.zeros(
        batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)

    for batch_idx, target in enumerate(targets):
        boxes = target['boxes']
        labels = target['labels']

        if len(boxes) == 0:
            continue

        grid_scale = grid_size / image_size

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            grid_x = center_x * grid_scale
            grid_y = center_y * grid_scale

            cell_x = int(grid_x)
            cell_y = int(grid_y)

            cell_x = max(0, min(grid_size - 1, cell_x))
            cell_y = max(0, min(grid_size - 1, cell_y))

            rel_x = grid_x - cell_x
            rel_y = grid_y - cell_y

            rel_w = width * grid_scale
            rel_h = height * grid_scale
            box_wh = torch.tensor([rel_w, rel_h])

            ious = []
            for anchor in anchors:
                anchor_tensor = torch.tensor(anchor) if not isinstance(
                    anchor, torch.Tensor) else anchor

                anchor_grid = anchor_tensor * grid_scale
                intersection = torch.min(box_wh, anchor_grid).prod()
                union = box_wh.prod() + anchor_grid.prod() - intersection
                iou = intersection / (union + 1e-6)
                ious.append(iou)

            best_anchor = torch.argmax(torch.tensor(ious))

            anchor_w = anchors[best_anchor][0] * grid_scale
            anchor_h = anchors[best_anchor][1] * grid_scale

            yolo_targets[batch_idx, best_anchor, 0,
                         cell_y, cell_x] = rel_x
            yolo_targets[batch_idx, best_anchor, 1,
                         cell_y, cell_x] = rel_y

            yolo_targets[batch_idx, best_anchor, 2, cell_y, cell_x] = torch.log(
                rel_w / anchor_w + 1e-6)
            yolo_targets[batch_idx, best_anchor, 3, cell_y, cell_x] = torch.log(
                rel_h / anchor_h + 1e-6)
            yolo_targets[batch_idx, best_anchor, 4, cell_y, cell_x] = 1.0

            class_idx = 5 + label
            yolo_targets[batch_idx, best_anchor,
                         class_idx, cell_y, cell_x] = 1.0

    return yolo_targets


def decode_yolo_predictions(predictions, anchors, grid_size=14, image_size=224, conf_threshold=0.5):
    batch_size = predictions.shape[0]

    num_anchors = len(anchors)
    pred_xy = torch.sigmoid(predictions[:, :, 0:2])
    pred_wh = predictions[:, :, 2:4]
    pred_obj = torch.sigmoid(predictions[:, :, 4])
    pred_cls = torch.softmax(predictions[:, :, 5:], dim=2)

    results = []

    for b in range(batch_size):
        boxes = []
        scores = []
        labels = []

        for a in range(num_anchors):
            obj_mask = pred_obj[b, a] > conf_threshold
            if not obj_mask.any():
                continue

            y_indices, x_indices = torch.where(obj_mask)
            for y_idx, x_idx in zip(y_indices, x_indices):
                rel_x = pred_xy[b, a, 0, y_idx, x_idx]
                rel_y = pred_xy[b, a, 1, y_idx, x_idx]
                rel_w = torch.exp(pred_wh[b, a, 0, y_idx, x_idx])
                rel_h = torch.exp(pred_wh[b, a, 1, y_idx, x_idx])

                center_x = (x_idx + rel_x) * image_size / grid_size
                center_y = (y_idx + rel_y) * image_size / grid_size

                anchor_w = anchors[a][0]
                anchor_h = anchors[a][1]
                width = rel_w * anchor_w
                height = rel_h * anchor_h

                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2

                x1 = torch.clamp(x1, 0, image_size)
                y1 = torch.clamp(y1, 0, image_size)
                x2 = torch.clamp(x2, 0, image_size)
                y2 = torch.clamp(y2, 0, image_size)

                class_probs = pred_cls[b, a, :, y_idx, x_idx]
                class_id = torch.argmax(class_probs)
                class_score = class_probs[class_id]

                final_score = pred_obj[b, a, y_idx, x_idx] * class_score

                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                scores.append(final_score.item())
                labels.append(class_id.item())

        results.append({
            'boxes': torch.tensor(boxes) if boxes else torch.empty(0, 4),
            'scores': torch.tensor(scores) if scores else torch.empty(0),
            'labels': torch.tensor(labels) if labels else torch.empty(0)
        })

    return results


def yolo_loss(predictions, targets, coord_weight=5.0, noobj_weight=0.5):
    batch_size = predictions.shape[0]
    device = predictions.device

    pred_xy = predictions[:, :, 0:2]
    pred_wh = predictions[:, :, 2:4]
    pred_obj = predictions[:, :, 4:5]
    pred_cls = predictions[:, :, 5:]

    target_xy = targets[:, :, 0:2]
    target_wh = targets[:, :, 2:4]
    target_obj = targets[:, :, 4:5]
    target_cls = targets[:, :, 5:]

    obj_mask = target_obj > 0
    obj_mask_expanded = obj_mask.expand_as(pred_xy)

    if obj_mask.sum() > 0:
        xy_loss = F.binary_cross_entropy_with_logits(
            pred_xy[obj_mask_expanded],
            target_xy[obj_mask_expanded],
            reduction='sum'
        )

        wh_loss = F.mse_loss(
            pred_wh[obj_mask_expanded],
            target_wh[obj_mask_expanded],
            reduction='sum'
        )
    else:
        xy_loss = torch.tensor(0.0, device=device)
        wh_loss = torch.tensor(0.0, device=device)

    objectness_loss = focal_loss(
        pred_obj.reshape(-1),
        target_obj.reshape(-1),
        alpha=0.25,
        gamma=2.0,
        reduction='sum'
    )

    if obj_mask.sum() > 0:
        obj_mask_cls = obj_mask.expand_as(pred_cls)
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_cls[obj_mask_cls],
            target_cls[obj_mask_cls],
            reduction='sum'
        )
    else:
        cls_loss = torch.tensor(0.0, device=device)

    total_loss = (
        coord_weight * (xy_loss + wh_loss) +
        objectness_loss +
        cls_loss
    ) / batch_size

    return {
        'total_loss': total_loss,
        'xy_loss': xy_loss / batch_size,
        'wh_loss': wh_loss / batch_size,
        'obj_loss': objectness_loss / batch_size,
        'cls_loss': cls_loss / batch_size
    }


def remove_bg(img, bboxes):
    if type(img) == str:
        original = Image.open(img).convert("RGBA")
    else:
        original = Image.fromarray(img).convert("RGBA")

    original_with_boxes = original.copy()
    draw = ImageDraw.Draw(original_with_boxes)

    mask = Image.new("L", original.size, 255)
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask.paste(0, (x1, y1, x2, y2))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    white_bg = Image.new("RGBA", original.size, (255, 255, 255, 255))
    result = Image.composite(white_bg, original, mask)

    final_result = Image.new("RGB", result.size, (255, 255, 255))
    final_result.paste(result, (0, 0), result)

    return original_with_boxes, final_result
