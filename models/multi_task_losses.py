import torch
import torch.nn.functional as F
from typing import Dict, Optional


def patchify(imgs, p):
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x



def compute_multitask_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    loss_weights = loss_weights or {}
    losses: Dict[str, torch.Tensor] = {}

    def _get_weight(name: str) -> float:
        return float(loss_weights.get(name, 1.0))

    if preds.get("vqa_logits") is not None and targets.get("vqa") is not None:
        vqa_logits = preds["vqa_logits"]
        vqa_target = targets["vqa"]
        
        if vqa_target.dtype not in (torch.float32, torch.float64):
            vqa_target = vqa_target.float()
        
        losses["vqa"] = F.binary_cross_entropy_with_logits(
            vqa_logits, 
            vqa_target, 
            reduction='mean'
        ) * _get_weight("vqa")

    if preds.get("modality_logits") is not None and targets.get("modality") is not None:
        modality_logits = preds["modality_logits"]
        modality_target = targets["modality"].long()
        
        losses["modality"] = F.cross_entropy(
            modality_logits,
            modality_target,
            reduction='mean'
        ) * _get_weight("modality")

    if preds.get("location_logits") is not None and targets.get("location") is not None:
        location_logits = preds["location_logits"]
        location_target = targets["location"].long()
        
        losses["location"] = F.cross_entropy(
            location_logits,
            location_target,
            reduction='mean'
        ) * _get_weight("location")

    if preds.get("seg_logits") is not None and targets.get("seg") is not None:
        seg_logits = preds["seg_logits"]  # (B, num_classes, H, W)
        seg_target = targets["seg"]
        
        if seg_target.dim() == 4:
            seg_target = seg_target.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        seg_target = seg_target.long()
        
        B, num_classes, H, W = seg_logits.shape
        seg_logits_flat = seg_logits.permute(0, 2, 3, 1).reshape(-1, num_classes)  # (B*H*W, num_classes)
        seg_target_flat = seg_target.reshape(-1)  # (B*H*W,)
        
        losses["seg"] = F.cross_entropy(
            seg_logits_flat,
            seg_target_flat,
            reduction='mean',
            ignore_index=-1
        ) * _get_weight("seg")

    if preds.get("det_cls_logit") is not None and preds.get("det_reg_pred") is not None:
        det_cls_logit = preds["det_cls_logit"]  # (B, num_classes, H, W)
        det_reg_pred = preds["det_reg_pred"]  # (B, 4, H, W)
        det_boxes = targets.get("det_boxes")  # list[Tensor(ni, 4)] in normalized [x1, y1, x2, y2]
        det_labels = targets.get("det_labels")  # list[Tensor(ni,)]
        
        if det_boxes is not None and det_labels is not None:
            loss_det_cls, loss_det_reg = compute_detection_loss(
                det_cls_logit, det_reg_pred, det_boxes, det_labels
            )
            losses["det_cls"] = loss_det_cls * _get_weight("det_cls")
            losses["det_reg"] = loss_det_reg * _get_weight("det_reg")

    if preds.get("mim_logits") is not None and preds.get("mim_mask") is not None:
        mim_pred = preds["mim_logits"]   # (B, L, p*p*3)
        mask = preds["mim_mask"]         # (B, L), 1=masked
        original_imgs = preds.get("original_imgs")  # (B, 3, H, W)
        patch_size = preds.get("patch_size", 16)
        
        if original_imgs is not None:
            target_patches = patchify(original_imgs, patch_size) 
            
            loss_mim = (mim_pred - target_patches) ** 2
            loss_mim = loss_mim.mean(dim=-1)  # (B, L)
            
            loss_mim = (loss_mim * mask).sum() / (mask.sum() + 1e-5)
            
            losses["mim"] = loss_mim * _get_weight("mim")

    if not losses:
        raise ValueError("No losses were computed. Please check preds/targets.")

    total_loss = sum(losses.values())
    losses["total"] = total_loss
    return losses


def compute_detection_loss(
    cls_logit: torch.Tensor,
    reg_pred: torch.Tensor,
    det_boxes: list,
    det_labels: list,
) -> tuple:

    B, num_classes, H, W = cls_logit.shape
    device = cls_logit.device
    
    loss_cls_list = []
    loss_reg_list = []
    num_pos = 0
    
    for i in range(B):
        boxes = det_boxes[i]  # (ni, 4)
        labels = det_labels[i]  # (ni,)
        
        if boxes.numel() == 0:
            continue
        
        for j in range(boxes.size(0)):
            box = boxes[j]  # normalized [x1, y1, x2, y2]
            cls_id = labels[j].item()
            
            cx_norm = (box[0] + box[2]) / 2.0
            cy_norm = (box[1] + box[3]) / 2.0
            
            cx = int(cx_norm * W)
            cy = int(cy_norm * H)
            
            cx = max(0, min(cx, W - 1))
            cy = max(0, min(cy, H - 1))
            
            cls_logit_ij = cls_logit[i, :, cy, cx].unsqueeze(0)  # (1, num_classes)
            target_cls = torch.tensor([cls_id], device=device, dtype=torch.long)
            loss_cls_list.append(F.cross_entropy(cls_logit_ij, target_cls))
            
            pred_reg_ij = reg_pred[i, :, cy, cx]  # (4,)
            target_reg = box.to(device)
            loss_reg_list.append(F.smooth_l1_loss(pred_reg_ij, target_reg))
            
            num_pos += 1
    
    if num_pos > 0:
        loss_cls = torch.stack(loss_cls_list).mean()
        loss_reg = torch.stack(loss_reg_list).mean()
    else:
        loss_cls = torch.tensor(0.0, device=device, requires_grad=True)
        loss_reg = torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss_cls, loss_reg


