import logging
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models.causal_gradient_optimizer import (
    compute_bias_removed_gradient,
    compute_evidence_aligned_update,
    extract_gradient_vector,
    get_encoder_layer_param_groups,
    GradientEMA,
)
from models.multi_task_losses import compute_multitask_loss
import numpy as np
import time
import warnings
from tqdm import tqdm

logger = logging.getLogger(__name__)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)  
    return scores


def train_epoch(model, data_loader, optimizer, scheduler, device, epoch,
                grad_clip=None, log_interval=10, config=None):
    """
    Training loop with causal gradient optimization (geometric projection).
    """
    model.train()

    multi_task_module = None
    if hasattr(model, 'multi_task_head'):
        multi_task_module = model.multi_task_head

    # Encoder parameter groups (used to extract/overwrite gradients)
    encoder_param_groups = {
        'image_encoder': ['vision_encoder', 'multi_modal_vision_proj'],
        'text_encoder': ['language_encoder', 'multi_modal_language_proj'],
    }

    grad_ema = GradientEMA(decay=config.get('ema_decay', 0.1) if config else 0.1)

    total_cls_loss = 0.0
    total_score = 0.0
    total_current = 0

    total_vqa_loss = 0.0
    total_modality_loss = 0.0
    total_location_loss = 0.0
    total_seg_loss = 0.0
    total_det_cls_loss = 0.0
    total_det_reg_loss = 0.0
    total_mim_loss = 0.0

    start_time = time.time()

    for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} training")):
        images = batch['image'].to(device)
        questions = batch['question']
        questions_ids = questions['input_ids'].to(device)
        attention_mask = questions['attention_mask'].to(device)

        targets = batch['target'].to(device)
        modality_targets = batch['modality_target'].to(device)
        location_targets = batch['location_target'].to(device)
        seg_targets = batch['seg_target'].to(device)
        
        det_boxes = batch.get('det_boxes', None)
        det_labels = batch.get('det_labels', None)

        batch_size = images.size(0)

        do_mim = config.get('enable_mim', True) if config else True
        logits, q_feats, v_feats, mim_logits, mim_mask, original_imgs = model(
            images,
            questions_ids,
            attention_mask,
            do_mim=do_mim
        )

        if multi_task_module is not None:
            image_global_feat = v_feats.mean(dim=1)
            text_global_feat = q_feats.mean(dim=1)
            multi_task_preds = multi_task_module(
                image_global_feat=image_global_feat,
                text_global_feat=text_global_feat,
                vision_spatial_feat=v_feats
            )
        else:
            multi_task_preds = {}

        if multi_task_preds is None:
            multi_task_preds = {}

        multi_task_preds['vqa_logits'] = logits
        
        if mim_logits is not None and mim_mask is not None and original_imgs is not None:
            multi_task_preds['mim_logits'] = mim_logits
            multi_task_preds['mim_mask'] = mim_mask
            multi_task_preds['original_imgs'] = original_imgs
            multi_task_preds['patch_size'] = model.patch_size if hasattr(model, 'patch_size') else 16

        multitask_targets = {
            'vqa': targets,
            'modality': modality_targets,
            'location': location_targets,
            'seg': seg_targets,
        }
        
        if det_boxes is not None and det_labels is not None:
            det_boxes_device = [box.to(device) for box in det_boxes]
            det_labels_device = [label.to(device) for label in det_labels]
            multitask_targets['det_boxes'] = det_boxes_device
            multitask_targets['det_labels'] = det_labels_device

        multitask_losses = compute_multitask_loss(
            preds=multi_task_preds,
            targets=multitask_targets,
            loss_weights=config.get('task_loss_weights', {})
        )

        vqa_loss = multitask_losses.get('vqa', multitask_losses['total'])

        # Bias losses (I-only / Q-only) for bias removal
        image_global_feat = v_feats.mean(dim=1)
        text_global_feat = q_feats.mean(dim=1)
        
        # I-only: image only (replace text with zeros)
        dummy_text_feat = torch.zeros_like(text_global_feat)
        i_only_fused_feat = torch.cat([dummy_text_feat, image_global_feat], dim=-1)
        i_only_logits = model.vqa_head(i_only_fused_feat)
        
        # Q-only: text only (replace image with zeros)
        dummy_image_feat = torch.zeros_like(image_global_feat)
        q_only_fused_feat = torch.cat([text_global_feat, dummy_image_feat], dim=-1)
        q_only_logits = model.vqa_head(q_only_fused_feat)

        i_only_preds = {'vqa_logits': i_only_logits}
        i_only_targets = {'vqa': targets}
        i_only_losses = compute_multitask_loss(
            preds=i_only_preds,
            targets=i_only_targets,
            loss_weights={'vqa': 1.0}
        )
        i_only_loss = i_only_losses['vqa']

        q_only_preds = {'vqa_logits': q_only_logits}
        q_only_targets = {'vqa': targets}
        q_only_losses = compute_multitask_loss(
            preds=q_only_preds,
            targets=q_only_targets,
            loss_weights={'vqa': 1.0}
        )
        q_only_loss = q_only_losses['vqa']

        model.zero_grad()
        vqa_loss.backward(retain_graph=True)

        vqa_gradients = {}
        vqa_param_refs = {}

        for enc_name, param_patterns in encoder_param_groups.items():
            param_refs = []
            for name, param in model.named_parameters():
                if any(pattern in name for pattern in param_patterns) and param.requires_grad:
                    param_refs.append((name, param))
            vqa_param_refs[enc_name] = param_refs

            grad_vec = extract_gradient_vector(model, param_patterns, reference_params=param_refs)
            if grad_vec is not None:
                vqa_gradients[enc_name] = grad_vec

        bias_gradients = {'image_encoder': {}, 'text_encoder': {}}

        model.zero_grad()
        i_only_loss.backward(retain_graph=True)
        i_only_grad = extract_gradient_vector(
            model,
            encoder_param_groups['image_encoder'],
            reference_params=vqa_param_refs.get('image_encoder', None)
        )
        if i_only_grad is not None:
            bias_gradients['image_encoder']['i_only'] = i_only_grad

        model.zero_grad()
        q_only_loss.backward(retain_graph=True)
        q_only_grad = extract_gradient_vector(
            model,
            encoder_param_groups['text_encoder'],
            reference_params=vqa_param_refs.get('text_encoder', None)
        )
        if q_only_grad is not None:
            bias_gradients['text_encoder']['q_only'] = q_only_grad

        evidence_gradients = {'image_encoder': {}, 'text_encoder': {}}

        if 'modality' in multitask_losses:
            model.zero_grad()
            multitask_losses['modality'].backward(retain_graph=True)
            modality_grad = extract_gradient_vector(
                model, encoder_param_groups['image_encoder']
            )
            if modality_grad is not None:
                evidence_gradients['image_encoder']['modality'] = modality_grad

        if 'location' in multitask_losses:
            model.zero_grad()
            multitask_losses['location'].backward(retain_graph=True)
            location_grad = extract_gradient_vector(
                model, encoder_param_groups['image_encoder']
            )
            if location_grad is not None:
                evidence_gradients['image_encoder']['location'] = location_grad

        if 'seg' in multitask_losses:
            model.zero_grad()
            multitask_losses['seg'].backward(retain_graph=True)
            seg_grad = extract_gradient_vector(
                model, encoder_param_groups['image_encoder']
            )
            if seg_grad is not None:
                evidence_gradients['image_encoder']['segmentation'] = seg_grad
        
        if 'mim' in multitask_losses:
            model.zero_grad()
            multitask_losses['mim'].backward(retain_graph=True)
            mim_grad = extract_gradient_vector(
                model, encoder_param_groups['image_encoder'],
                reference_params=vqa_param_refs.get('image_encoder', None)
            )
            if mim_grad is not None:
                evidence_gradients['image_encoder']['mim'] = mim_grad
        
        det_weight = config.get('task_loss_weights', {}).get('det_cls', 0.0)
        if 'det_cls' in multitask_losses or 'det_reg' in multitask_losses:
            if det_weight > 0:
                model.zero_grad()
                det_loss = multitask_losses.get('det_cls', torch.tensor(0.0, device=device))
                if 'det_reg' in multitask_losses:
                    det_loss = det_loss + multitask_losses['det_reg']
                if det_loss.item() > 0:
                    det_loss.backward(retain_graph=True)
                    det_grad = extract_gradient_vector(
                        model, encoder_param_groups['image_encoder']
                    )
                    if det_grad is not None:
                        evidence_gradients['image_encoder']['detection'] = det_grad

        for enc_name, bias_dict in bias_gradients.items():
            for bias_type, grad in bias_dict.items():
                bias_gradients[enc_name][bias_type] = grad_ema.update(f"{enc_name}_bias_{bias_type}", grad)
        
        for enc_name, evi_dict in evidence_gradients.items():
            for evi_type, grad in evi_dict.items():
                evidence_gradients[enc_name][evi_type] = grad_ema.update(f"{enc_name}_evi_{evi_type}", grad)

        # Stage 1: bias removal
        cleaned_gradients = compute_bias_removed_gradient(
            vqa_gradients=vqa_gradients,
            bias_gradients=bias_gradients,
            encoder_names=list(encoder_param_groups.keys()),
        )

        # Stage 2: evidence alignment
        final_gradients = compute_evidence_aligned_update(
            cleaned_gradients=cleaned_gradients,
            evidence_gradients=evidence_gradients,
            encoder_names=list(encoder_param_groups.keys())
        )

        model.zero_grad()

        total_loss = multitask_losses.get('total', vqa_loss)
        total_loss.backward()

        with torch.no_grad():
            for enc_name, param_patterns in encoder_param_groups.items():
                if enc_name not in final_gradients:
                    continue

                corrected_grad = final_gradients[enc_name]
                if corrected_grad is None:
                    continue

                idx = 0
                for name, param in model.named_parameters():
                    if any(pattern in name for pattern in param_patterns) and param.requires_grad:
                        param_size = param.numel()
                        if idx + param_size <= corrected_grad.numel():
                            param.grad = corrected_grad[idx:idx+param_size].reshape(param.shape)
                            idx += param_size
                
                if idx != corrected_grad.numel():
                    warnings.warn(
                        f"Gradient vector size mismatch: {enc_name}, "
                        f"expected {corrected_grad.numel()}, used {idx}"
                    )

        optimizer.step()
        optimizer.zero_grad()

        total_current += batch_size
        total_score += compute_score_with_logits(logits, targets).sum().item()
        total_cls_loss += total_loss.item() * batch_size
        
        if 'vqa' in multitask_losses:
            total_vqa_loss += multitask_losses['vqa'].item() * batch_size
        if 'modality' in multitask_losses:
            total_modality_loss += multitask_losses['modality'].item() * batch_size
        if 'location' in multitask_losses:
            total_location_loss += multitask_losses['location'].item() * batch_size
        if 'seg' in multitask_losses:
            total_seg_loss += multitask_losses['seg'].item() * batch_size
        if 'det_cls' in multitask_losses:
            total_det_cls_loss += multitask_losses['det_cls'].item() * batch_size
        if 'det_reg' in multitask_losses:
            total_det_reg_loss += multitask_losses['det_reg'].item() * batch_size
        if 'mim' in multitask_losses:
            total_mim_loss += multitask_losses['mim'].item() * batch_size
        
        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            
            avg_total_loss = total_cls_loss / total_current
            avg_vqa_loss = total_vqa_loss / total_current if total_current > 0 else 0.0
            avg_modality_loss = total_modality_loss / total_current if total_current > 0 else 0.0
            avg_location_loss = total_location_loss / total_current if total_current > 0 else 0.0
            avg_seg_loss = total_seg_loss / total_current if total_current > 0 else 0.0
            avg_det_cls_loss = total_det_cls_loss / total_current if total_current > 0 else 0.0
            avg_det_reg_loss = total_det_reg_loss / total_current if total_current > 0 else 0.0
            avg_mim_loss = total_mim_loss / total_current if total_current > 0 else 0.0
            
            log_msg = (
                f"| iter {i + 1}/{len(data_loader)} | {elapsed * 1000 / log_interval:.2f} ms/iter | "
                f"loss {avg_total_loss:.4f} | "
                f"VQA {avg_vqa_loss:.5f} | "
                f"modality {avg_modality_loss:.5f} | "
                f"location {avg_location_loss:.5f} | "
                f"seg {avg_seg_loss:.4f} | "
            )
            if avg_det_cls_loss > 0 or avg_det_reg_loss > 0:
                log_msg += f"det_cls {avg_det_cls_loss:.4f} | det_reg {avg_det_reg_loss:.4f} | "
            if avg_mim_loss > 0:
                log_msg += f"MIM {avg_mim_loss:.4f} | "
            log_msg += (
                f"MedVQA score {total_score / total_current * 100:.2f}% | "
                f"lr {lr:.6f} | mode: causal-gradient"
            )
            logger.info(log_msg)
            start_time = time.time()

    if scheduler is not None:
        scheduler.step()
    
    avg_total_loss = total_cls_loss / total_current if total_current > 0 else 0.0
    avg_score = total_score / total_current if total_current > 0 else 0.0
    avg_vqa_loss = total_vqa_loss / total_current if total_current > 0 else 0.0
    avg_modality_loss = total_modality_loss / total_current if total_current > 0 else 0.0
    avg_location_loss = total_location_loss / total_current if total_current > 0 else 0.0
    avg_seg_loss = total_seg_loss / total_current if total_current > 0 else 0.0
    avg_det_cls_loss = total_det_cls_loss / total_current if total_current > 0 else 0.0
    avg_det_reg_loss = total_det_reg_loss / total_current if total_current > 0 else 0.0
    avg_mim_loss = total_mim_loss / total_current if total_current > 0 else 0.0
    
    loss_info = {
        'epoch': epoch,
        'total_cls_loss': avg_total_loss,
        'total_loss': avg_total_loss,
        'vqa_loss': avg_vqa_loss,
        'modality_loss': avg_modality_loss,
        'location_loss': avg_location_loss,
        'seg_loss': avg_seg_loss,
        'det_cls_loss': avg_det_cls_loss,
        'det_reg_loss': avg_det_reg_loss,
        'mim_loss': avg_mim_loss,
        'accuracy': avg_score,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    return avg_vqa_loss, avg_score, loss_info


def validate(model, data_loader, device):
    model.eval()
    
    total_loss = 0.0
    total_score = 0.0
    total_current = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="validating"):
            images = batch['image'].to(device)
            questions = batch['question']
            questions_ids = questions['input_ids'].to(device)
            attention_mask = questions['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            batch_size = images.size(0)
            
            logits, _, _, _, _, _ = model(
                images,
                questions_ids,
                attention_mask,
                do_mim=False
            )
            
            vqa_target = targets.float() if targets.dtype not in (torch.float32, torch.float64) else targets
            vqa_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, 
                vqa_target, 
                reduction='mean'
            )
            
            score = compute_score_with_logits(logits, targets).sum().item()
            
            total_loss += vqa_loss.item() * batch_size
            total_score += score
            total_current += batch_size
    
    avg_loss = total_loss / total_current if total_current > 0 else 0.0
    avg_score = (total_score / total_current * 100) if total_current > 0 else 0.0
    
    return avg_loss, avg_score
