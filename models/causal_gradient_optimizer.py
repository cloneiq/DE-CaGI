import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import warnings
import re
import logging


def get_encoder_layer_param_groups(
    model: nn.Module,
    trainable_ratio: float = 0.4,
) -> Dict[str, List[str]]:
    """
    Return parameter name patterns for training the last fraction of encoder layers.

    This inspects the model parameter names to infer layer indices for the vision and language
    encoders, then returns patterns for the last `trainable_ratio` portion of layers. If the
    layer structure cannot be inferred, it falls back to coarse patterns.

    Args:
        model: model instance
        trainable_ratio: fraction of layers to keep trainable (e.g., 0.4 = last 40%)

    Returns:
        A dict with keys 'image_encoder' and 'text_encoder', each containing a list of
        parameter-name patterns.
    """
    encoder_param_groups = {
        'image_encoder': [],
        'text_encoder': [],
    }
    
    all_param_names = [name for name, _ in model.named_parameters()]
    
    # Vision encoder layer patterns:
    # - vision_encoder.transformer.resblocks.X (CLIP ViT)
    # - vision_encoder.trunk.transformer.resblocks.X (open_clip)
    # - vision_encoder.blocks.X (other ViT variants)
    
    vision_layer_dict = {}  # {layer_idx: layer_prefix}
    for name in all_param_names:
        if 'vision_encoder' in name:
            match = re.search(r'(transformer\.resblocks|trunk\.transformer\.resblocks|blocks)\.(\d+)', name)
            if match:
                layer_idx = int(match.group(2))
                layer_prefix = match.group(0)  # e.g., "transformer.resblocks.5"
                if layer_idx not in vision_layer_dict:
                    vision_layer_dict[layer_idx] = layer_prefix
    
    if vision_layer_dict:
        sorted_layers = sorted(vision_layer_dict.items())
        total_layers = len(sorted_layers)
        start_idx = int(total_layers * (1 - trainable_ratio))
        
        for layer_idx, layer_prefix in sorted_layers[start_idx:]:
            pattern = f'vision_encoder.{layer_prefix}'
            encoder_param_groups['image_encoder'].append(pattern)
        
        encoder_param_groups['image_encoder'].append('multi_modal_vision_proj')
        
        logger = logging.getLogger(__name__)
        logger.info(
            f"Vision encoder: total_layers={total_layers}, "
            f"trainable_layers={total_layers - start_idx} (last {100*trainable_ratio:.0f}%), "
            f"layer_index_range={sorted_layers[start_idx][0]}-{sorted_layers[-1][0]}"
        )
    
    # Text encoder layer pattern: language_encoder.encoder.layer.X (BERT-like)
    text_layer_dict = {}  # {layer_idx: layer_prefix}
    for name in all_param_names:
        if 'language_encoder' in name:
            match = re.search(r'encoder\.layer\.(\d+)', name)
            if match:
                layer_idx = int(match.group(1))
                layer_prefix = f'encoder.layer.{layer_idx}'
                if layer_idx not in text_layer_dict:
                    text_layer_dict[layer_idx] = layer_prefix
    
    if text_layer_dict:
        sorted_layers = sorted(text_layer_dict.items())
        total_layers = len(sorted_layers)
        start_idx = int(total_layers * (1 - trainable_ratio))
        
        for layer_idx, layer_prefix in sorted_layers[start_idx:]:
            pattern = f'language_encoder.{layer_prefix}'
            encoder_param_groups['text_encoder'].append(pattern)
        
        encoder_param_groups['text_encoder'].append('multi_modal_language_proj')
        
        logger = logging.getLogger(__name__)
        logger.info(
            f"Text encoder: total_layers={total_layers}, "
            f"trainable_layers={total_layers - start_idx} (last {100*trainable_ratio:.0f}%), "
            f"layer_index_range={sorted_layers[start_idx][0]}-{sorted_layers[-1][0]}"
        )
    
    if not encoder_param_groups['image_encoder'] and not encoder_param_groups['text_encoder']:
        warnings.warn(
            "Failed to infer encoder layer structure; falling back to all parameters. "
            "Please check the model structure or specify parameter patterns manually."
        )
        return {
            'image_encoder': ['vision_encoder', 'multi_modal_vision_proj'],
            'text_encoder': ['language_encoder', 'multi_modal_language_proj'],
        }
    
    return encoder_param_groups


class GradientEMA:
    """
    Exponential moving average (EMA) tracker for gradients.
    """
    def __init__(self, decay: float = 0.1):
        """
        Args:
            decay: smoothing factor in (0, 1]; larger values react faster (less smoothing)
        """
        self.decay = decay
        self.means = {}
        self.variances = {}

    def update(self, key: str, gradient: torch.Tensor) -> torch.Tensor:
        """
        Update EMA statistics for `key` and return the EMA mean (same shape as input).

        Args:
            key: unique identifier for the gradient stream
            gradient: current batch gradient tensor
        """
        if gradient is None:
            return None
            
        flat_grad = gradient.flatten()
        
        if key not in self.means:
            self.means[key] = flat_grad.clone()
            self.variances[key] = torch.zeros_like(flat_grad)
        else:
            mu_prev = self.means[key]
            v_prev = self.variances[key]
            
            mu_new = self.decay * flat_grad + (1 - self.decay) * mu_prev
            
            v_new = self.decay * (flat_grad - mu_new)**2 + (1 - self.decay) * v_prev
            
            self.means[key] = mu_new
            self.variances[key] = v_new
            
        return self.means[key].reshape(gradient.shape)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute an RBF (Gaussian) kernel matrix.

    \(K(x, y) = \exp(-||x-y||^2 / (2\sigma^2))\)

    Args:
        x: [N, D]
        y: [M, D]
        sigma: bandwidth

    Returns:
        Kernel matrix [N, M]
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    y_norm = (y ** 2).sum(dim=1, keepdim=True)  # [M, 1]
    xy = torch.matmul(x, y.t())  # [N, M]
    
    dist_sq = x_norm - 2 * xy + y_norm.t()  # [N, M]
    
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
    
    return kernel


def compute_kernelized_debias(
    g_main: torch.Tensor,
    bias_grads_list: List[torch.Tensor],
    sigma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Kernelized (non-linear) debiasing using an RBF kernel.

    Args expect flattened vectors. This removes a weighted combination of bias gradients from
    the main gradient using kernel similarities.

    Args:
        g_main: main gradient [D]
        bias_grads_list: list of bias gradients, each [D]
        sigma: RBF bandwidth
        eps: numerical stability term

    Returns:
        Debiased gradient [D]
    """
    if not bias_grads_list:
        return g_main
    
    G_B = torch.stack(bias_grads_list)  # [K, D]
    g_m = g_main.unsqueeze(0)  # [1, D]
    
    similarity = rbf_kernel(g_m, G_B, sigma=sigma)  # [1, K]
    
    K_BB = rbf_kernel(G_B, G_B, sigma=sigma)  # [K, K]
    
    K_BB_reg = K_BB + eps * torch.eye(K_BB.shape[0], device=K_BB.device, dtype=K_BB.dtype)
    
    try:
        alpha = torch.linalg.solve(K_BB_reg, similarity.transpose(0, 1))  # [K, 1]
    except RuntimeError:
        warnings.warn("Kernel system solve failed; falling back to pseudo-inverse.")
        alpha = torch.linalg.pinv(K_BB_reg) @ similarity.transpose(0, 1)  # [K, 1]
    
    weighted_bias = torch.matmul(alpha.t(), G_B)  # [1, D]
    
    g_clean = g_main - weighted_bias.squeeze(0)  # [D]
    
    return g_clean


def compute_bias_removed_gradient(
    vqa_gradients: Dict[str, torch.Tensor],
    bias_gradients: Dict[str, Dict[str, torch.Tensor]],
    encoder_names: List[str] = ['image_encoder', 'text_encoder'],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Stage 1: remove bias gradients from VQA gradients (kernelized debiasing).

    Args:
        vqa_gradients: VQA gradients per encoder
        bias_gradients: bias gradients per encoder
        encoder_names: encoder names
        eps: numerical stability term

    Returns:
        Debiased gradients per encoder
    """
    cleaned_gradients = {}
    
    for encoder_name in encoder_names:
        if encoder_name not in vqa_gradients:
            warnings.warn(f"Encoder '{encoder_name}' not found in VQA gradients; skipping.")
            continue
        
        g_v = vqa_gradients[encoder_name]
        
        if encoder_name not in bias_gradients or not bias_gradients[encoder_name]:
            cleaned_gradients[encoder_name] = g_v
            continue
        
        bias_grads_dict = bias_gradients[encoder_name]
        
        bias_grad_list = []
        for bias_name, bias_grad in bias_grads_dict.items():
            if bias_grad is not None:
                bias_grad_flat = bias_grad.flatten()
                if bias_grad_flat.shape != g_v.shape:
                    if bias_grad_flat.numel() < g_v.numel():
                        padding = torch.zeros(g_v.numel() - bias_grad_flat.numel(), device=bias_grad_flat.device, dtype=bias_grad_flat.dtype)
                        bias_grad_flat = torch.cat([bias_grad_flat, padding])
                    elif bias_grad_flat.numel() > g_v.numel():
                        bias_grad_flat = bias_grad_flat[:g_v.numel()]
                    else:
                        bias_grad_flat = bias_grad_flat.reshape(g_v.shape).flatten()
                bias_grad_list.append(bias_grad_flat)
        
        if not bias_grad_list:
            cleaned_gradients[encoder_name] = g_v
            continue
        
        g_v_flat = g_v.flatten()
        
        grad_std = g_v_flat.std().item()
        sigma = max(grad_std * 0.5, 0.1)
        
        g_v_clean_flat = compute_kernelized_debias(
            g_main=g_v_flat,
            bias_grads_list=bias_grad_list,
            sigma=sigma,
            eps=eps
        )
        
        cleaned_gradients[encoder_name] = g_v_clean_flat.reshape(g_v.shape)
    
    return cleaned_gradients


def find_pareto_optimal_direction(grads: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the Pareto-stationary direction for two gradients (MGDA for 2 tasks).

    Args:
        grads: [g_main, g_evidence], both flattened vectors

    Returns:
        Combined gradient vector
    """
    assert len(grads) == 2, "Only 2-gradient MGDA is supported (main vs evidence)."
    
    g1 = grads[0]
    g2 = grads[1]
    
    diff = g1 - g2
    
    diff_sq_norm = torch.dot(diff, diff)
    
    if diff_sq_norm < 1e-7:
        return 0.5 * g1 + 0.5 * g2
    
    numerator = torch.dot(g2, -diff)
    lambda_opt = numerator / diff_sq_norm
    
    lambda_star = torch.clamp(lambda_opt, 0.0, 1.0)
    
    g_pareto = lambda_star * g1 + (1.0 - lambda_star) * g2
    
    return g_pareto


def compute_evidence_aligned_update(
    cleaned_gradients: Dict[str, torch.Tensor],
    evidence_gradients: Dict[str, Dict[str, torch.Tensor]],
    encoder_names: List[str] = ['image_encoder', 'text_encoder'],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Stage 2: align debiased gradients with evidence gradients via MGDA (Pareto direction).

    Args:
        cleaned_gradients: debiased VQA gradients
        evidence_gradients: evidence-task gradients
        encoder_names: encoder names
        eps: kept for backward compatibility (currently unused)

    Returns:
        Final per-encoder gradients
    """
    final_gradients = {}
    
    for encoder_name in encoder_names:
        if encoder_name not in cleaned_gradients:
            continue
        
        g_v_clean = cleaned_gradients[encoder_name]
        g_clean_flat = g_v_clean.flatten()
        
        g_evidence_total = torch.zeros_like(g_clean_flat)
        has_evidence = False
        
        if encoder_name in evidence_gradients and evidence_gradients[encoder_name]:
            evidence_grads_dict = evidence_gradients[encoder_name]
            
            evidence_grad_list = []
            for ev_name, ev_grad in evidence_grads_dict.items():
                if ev_grad is not None:
                    ev_grad_flat = ev_grad.flatten()
                    if ev_grad_flat.shape == g_clean_flat.shape:
                        evidence_grad_list.append(ev_grad_flat)
            
            if evidence_grad_list:
                g_evidence_total = torch.stack(evidence_grad_list).mean(dim=0)
                has_evidence = True
        
        if not has_evidence:
            final_gradients[encoder_name] = g_v_clean
            continue

        g_pareto_flat = find_pareto_optimal_direction([g_clean_flat, g_evidence_total])
        
        final_gradients[encoder_name] = g_pareto_flat.reshape(g_v_clean.shape)
    
    return final_gradients


def extract_gradient_vector(
    model: nn.Module,
    param_name_patterns: List[str],
    reference_params: Optional[List[Tuple[str, torch.Tensor]]] = None,
) -> Optional[torch.Tensor]:
    """Extract a flattened gradient vector for parameters matching name patterns."""
    grad_list = []
    if reference_params is not None:
        for name, param in reference_params:
            if any(pattern in name for pattern in param_name_patterns) and param.requires_grad:
                if param.grad is not None:
                    grad_list.append(param.grad.flatten())
                else:
                    grad_list.append(torch.zeros_like(param.data.flatten()))
    else:
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in param_name_patterns) and param.requires_grad:
                if param.grad is not None:
                    grad_list.append(param.grad.flatten())
                else:
                    grad_list.append(torch.zeros_like(param.data.flatten()))
    
    if grad_list:
        return torch.cat(grad_list)
    return None


def collect_gradients(
    model: nn.Module,
    encoder_param_groups: Dict[str, List[str]],
    loss_dict: Dict[str, torch.Tensor],
    retain_graph: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """Collect gradients (helper)."""
    model.zero_grad()
    vqa_loss = loss_dict.get('vqa', None)
    if vqa_loss is None:
        raise ValueError("loss_dict must contain key 'vqa'")
    vqa_gradients = {}
    task_gradients = {enc_name: {} for enc_name in encoder_param_groups.keys()}
    # NOTE: This is a stub. The full implementation should use `extract_gradient_vector`.
    return vqa_gradients, task_gradients
