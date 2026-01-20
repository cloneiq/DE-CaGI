import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from models.language_encoders.bert_model import BertCrossLayer
from transformers import BertConfig
from transformers import AutoModel
import open_clip
from models.utils import init_weights
from typing import Dict, Optional



class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SimpleSegmentationDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        mid_channels = max(hidden_channels // 2, 64)
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:

        if feature_map is None:
            raise ValueError("feature_map must not be None (segmentation requires a visual feature map).")

        x = self.decoder(feature_map)
        logits = self.classifier(x)

        if self.output_size is not None:
            logits = F.interpolate(
                logits, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
            )
        return logits


class SimpleDetHead(nn.Module):
    """
    A simple anchor-free detection head.
    """
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in [self.conv, self.cls_head, self.reg_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, feat_map: torch.Tensor) -> tuple:
        """
        Args:
            feat_map: (B, C, H, W) feature map
        
        Returns:
            cls_logit: (B, num_classes, H, W)
            reg_pred: (B, 4, H, W) in normalized (x1, y1, x2, y2)
        """
        x = self.conv(feat_map)
        cls_logit = self.cls_head(x)  # (B, num_classes, H, W)
        reg_pred = self.reg_head(x)  # (B, 4, H, W)
        return cls_logit, reg_pred


class MIMHead(nn.Module):
    """
    Masked Image Modeling (MIM) head.
    """
    def __init__(self, hidden_size, patch_size=16, in_chans=3):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.output_dim = patch_size * patch_size * in_chans
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, C) visual tokens without CLS
        
        Returns:
            (B, L, patch_size*patch_size*in_chans) patch pixel predictions
        """
        x = self.decoder(x)
        return x


class MultiTaskHead(nn.Module):
    """
    Multi-task heads used during training (auxiliary tasks).
    """
    def __init__(
        self,
        hidden_size: int,
        answer_num: int,
        modality_num: int,
        location_num: int,
        seg_num: int,
        feature_map_size: int,
        seg_head: nn.Module,
        modality_head: nn.Module,
        location_head: nn.Module,
        vqa_head: nn.Module,  # kept for interface compatibility (not used here)
        fusion_dropout: float = 0.1,
        det_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_map_size = feature_map_size
        self.seg_head = seg_head
        self.modality_head = modality_head
        self.location_head = location_head
        self.vqa_head = vqa_head
        self.det_head = det_head
        
    def forward(
        self,
        image_global_feat: torch.Tensor,
        text_global_feat: Optional[torch.Tensor] = None,
        vision_spatial_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_global_feat: (B, hidden_size)
            text_global_feat: (B, hidden_size), unused
            vision_spatial_feat: (B, L, hidden_size) token sequence including CLS (if present)
        
        Returns:
            Dict with keys like 'modality_logits', 'location_logits', 'seg_logits', and optional detection heads.
        """
        preds = {}
        
        preds['modality_logits'] = self.modality_head(image_global_feat)
        
        preds['location_logits'] = self.location_head(image_global_feat)
        
        if vision_spatial_feat is not None:
            B, L, C = vision_spatial_feat.shape
            spatial_tokens = vision_spatial_feat[:, 1:, :]  # (B, L-1, C)
            
            num_spatial_tokens = spatial_tokens.shape[1]
            actual_feature_map_size = int(num_spatial_tokens ** 0.5)
            
            if actual_feature_map_size * actual_feature_map_size == num_spatial_tokens:
                H = W = actual_feature_map_size
                spatial_feat_map = spatial_tokens.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                warnings.warn(
                    f"Segmentation token count {num_spatial_tokens} is not a perfect square; "
                    f"interpolating to {self.feature_map_size}x{self.feature_map_size}."
                )
                temp_size = actual_feature_map_size
                temp_feat_map = spatial_tokens[:, :temp_size*temp_size, :].permute(0, 2, 1).reshape(B, C, temp_size, temp_size)
                H = W = self.feature_map_size
                spatial_feat_map = F.interpolate(
                    temp_feat_map, size=(H, W), mode='bilinear', align_corners=False
                )
            
            preds['seg_logits'] = self.seg_head(spatial_feat_map)
            
            if self.det_head is not None:
                det_cls_logit, det_reg_pred = self.det_head(spatial_feat_map)
                preds['det_cls_logit'] = det_cls_logit
                preds['det_reg_pred'] = det_reg_pred
        else:
            preds['seg_logits'] = None
            if self.det_head is not None:
                preds['det_cls_logit'] = None
                preds['det_reg_pred'] = None
        
        return preds


class CausalVQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        answer_num = config.get('num_answer_classes', False)
        modality_num = config.get('num_modalities', False)
        location_num = config.get('num_locations', False)
        seg_num = config.get('num_seg_classes', False)
        self.config = config
        clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.vision_encoder = clip_model.visual
        bert_config = BertConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        self.language_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")



        self.embedding_dim = self.language_encoder.embeddings.word_embeddings.weight.shape[1]

        self.multi_modal_language_proj = nn.Linear(config.get('input_text_embed_size', 768),
                                                   config.get('hidden_size', 768))
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config.get('input_image_embed_size', 768),
                                                 config.get('hidden_size', 768))
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config.get("hidden_size", 768))
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config.get('num_top_layer', 6))])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config.get('num_top_layer', 6))])
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_language_pooler.apply(init_weights)


        self.vqa_head = nn.Sequential(
            nn.Linear(config.get("hidden_size", 768) * 2, config.get("hidden_size", 768) * 4),
            nn.LayerNorm(config.get("hidden_size", 768) * 4),
            nn.GELU(),
            nn.Linear(config.get("hidden_size", 768) * 4, answer_num),
        )
        self.vqa_head.apply(init_weights)

        hidden_size = config.get('hidden_size', 768)
        self.modality_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, modality_num),
        )
        self.modality_head.apply(init_weights)

        self.location_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, location_num),
        )
        self.location_head.apply(init_weights)

        self.feature_map_size = config.get('image_size', 224) // config.get('patch_size', 16)
        
        seg_input_channels = config.get('hidden_size', 768)
        self.seg_head = SimpleSegmentationDecoder(
            in_channels=seg_input_channels,
            num_classes=seg_num,
            output_size=config.get('image_size', 224),
            hidden_channels=256,
            dropout=config.get('fusion_dropout', 0.1)
        )
        
        det_num_classes = config.get('num_det_classes', None)
        if det_num_classes is not None and det_num_classes > 0:
            self.det_head = SimpleDetHead(
                in_channels=seg_input_channels,
                num_classes=det_num_classes,
                hidden_channels=256
            )
        else:
            self.det_head = None
        
        self.patch_size = config.get('patch_size', 16)
        self.mim_head = MIMHead(
            hidden_size=config.get('hidden_size', 768),
            patch_size=self.patch_size,
            in_chans=3
        )
        self.mim_head.apply(init_weights)
        
        self.multi_task_head = MultiTaskHead(
            hidden_size=config.get('hidden_size', 768),
            answer_num=answer_num,
            modality_num=modality_num,
            location_num=location_num,
            seg_num=seg_num,
            feature_map_size=self.feature_map_size,
            seg_head=self.seg_head,
            modality_head=self.modality_head,
            location_head=self.location_head,
            vqa_head=self.vqa_head,
            det_head=self.det_head,
            fusion_dropout=config.get('fusion_dropout', 0.1)
        )

    def random_masking(self, x, mask_ratio=0.5):
        """
        Random patch masking on images.

        Returns:
            x_masked: (B, C, H, W)
            mask: (B, L), 1=masked, 0=visible
            ids_restore: (B, L)
        """
        N, C, H, W = x.shape
        P = self.patch_size
        L = (H // P) * (W // P)
        
        noise = torch.rand(N, L, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        len_keep = int(L * (1 - mask_ratio))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        mask_img = mask.reshape(N, 1, H // P, W // P).float()
        mask_img = F.interpolate(mask_img, size=(H, W), mode='nearest')
        
        x_masked = x * (1 - mask_img)
        
        return x_masked, mask, ids_restore

    def forward(self, images, questions_ids, attention_mask, image_token_type_idx=1, do_mim=False):
        #  == Begin: Ori Text Encoding ==
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=questions_ids)
        text_input_shape = attention_mask.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(attention_mask, text_input_shape,
                                                                                questions_ids.device)

        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        q_feats = uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)


        # == Begin: Image Encoding ==
        uni_modal_image_feats = self.vision_encoder.trunk.forward_features(images)
        v_feats = uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        
        mim_logits = None
        mim_mask = None
        if self.training and do_mim:
            images_masked, mim_mask, _ = self.random_masking(images, mask_ratio=0.4)
            mim_image_feats = self.vision_encoder.trunk.forward_features(images_masked)
            mim_image_feats = self.multi_modal_vision_proj(mim_image_feats)
            seq_mim_feats = mim_image_feats[:, 1:, :]
            mim_logits = self.mim_head(seq_mim_feats)  # (B, L, P*P*3)
        else:
            images_masked = None
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long,
                                 device=images.device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                 images.device)
        # == End: Ori Image Encoding ==
        uni_modal_text_feats, uni_modal_image_feats= (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(attention_mask)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        )


        x, y = uni_modal_text_feats, uni_modal_image_feats
        for _, (text_layer, image_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                                 self.multi_modal_vision_layers)):
            x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]

        # # == End: do Co-Attention =
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        multi_modal_cls_feats = torch.cat(
            [multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
        logits = self.vqa_head(multi_modal_cls_feats)
        
        return logits, q_feats, v_feats, mim_logits, mim_mask, images
