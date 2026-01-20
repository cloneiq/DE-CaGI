import os
import json
import torch
import argparse
import logging

from train import train_epoch, validate
from utils.dataloader import VQADataLoader
from utils.get_optimizer import get_optimizer
from models.multi_task_model import CausalVQAModel
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VQADataLoader')


def parse_args():
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('--data_dir', type=str, default='dataset/slake', help='data directory')
    parser.add_argument('--image_dir', type=str, default='dataset/slake/imgs', help='image directory')
    parser.add_argument('--train_json', type=str, default='dataset/slake/train.json', help='train data json')
    parser.add_argument('--val_json', type=str, default='dataset/slake/validate.json', help='validate data json')
    parser.add_argument('--test_json', type=str, default='dataset/slake/test.json', help='test data json')

    parser.add_argument('--vocab', type=str, default='biomedbert', help='vocabulary')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')
    parser.add_argument('--max_length', type=int, default=32, help='max sequence length')
    parser.add_argument('--visual_backbone', type=str, default='ViT-B/16', help='visual backbone')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')
    parser.add_argument('--input_text_embed_size', type=int, default=768, help='input text embedding size')
    parser.add_argument('--input_image_embed_size', type=int, default=768, help='input image embedding size')
    parser.add_argument('--num_top_layer', type=int, default=4, help='number of top layers')
    parser.add_argument('--fusion_dropout', type=float, default=0.1, help='fusion dropout')
    parser.add_argument('--num_modalities', type=int, default=3, help='number of modalities')
    parser.add_argument('--num_locations', type=int, default=10, help='number of locations')
    parser.add_argument('--num_seg_classes', type=int, default=40, help='number of segmentation classes')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=3, help='gradient clip')
    parser.add_argument('--early_stop', type=int, default=50, help='early stop')
    parser.add_argument('--seed', type=int, default=105, help='seed')
    parser.add_argument('--log_interval', type=int, default=5, help='log interval')
    parser.add_argument('--device', type=str, default='cuda', help='device (leave empty for auto selection)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'infer'], help='running mode')
    parser.add_argument('--min_answer_freq', type=int, default=5, help='min answer frequency')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path (for evaluation or inference)')
    parser.add_argument('--rebuild_vocab', action='store_true', help='rebuild vocabulary')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='model save directory')


    parser.add_argument('--val_freq', type=int, default=1, help='validate frequency')
    
    parser.add_argument('--disable_mim', action='store_true', help='disable image masking task')

    # Training config / loss weights
    parser.add_argument('--ema_decay', type=float, default=0.1, help='EMA decay')
    parser.add_argument('--loss_weight_vqa', type=float, default=1.0, help='loss weight for VQA task')
    parser.add_argument('--loss_weight_modality', type=float, default=0.1, help='loss weight for modality task')
    parser.add_argument('--loss_weight_location', type=float, default=0.1, help='loss weight for location task')
    parser.add_argument('--loss_weight_seg', type=float, default=0.1, help='loss weight for segmentation task')
    parser.add_argument('--loss_weight_det_cls', type=float, default=0.1, help='loss weight for detection classification')
    parser.add_argument('--loss_weight_det_reg', type=float, default=0.1, help='loss weight for detection regression')
    parser.add_argument('--loss_weight_mim', type=float, default=0.1, help='loss weight for MIM task')

    return parser.parse_args()


def train(args, device):
    import torch.nn as nn
    import torch.optim as optim

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    logger.info("===== start training preparation =====")

    data_config = {
        'data_dir': args.data_dir,
        'image_dir': args.image_dir,
        'train_json': args.train_json,
        'val_json': args.val_json,
        'test_json': args.test_json,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'max_length': args.max_length,
        'tokenizer': args.vocab,
        'min_answer_freq': args.min_answer_freq,
        'rebuild_vocab': args.rebuild_vocab,
        'device': str(device),
        'mask_label_file': os.path.join(args.data_dir, 'mask.txt'),  # Mask label mapping file
        'detection_vocab_path': os.path.join(args.data_dir, 'detection_vocab.json'),  # Detection class vocabulary path
    }

    logger.info("initialize data loader...")
    data_loader = VQADataLoader(data_config)
    loaders = data_loader.get_loaders()
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')

    # Get the number of answer classes
    answer_vocab = data_loader.get_answer_vocab()
    num_classes = answer_vocab['vocab_size']
    logger.info(f"Number of answer classes: {num_classes}")
    
    # Load detection class vocabulary (if available)
    detection_vocab_path = os.path.join(data_config.get('data_dir', 'data'), 'detection_vocab.json')
    num_det_classes = 0
    if os.path.exists(detection_vocab_path):
        import json
        with open(detection_vocab_path, 'r', encoding='utf-8') as f:
            detection_vocab = json.load(f)
            num_det_classes = detection_vocab.get('num_classes', 0)
            logger.info(f"Loaded detection class vocabulary: {num_det_classes} classes")
    else:
        logger.warning(f"Detection class vocabulary not found: {detection_vocab_path}. Detection task will be disabled.")
    
    # Model configuration
    model_config = {
        'num_answer_classes': num_classes,
        'num_modalities': args.num_modalities,
        'num_locations': args.num_locations,
        'num_seg_classes': args.num_seg_classes,
        'num_det_classes': num_det_classes,  # Number of detection classes (including background)
        'visual_backbone': args.visual_backbone,
        'image_size': args.image_size,
        'patch_size': args.patch_size,  # MIM task uses this patch_size
        'hidden_size': args.hidden_size,
        'input_text_embed_size': args.input_text_embed_size,
        'input_image_embed_size': args.input_image_embed_size,
        'num_top_layer': args.num_top_layer,
        'fusion_dropout': args.fusion_dropout
    }
    
    enable_mim = not args.disable_mim
    train_config = {
        "task_loss_weights": {
            "vqa": args.loss_weight_vqa,
            "modality": args.loss_weight_modality,
            "location": args.loss_weight_location,
            "seg": args.loss_weight_seg,
            "det_cls": args.loss_weight_det_cls,
            "det_reg": args.loss_weight_det_reg,
            "mim": args.loss_weight_mim,
        },
        'ema_decay': args.ema_decay,
        'enable_mim': enable_mim,  # Enable MIM by default
    }

    # Initialize model
    logger.info("Initializing model...")
    model = CausalVQAModel(model_config)

    model = model.to(device)
    # Print model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = get_optimizer(model)

    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)

    epoch = 1
    best_val_score = 0
    early_stop_count = 0
    

    # Check whether validation data exists
    has_val_data = val_loader is not None and len(val_loader) > 0
    if not has_val_data:
        logger.warning("No validation set provided; the best checkpoint will be selected based on training loss.")

    # Training loop
    logger.info("Start training...")
    while epoch < args.epochs:
        # Train for one epoch
        model.train()

        # Run training for one epoch
        train_loss, train_acc, loss_info = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            config=train_config
        )
        
        
        run_validation = has_val_data and epoch % args.val_freq == 0
        if run_validation:
            val_loss, val_acc = validate(
                model=model,
                data_loader=val_loader,
                device=device
            )
            current_score = val_acc
            
            logger.info(
                f"Epoch {epoch}: train_loss {train_loss:.4f}, train_acc {train_acc*100:.2f}%, val_loss {val_loss:.4f}, val_acc {val_acc:.2f}%")

            if current_score > best_val_score:
                logger.info(f"Validation improved: {best_val_score:.2f}% -> {current_score:.2f}%. Saving checkpoint...")
                best_val_score = current_score

                best_model_path = os.path.join(args.save_dir, 'slake_de_Bconloss_model.pth')
                torch.save(model.state_dict(), best_model_path)

                early_stop_count = 0
            else:
                early_stop_count += 1
                logger.info(f"Validation did not improve. Early-stop counter: {early_stop_count}/{args.early_stop}")
        else:
            logger.info(f"Epoch {epoch}: train_loss {train_loss:.4f}, train_acc {train_acc:.2f}%")

        if args.early_stop > 0 and early_stop_count >= args.early_stop:
            logger.info(f"Early stopping triggered: no improvement for {args.early_stop} epochs. Stopping training.")
            break

        epoch += 1

    logger.info(f"Training finished. Best score: {best_val_score:.2f}%")

    
    return model


if __name__ == "__main__":
    import random
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
