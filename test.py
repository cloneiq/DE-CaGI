import torch
import argparse
import logging
from tqdm import tqdm
from models.multi_task_model import CausalVQAModel
from utils.dataloader import VQADataLoader
from train import compute_score_with_logits
from torch.nn import functional as F
import os
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('--data_dir', type=str, default='dataset/slake', help='data directory')
    parser.add_argument('--image_dir', type=str, default='dataset/slake/imgs', help='image directory')
    parser.add_argument('--train_json', type=str, default='dataset/slake/train_all_labeled.json', help='train data json')
    parser.add_argument('--val_json', type=str, default='dataset/slake/validate_labeled.json', help='validate data json')
    parser.add_argument('--test_json', type=str, default='dataset/slake/test_labeled.json', help='test data json')

    parser.add_argument('--vocab', type=str, default='biomedbert', help='vocabulary')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')
    parser.add_argument('--max_length', type=int, default=32, help='max sequence length')
    parser.add_argument('--load_path', type=str, default='pretrained_weights/m3ae.ckpt', help='load model path')
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


def test_accuracy(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"load checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    data_config = {
        'data_dir': args.data_dir,
        'image_dir': args.image_dir,
        'test_json': args.test_json,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'tokenizer': args.vocab,
        'image_size': args.image_size,
        'max_length': args.max_length,
        'device': str(device)
    }

    logger.info("initialize data loader...")
    data_loader = VQADataLoader(data_config)
    loaders = data_loader.get_loaders()
    test_loader = loaders.get('test')

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

    if test_loader is None or len(test_loader) == 0:
        logger.error("Test data loading failed!")
        return

    logger.info("initialize model...")
    model_config = {
        'num_answer_classes': data_loader.get_answer_vocab()['vocab_size'],
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
        'fusion_dropout': args.fusion_dropout,
        'load_path': args.load_path,
    }
    model = CausalVQAModel(model_config)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    logger.info("start evaluation...")
    criterion = F.binary_cross_entropy_with_logits
    
    with torch.no_grad():
        total_correct = 0
        total_loss = 0
        total_samples = 0
        closed_correct = 0  
        closed_total = 0    
        open_correct = 0    
        open_total = 0      
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch['image'].to(device)
            questions = batch['question']
            questions_ids = questions['input_ids'].to(device)
            attention_mask = questions['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            batch_size = images.size(0)

            answer_types = batch['answer_type']
            
            logits, _, _,_,_,_ = model(
                images,
                questions_ids,
                attention_mask
            )
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            pred_indices = torch.max(logits, 1)[1].data
            batch_scores = compute_score_with_logits(logits, targets)
            
            for i, (pred, score, ans_type) in enumerate(zip(pred_indices, batch_scores, answer_types)):
                total_samples += 1
                total_correct += score.sum().item()
                
                ans_upper = ans_type.upper()
                if ans_upper == 'CLOSED':
                    closed_total += 1
                    closed_correct += score.sum().item()
                else:
                    open_total += 1
                    open_correct += score.sum().item()

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    closed_accuracy = closed_correct / closed_total if closed_total > 0 else 0
    open_accuracy = open_correct / open_total if open_total > 0 else 0
    
    logger.info("=" * 50)
    logger.info("Test results:")
    logger.info(f"Overall test accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
    logger.info(f"Closed question accuracy: {closed_accuracy:.2%} ({closed_correct}/{closed_total})")
    logger.info(f"Open question accuracy: {open_accuracy:.2%} ({open_correct}/{open_total})")
    logger.info(f"Average loss: {total_loss / total_samples:.4f}")
    logger.info("=" * 50)

    return overall_accuracy


if __name__ == "__main__":
    args = parse_args()
    test_accuracy(args)
