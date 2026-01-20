import re
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer, RobertaTokenizer
from transformers import AutoTokenizer
from collections import Counter
import time
import logging
from tqdm import tqdm
import os
from typing import List, Dict



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VQADataLoader')

for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if env_var in os.environ:
        os.environ.pop(env_var)
os.environ['NO_PROXY'] = '*'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(answer):
    answer = str(answer)
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray')
    return answer



class VQADataset(Dataset):

    def __init__(self, data_dir, data_entries, image_dir, transform=None, max_length=32, tokenizer='biomedbert',
                 answer_vocab=None, mode='train', image_size=384, mask_label_file=None, detection_vocab=None):
        self.data_entries = data_entries
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.mode = mode

        if tokenizer == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif tokenizer == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  
        elif tokenizer == 'biomedbert':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer}")

        self.answer_vocab = answer_vocab
        
        self.mask_value_to_index = self._load_mask_label_mapping(mask_label_file)
        
        self.detection_vocab = detection_vocab


    def _load_mask_label_mapping(self, mask_label_file):
        """Load mask label mapping (map discrete grayscale values to contiguous indices)."""
        mask_value_to_index = {0: 0}  # 0 = background
        
        if mask_label_file and os.path.exists(mask_label_file):
            try:
                with open(mask_label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # mask.txt format: value:label_name
                for idx, line in enumerate(lines, start=1):
                    line = line.strip()
                    if ':' in line:
                        value_str = line.split(':')[0].strip()
                        try:
                            value = int(value_str)
                            mask_value_to_index[value] = idx  # start from 1; reserve 0 for background
                        except ValueError:
                            logger.warning(f"Failed to parse mask value: {value_str}")
                
                logger.info(f"Loaded {len(mask_value_to_index) - 1} mask label mappings (excluding background)")
            except Exception as e:
                logger.warning(f"Failed to load mask label file: {e}. Using default mapping.")
        else:
            default_values = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 95, 105, 110, 
                            120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 
                            185, 200, 205, 210, 215, 225, 230, 235, 240, 245, 250]
            for idx, value in enumerate(default_values, start=1):
                mask_value_to_index[value] = idx
            logger.info(f"Using default mask label mapping with {len(default_values)} labels")
        
        return mask_value_to_index

    def _get_tokenized(self, question):
        """Tokenize question text."""
        encoded = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def _load_image(self, img_path):
        """Load image as a tensor."""
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

    def _load_mask(self, mask_path):
        """Load mask and map discrete values to contiguous indices."""
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.long)
        
        if not mask_path or not os.path.exists(mask_path):
            return mask.unsqueeze(0)
        
        try:
            mask_img = Image.open(mask_path).convert('L')
            mask_array = np.array(mask_img)
            
            if mask_array.shape != (self.image_size, self.image_size):
                mask_img_resized = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
                mask_array = np.array(mask_img_resized)
            
            unique_values = np.unique(mask_array)
            for value in unique_values:
                value_int = int(value)
                if value_int in self.mask_value_to_index:
                    mask_array[mask_array == value_int] = self.mask_value_to_index[value_int]
                else:
                    mask_array[mask_array == value_int] = 0
            
            mask = torch.from_numpy(mask_array).long()
            return mask.unsqueeze(0)
            
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_path}: {e}")
            return mask.unsqueeze(0)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        """Get a single data sample."""
        entry = self.data_entries[idx]

        img_path = os.path.join(self.image_dir, entry['img_name'])
        
        image = self._load_image(img_path)

        question_text = entry['question']
        tokenized = self._get_tokenized(question_text)
        question = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }

        answer_text = preprocess_answer(entry['answer'])
        target = torch.zeros(self.answer_vocab['vocab_size'])
        answer_idx = self.answer_vocab['answer2idx'].get(answer_text, -1)
        if answer_idx >= 0:
            scores = self.answer_vocab['answer2score'].get(answer_text, 1.0)
            target[answer_idx] = scores

        modality_target = torch.tensor(entry.get('modality_label', 0), dtype=torch.long)
        location_target = torch.tensor(entry.get('location_label', 0), dtype=torch.long)

        mask_path = entry.get('mask_path', '')
        if mask_path:
            if not os.path.isabs(mask_path):
                if mask_path.startswith('imgs/'):
                    mask_path_clean = mask_path[5:]  # strip "imgs/" prefix
                    full_mask_path = os.path.join(self.image_dir, mask_path_clean)
                else:
                    full_mask_path = os.path.join(self.image_dir, mask_path)
                
                if not os.path.exists(full_mask_path):
                    parent_dir = os.path.dirname(self.image_dir)
                    full_mask_path = os.path.join(parent_dir, mask_path)
                
                mask_path = full_mask_path
        seg_target = self._load_mask(mask_path)

        det_boxes = []
        det_labels = []
        if self.detection_vocab is not None:
            detection = entry.get('detection', [])
            for det_item in detection:
                for class_name, box in det_item.items():
                    x, y, w, h = box
                    x1 = float(x)
                    y1 = float(y)
                    x2 = x1 + float(w)
                    y2 = y1 + float(h)
                    
                    x1_norm = x1 / self.image_size
                    y1_norm = y1 / self.image_size
                    x2_norm = x2 / self.image_size
                    y2_norm = y2 / self.image_size
                    
                    det_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                    
                    class_id = self.detection_vocab['class2id'].get(class_name, 0)  # 0 = background
                    det_labels.append(class_id)
        
        if det_boxes:
            det_boxes_tensor = torch.tensor(det_boxes, dtype=torch.float32)
            det_labels_tensor = torch.tensor(det_labels, dtype=torch.long)
        else:
            det_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            det_labels_tensor = torch.zeros((0,), dtype=torch.long)

        answer_type = entry.get('answer_type', 'OPEN')

        result = {
            'image': image,
            'question': question,
            'question_text': question_text,
            'target': target,
            'answer_idx': answer_idx,
            'answer_text': answer_text,
            'answer_type': answer_type,
            'image_path': img_path,
            'modality_target': modality_target,
            'location_target': location_target,
            'seg_target': seg_target,
            'det_boxes': det_boxes_tensor,
            'det_labels': det_labels_tensor,
        }

        return result


class VQADataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.image_dir = config.get('image_dir', os.path.join(self.data_dir, 'slake/imgs'))
        self.train_json = config.get('train_json', os.path.join(self.data_dir, 'slake/train.json'))
        self.val_json = config.get('val_json', os.path.join(self.data_dir, 'slake/validate.json'))
        self.test_json = config.get('test_json', os.path.join(self.data_dir, 'slake/test.json'))
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 16)
        self.image_size = config.get('image_size', 224)
        self.max_length = config.get('max_length', 32)
        self.tokenizer = config.get('tokenizer', 'biomedbert')
        self.min_answer_freq = config.get('min_answer_freq', 5)
        self.rebuild_vocab = config.get('rebuild_vocab', False)
        self.device = config.get('device', 'cuda')
        self.mask_label_file = config.get('mask_label_file', os.path.join(self.data_dir, 'mask.txt'))
        self.detection_vocab_path = config.get('detection_vocab_path', os.path.join(self.data_dir, 'detection_vocab.json'))

        self._init_transforms()
        self._load_data()
        self._build_answer_vocab()
        self._load_detection_vocab()
        self._init_datasets()
        self._init_loaders()

        if not config.get('keep_raw_data', False):
            self.train_data = None
            self.val_data = None
            self.test_data = None

    def _init_transforms(self):
        """Initialize image transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def _load_data(self):
        """Load datasets with basic validation."""
        def load_json(path, name):
            if not os.path.exists(path):
                logger.warning(f"{name} data file not found: {path}")
                return []

            logger.info(f"Loading {name} data: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} {name} samples")
                return data
            except Exception as e:
                logger.error(f"Failed to load {name} data: {e}")
                return []

        self.train_data = load_json(self.train_json, "train")
        self.val_data = load_json(self.val_json, "val")
        self.test_data = load_json(self.test_json, "test")

        if self.train_data:
            self._validate_data_format(self.train_data[0], "train")
        if self.val_data:
            self._validate_data_format(self.val_data[0], "val")
        if self.test_data:
            self._validate_data_format(self.test_data[0], "test")

    def _validate_data_format(self, entry, name):
        """Validate basic data format."""
        required_fields = ['question', 'img_name', 'img_id', 'answer']
        for field in required_fields:
            if field not in entry:
                logger.warning(f"{name} sample is missing required field: {field}")

        logger.info(f"{name} sample keys: {list(entry.keys())}")

    def _build_answer_vocab(self):
        """Build answer vocabulary and soft scores based on frequency."""
        vocab_path = os.path.join(self.data_dir, 'answer_vocab.json')

        if os.path.exists(vocab_path) and not self.rebuild_vocab:
            print(f"Loading existing answer vocab: {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.answer_vocab = json.load(f)

            if 'answer2score' not in self.answer_vocab:
                print("Answer vocab is missing frequency/score info. Adding defaults...")
                self.answer_vocab['answer2freq'] = {}
                self.answer_vocab['answer2score'] = {}

                for ans in self.answer_vocab['answer2idx'].keys():
                    if ans != '<UNK>':
                        self.answer_vocab['answer2freq'][ans] = 1
                        self.answer_vocab['answer2score'][ans] = 0.3

                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(self.answer_vocab, f, ensure_ascii=False, indent=2)

            print(f"Answer vocab size: {self.answer_vocab['vocab_size']}")
            return

        print("Building a new answer vocab...")

        answer_counter: Counter[str] = Counter()
        norm2raw: Dict[str, str] = {}  # normalized -> first observed raw form
        answer_idx2text: Dict[int, str] = {}

        def process_dataset(dataset: List[Dict], name: str) -> int:
            if not dataset:
                return 0
            cnt = 0
            for item in dataset:
                answer_key = "answer" if "answer" in item else "a" if "a" in item else None
                if not answer_key:
                    continue

                ans_field = item[answer_key]

                norm = preprocess_answer(ans_field)
                answer_counter[norm] += 1
                norm2raw.setdefault(norm, ans_field)
                cnt += 1
            return cnt

        train_cnt = process_dataset(self.train_data, "train")
        val_cnt = process_dataset(self.val_data, "val")
        test_cnt = process_dataset(self.test_data, "test")

        print(f"Collected {train_cnt} answers from train")
        print(f"Collected {val_cnt} answers from val")
        print(f"Collected {test_cnt} answers from test")
        print(f"Total unique normalized answers: {len(answer_counter)}")

        sorted_answers = sorted(answer_counter.items(), key=lambda kv: kv[1], reverse=True)
        print("Top 10 frequent normalized answers:", sorted_answers[:10])

        answer2idx: Dict[str, int] = {}
        idx2answer: Dict[int, str] = {}
        answer2freq: Dict[str, int] = {}
        answer2score: Dict[str, float] = {}

        for i, (norm_ans, freq) in enumerate(sorted_answers):
            answer2idx[norm_ans] = i
            idx2answer[i] = norm2raw[norm_ans]
            answer2freq[norm_ans] = freq

            if freq == 0:
                score = 0.0
            elif freq == 1:
                score = 0.3
            elif freq == 2:
                score = 0.6
            elif freq == 3:
                score = 0.9
            else:
                score = 1.0
            answer2score[norm_ans] = score

        self.answer_vocab = {
            "answer2idx": answer2idx,
            "idx2answer": idx2answer,
            "answer2freq": answer2freq,
            "answer2score": answer2score,
            "vocab_size": len(answer2idx),
        }

        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Saving answer vocab to: {vocab_path}")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.answer_vocab, f, ensure_ascii=False, indent=2)

    def _load_detection_vocab(self):
        """Load detection class vocabulary."""
        if os.path.exists(self.detection_vocab_path):
            logger.info(f"Loading detection vocab: {self.detection_vocab_path}")
            with open(self.detection_vocab_path, 'r', encoding='utf-8') as f:
                self.detection_vocab = json.load(f)
            logger.info(f"Number of detection classes: {self.detection_vocab.get('num_classes', 0)}")
        else:
            logger.warning(f"Detection vocab not found: {self.detection_vocab_path}. Detection task will be disabled.")
            self.detection_vocab = None

    def _init_datasets(self):
        """Initialize dataset objects."""
        if self.train_data:
            self.train_dataset = VQADataset(
                self.data_dir,
                self.train_data,
                self.image_dir,
                transform=self.transform,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='train',
                image_size=self.image_size,
                mask_label_file=self.mask_label_file,
                detection_vocab=getattr(self, 'detection_vocab', None)
            )
            logger.info(f"Train set size: {len(self.train_dataset)}")
        else:
            logger.warning("Train set is empty")
            self.train_dataset = None

        if self.val_data:
            self.val_dataset = VQADataset(
                self.data_dir,
                self.val_data,
                self.image_dir,
                transform=self.transform,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='val',
                image_size=self.image_size,
                mask_label_file=self.mask_label_file,
                detection_vocab=getattr(self, 'detection_vocab', None)
            )
            logger.info(f"Val set size: {len(self.val_dataset)}")
        else:
            logger.warning("Val set is empty")
            self.val_dataset = None

        if self.test_data:
            self.test_dataset = VQADataset(
                self.data_dir,
                self.test_data,
                self.image_dir,
                transform=self.transform,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='test',
                image_size=self.image_size,
                mask_label_file=self.mask_label_file,
                detection_vocab=getattr(self, 'detection_vocab', None)
            )
            logger.info(f"Test set size: {len(self.test_dataset)}")
        else:
            logger.info("Test set not provided")

    def _init_loaders(self):
        """Initialize data loaders."""
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                prefetch_factor=2 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0
            )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                persistent_workers=False,
                collate_fn=self.collate_fn,
                pin_memory=True
            )

        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=False,
                collate_fn=self.collate_fn,
                pin_memory=True
            )

    def collate_fn(self, batch):
        """Collate a list of samples into a mini-batch."""
        images = torch.stack([item['image'] for item in batch])
        input_ids = torch.stack([item['question']['input_ids'] for item in batch])
        attention_mask = torch.stack([item['question']['attention_mask'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        answer_indices = torch.tensor([item['answer_idx'] for item in batch], dtype=torch.long)
        modality_targets = torch.stack([item['modality_target'] for item in batch])
        location_targets = torch.stack([item['location_target'] for item in batch])
        seg_targets = torch.stack([item['seg_target'] for item in batch])
        
        det_boxes = [item['det_boxes'] for item in batch]  # list[Tensor(ni, 4)]
        det_labels = [item['det_labels'] for item in batch]  # list[Tensor(ni,)]
        
        question_texts = [item['question_text'] for item in batch]
        answer_texts = [item['answer_text'] for item in batch]
        answer_types = [item['answer_type'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        
        batch_dict = {
            'image': images,
            'question': {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            },
            'question_text': question_texts,
            'target': targets,
            'answer_idx': answer_indices,
            'answer_text': answer_texts,
            'answer_type': answer_types,
            'image_path': image_paths,
            'modality_target': modality_targets,
            'location_target': location_targets,
            'seg_target': seg_targets,
            'det_boxes': det_boxes,
            'det_labels': det_labels,
        }

        return batch_dict

    def get_loaders(self):
        """Get all available data loaders."""
        loaders = {}

        if hasattr(self, 'train_loader'):
            loaders['train'] = self.train_loader

        if hasattr(self, 'val_loader'):
            loaders['val'] = self.val_loader

        if hasattr(self, 'test_loader'):
            loaders['test'] = self.test_loader

        return loaders

    def get_answer_vocab(self):
        """Get answer vocabulary."""
        return self.answer_vocab

    def idx2answer(self, idx):
        """Convert answer index to text."""
        if isinstance(idx, int):
            idx_str = str(idx)
        else:
            idx_str = idx

        return self.answer_vocab['idx2answer'].get(idx_str, '<UNK>')

    def answer2idx(self, answer):
        """Convert answer text to index."""
        return self.answer_vocab['answer2idx'].get(answer, self.answer_vocab['answer2idx'].get('<UNK>', 0))

