import os
import random
import json
import torch
from torchvision import transforms
from PIL import Image 
from transformers import CLIPImageProcessor

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, 
                 image_root_path="", face_id_dir="", face_img_dir="", control_img_dir="", parsing_mask_dir=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.face_id_dir = face_id_dir
        self.face_img_dir = face_img_dir
        self.control_img_dir = control_img_dir
        self.parsing_mask_dir = parsing_mask_dir  # New parameter for parsing masks
        # Load and filter data
        self.data = json.load(open(json_file))
        
        self.clip_image_processor = CLIPImageProcessor()
        data_ = []
        for item in self.data:
            image_file = item["face"][0]
            face_id_file = image_file.replace(".jpg", ".pt")
            control_img_file = item['image']
            # Check if required files exist
            valid = True
            if not (os.path.exists(os.path.join(self.face_id_dir, face_id_file)) 
                    and os.path.exists(os.path.join(self.control_img_dir, control_img_file))):
                valid = False
            # Check if all parsing masks exist
            for mask_info in item.get("parsing_masks", []):
                mask_path = os.path.join(self.parsing_mask_dir, mask_info["mask"])
                if not os.path.exists(mask_path):
                    valid = False
                    break
            if valid:
                data_.append(item)
        self.data = data_
        print(f"{len(self.data)} images loaded.")

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.condition_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load images and text (existing code)
        text = item['face_caption'][0]
        image_file = item["image"]
        face_id_embed_file = item["face"][0].replace(".jpg", ".pt")
        
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        face_image = Image.open(os.path.join(self.face_img_dir, item["face"][0]))     
        contorlnet_image = Image.open(os.path.join(self.control_img_dir, image_file))
        contorlnet_image = self.condition_transform(contorlnet_image)
        image = self.transform(raw_image.convert("RGB"))
        clip_image_embed = self.clip_image_processor(images=face_image, return_tensors="pt")['pixel_values']
        
        try:
            face_id_embed = torch.load(os.path.join(self.face_id_dir, face_id_embed_file), map_location="cpu")
        except:
            face_id_embed = torch.zeros((1, 512))
        
        # Random drop for clip embedding (existing code)
        drop_image_embed = 0
        rand_num = random.randint(0, 9)
        if rand_num < 5:
            drop_image_embed = 1
        if drop_image_embed:
            clip_image_embed = torch.zeros_like(clip_image_embed)
        
        # Tokenize text (existing code)
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Process parsing masks
        parsing_masks = []
        parsing_token_indices = []
        for mask_info in item.get("parsing_masks", []):
            # Load mask image
            mask_path = os.path.join(self.parsing_mask_dir, mask_info["mask"])
            mask_image = Image.open(mask_path)
            # Apply transformations and binarize
            mask_tensor = self.condition_transform(mask_image.convert("L"))  # Convert to grayscale
            mask_tensor = (mask_tensor > 0.5).float()  # Binarize
            parsing_masks.append(mask_tensor)
            parsing_token_indices.append(mask_info["token_indices"])

        return {
            "image": image,
            "contorlnet_image": contorlnet_image,
            "text_input_ids": text_input_ids,
            "face_id_embed": face_id_embed,
            "clip_image_embed": clip_image_embed,
            "drop_image_embed": drop_image_embed,
            "parsing_masks": parsing_masks,  # New fields
            "parsing_token_indices": parsing_token_indices,
        }

    def __len__(self):
        return len(self.data)

class Dataset4ShortCaption(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, 
                 image_root_path="", face_id_dir="", face_img_dir="", control_img_dir="", parsing_mask_dir=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.face_id_dir = face_id_dir
        self.face_img_dir = face_img_dir
        self.control_img_dir = control_img_dir
        self.parsing_mask_dir = parsing_mask_dir  # New parameter for parsing masks
        # Load and filter data
        self.data = json.load(open(json_file))

        self.clip_image_processor = CLIPImageProcessor()
        data_ = []
        for item in self.data:
            image_file = item["face"][0]
            face_id_file = image_file.replace(".jpg", ".pt")
            control_img_file = item['image']
            # Check if required files exist
            valid = True
            if not (os.path.exists(os.path.join(self.face_id_dir, face_id_file)) 
                    and os.path.exists(os.path.join(self.control_img_dir, control_img_file))):
                valid = False
            # Check if all parsing masks exist
            for mask_info in item.get("parsing_masks", []):
                mask_path = os.path.join(self.parsing_mask_dir, mask_info["mask"])
                if not os.path.exists(mask_path):
                    valid = False
                    break
            if valid:
                data_.append(item)
        self.data = data_
        print(f"{len(self.data)} images loaded.")

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.condition_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load images and text (existing code)
        text = item['face_short_caption']
        image_file = item["image"]
        face_id_embed_file = item["face"][0].replace(".jpg", ".pt")
        
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        face_image = Image.open(os.path.join(self.face_img_dir, item["face"][0]))     
        contorlnet_image = Image.open(os.path.join(self.control_img_dir, image_file))
        contorlnet_image = self.condition_transform(contorlnet_image)
        image = self.transform(raw_image.convert("RGB"))
        clip_image_embed = self.clip_image_processor(images=face_image, return_tensors="pt")['pixel_values']
        
        try:
            face_id_embed = torch.load(os.path.join(self.face_id_dir, face_id_embed_file), map_location="cpu")
        except:
            face_id_embed = torch.zeros((1, 512))
        
        # Random drop for clip embedding (existing code)
        drop_image_embed = 0
        rand_num = random.randint(0, 9)
        if rand_num < 5:
            drop_image_embed = 1
        if drop_image_embed:
            clip_image_embed = torch.zeros_like(clip_image_embed)
        
        # Tokenize text (existing code)
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Process parsing masks
        parsing_masks = []
        parsing_token_indices = []
        for mask_info in item.get("parsing_masks", []):
            # Load mask image
            mask_path = os.path.join(self.parsing_mask_dir, mask_info["mask"])
            mask_image = Image.open(mask_path)
            # Apply transformations and binarize
            mask_tensor = self.condition_transform(mask_image.convert("L"))  # Convert to grayscale
            mask_tensor = (mask_tensor > 0.5).float()  # Binarize
            parsing_masks.append(mask_tensor)
            parsing_token_indices.append(mask_info["token_indices"])

        return {
            "image": image,
            "contorlnet_image": contorlnet_image,
            "text_input_ids": text_input_ids,
            "face_id_embed": face_id_embed,
            "clip_image_embed": clip_image_embed,
            "drop_image_embed": drop_image_embed,
            "parsing_masks": parsing_masks,  # New fields
            "parsing_token_indices": parsing_token_indices,
        }

    def __len__(self):
        return len(self.data)

class Dataset4ShortCaptionDropText(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, 
                 image_root_path="", face_id_dir="", face_img_dir="", control_img_dir="", parsing_mask_dir=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.face_id_dir = face_id_dir
        self.face_img_dir = face_img_dir
        self.control_img_dir = control_img_dir
        self.parsing_mask_dir = parsing_mask_dir  # New parameter for parsing masks
        # Load and filter data
        self.data = json.load(open(json_file))

        self.clip_image_processor = CLIPImageProcessor()
        data_ = []
        for item in self.data:
            image_file = item["face"][0]
            face_id_file = image_file.replace(".jpg", ".pt")
            control_img_file = item['image']
            # Check if required files exist
            valid = True
            if not (os.path.exists(os.path.join(self.face_id_dir, face_id_file)) 
                    and os.path.exists(os.path.join(self.control_img_dir, control_img_file))):
                valid = False
            # Check if all parsing masks exist
            for mask_info in item.get("parsing_masks", []):
                mask_path = os.path.join(self.parsing_mask_dir, mask_info["mask"])
                if not os.path.exists(mask_path):
                    valid = False
                    break
            if valid:
                data_.append(item)
        self.data = data_
        print(f"{len(self.data)} images loaded.")

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.condition_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load images and text (existing code)
        text = item['face_short_caption']
        if random.random() < self.t_drop_rate:
            text = ""
        image_file = item["image"]
        face_id_embed_file = item["face"][0].replace(".jpg", ".pt")
        
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        face_image = Image.open(os.path.join(self.face_img_dir, item["face"][0]))     
        contorlnet_image = Image.open(os.path.join(self.control_img_dir, image_file))
        contorlnet_image = self.condition_transform(contorlnet_image)
        image = self.transform(raw_image.convert("RGB"))
        clip_image_embed = self.clip_image_processor(images=face_image, return_tensors="pt")['pixel_values']
        
        try:
            face_id_embed = torch.load(os.path.join(self.face_id_dir, face_id_embed_file), map_location="cpu")
        except:
            face_id_embed = torch.zeros((1, 512))
        
        # Random drop for clip embedding (existing code)
        drop_image_embed = 0
        rand_num = random.randint(0, 9)
        if rand_num < 5:
            drop_image_embed = 1
        if drop_image_embed:
            clip_image_embed = torch.zeros_like(clip_image_embed)
        
        # Tokenize text (existing code)
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Process parsing masks
        parsing_masks = []
        parsing_token_indices = []
        for mask_info in item.get("parsing_masks", []):
            # Load mask image
            mask_path = os.path.join(self.parsing_mask_dir, mask_info["mask"])
            mask_image = Image.open(mask_path)
            # Apply transformations and binarize
            mask_tensor = self.condition_transform(mask_image.convert("L"))  # Convert to grayscale
            mask_tensor = (mask_tensor > 0.5).float()  # Binarize
            parsing_masks.append(mask_tensor)
            parsing_token_indices.append(mask_info["token_indices"])

        return {
            "image": image,
            "contorlnet_image": contorlnet_image,
            "text_input_ids": text_input_ids,
            "face_id_embed": face_id_embed,
            "clip_image_embed": clip_image_embed,
            "drop_image_embed": drop_image_embed,
            "parsing_masks": parsing_masks,  # New fields
            "parsing_token_indices": parsing_token_indices,
        }

    def __len__(self):
        return len(self.data)

class Dataset4ShortCaptionRandomText(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 image_root_path="", face_id_dir="", face_img_dir="", control_img_dir="", parsing_mask_dir="",
                 sample_size=None, sample_seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.face_id_dir = face_id_dir
        self.face_img_dir = face_img_dir
        self.control_img_dir = control_img_dir
        self.parsing_mask_dir = parsing_mask_dir # 这个路径现在应该是 "Emotion" 和 "Attribute" 文件夹的父目录

        # 加载和过滤数据
        self.data = json.load(open(json_file))

        self.clip_image_processor = CLIPImageProcessor()

        # --- 数据过滤逻辑更新 ---
        data_ = []
        for item in self.data:
            # 基础文件检查
            image_file = item["face"][0]
            face_id_file = image_file.replace(".jpg", ".pt")
            control_img_file = item['image']
            if not (os.path.exists(os.path.join(self.face_id_dir, face_id_file))
                    and os.path.exists(os.path.join(self.control_img_dir, control_img_file))):
                continue

            # 检查 emotion 和 attribute 数据是否有效
            has_valid_emotion = False
            if "face_emotion_caption" in item and "emotion_parsing_masks" in item:
                has_valid_emotion = True
                for mask_info in item["emotion_parsing_masks"]:
                    mask_path = os.path.join(self.parsing_mask_dir, "Emotion", mask_info["mask"])
                    if not os.path.exists(mask_path):
                        has_valid_emotion = False
                        break

            has_valid_attribute = False
            if "face_attribute_caption" in item and "atrribute_parsing_masks" in item:
                has_valid_attribute = True
                for mask_info in item["atrribute_parsing_masks"]:
                    mask_path = os.path.join(self.parsing_mask_dir, "Attribute", mask_info["mask"])
                    if not os.path.exists(mask_path):
                        has_valid_attribute = False
                        break

            # 记录哪些数据源是可用的
            item["_valid_sources"] = []
            if has_valid_emotion:
                item["_valid_sources"].append("emotion")
            if has_valid_attribute:
                item["_valid_sources"].append("attribute")

            # 只有当至少有一个有效的数据源时，才保留该数据项
            if item["_valid_sources"]:
                data_.append(item)

        # Apply random sampling if specified
        if sample_size is not None and sample_size < len(data_):
            print(f"Randomly sampling {sample_size} samples from {len(data_)} total filtered samples")
            print(f"Using random seed: {sample_seed}")
            random.seed(sample_seed)
            data_ = random.sample(data_, sample_size)
        elif sample_size is not None and sample_size >= len(data_):
            print(f"Warning: Requested sample size ({sample_size}) is >= total filtered dataset size ({len(data_)}). Using full dataset.")

        self.data = data_
        print(f"{len(self.data)} images loaded.")

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.condition_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- 随机选择文本和掩码源 ---
        source_type = random.choice(item["_valid_sources"])

        if source_type == "emotion":
            text = item['face_emotion_caption']
            mask_list = item.get('emotion_parsing_masks', [])
            mask_subdir = "Emotion"
        else: # attribute
            text = item['face_attribute_caption']
            mask_list = item.get('atrribute_parsing_masks', [])
            mask_subdir = "Attribute"

        # 文本丢弃
        if random.random() < self.t_drop_rate:
            text = ""

        # 加载图像（与之前相同）
        image_file = item["image"]
        face_id_embed_file = item["face"][0].replace(".jpg", ".pt")
        
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        face_image = Image.open(os.path.join(self.face_img_dir, item["face"][0]))
        contorlnet_image = Image.open(os.path.join(self.control_img_dir, image_file))
        contorlnet_image = self.condition_transform(contorlnet_image)
        image = self.transform(raw_image.convert("RGB"))
        clip_image_embed = self.clip_image_processor(images=face_image, return_tensors="pt")['pixel_values']
        
        try:
            face_id_embed = torch.load(os.path.join(self.face_id_dir, face_id_embed_file), map_location="cpu")
        except:
            face_id_embed = torch.zeros((1, 512))
        
        # 随机丢弃 clip embedding（与之前相同）
        drop_image_embed = 0
        if random.random() < 0.5: # 简化了原先的 rand_num < 5
            drop_image_embed = 1
        if drop_image_embed:
            clip_image_embed = torch.zeros_like(clip_image_embed)
        
        # Tokenize 文本（与之前相同）
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # --- 根据选择的源处理掩码 ---
        parsing_masks = []
        parsing_token_indices = []
        for mask_info in mask_list:
            # 使用动态子目录加载掩码
            mask_path = os.path.join(self.parsing_mask_dir, mask_subdir, mask_info["mask"])
            mask_image = Image.open(mask_path)
            # 转换和二值化
            mask_tensor = self.condition_transform(mask_image.convert("L")) # 转为灰度图
            mask_tensor = (mask_tensor > 0.5).float() # 二值化
            parsing_masks.append(mask_tensor)
            parsing_token_indices.append(mask_info["token_indices"])

        return {
            "image": image,
            "contorlnet_image": contorlnet_image,
            "text_input_ids": text_input_ids,
            "face_id_embed": face_id_embed,
            "clip_image_embed": clip_image_embed,
            "drop_image_embed": drop_image_embed,
            "parsing_masks": parsing_masks,
            "parsing_token_indices": parsing_token_indices,
        }

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    contorlnet_images = torch.stack([example["contorlnet_image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    face_id_embed = torch.stack([example["face_id_embed"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    clip_image_embed = torch.cat([example["clip_image_embed"] for example in data], dim=0)
    
    # New fields: parsing masks and token indices
    parsing_masks = [example["parsing_masks"] for example in data]
    parsing_token_indices = [example["parsing_token_indices"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "contorlnet_image": contorlnet_images,
        "face_id_embed": face_id_embed,
        "clip_image_embed": clip_image_embed,
        "drop_image_embeds": drop_image_embeds,
        "parsing_masks": parsing_masks,  # List of lists of tensors
        "parsing_token_indices": parsing_token_indices,  # List of lists of token indices
    }
