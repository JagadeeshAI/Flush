import torch
import timm
from peft import LoraConfig, get_peft_model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model(num_classes, model_path=None, device="cpu"):
    if model_path is not None:
        print(f"Loading model from checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
    else:
        print("Creating new DeiT tiny model with LoRA")
        model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
    
    # Make classifier head trainable
    if hasattr(model, 'base_model'):
        # For LoRA wrapped model
        if hasattr(model.base_model, 'head'):
            for param in model.base_model.head.parameters():
                param.requires_grad = True
        elif hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'head'):
            for param in model.base_model.model.head.parameters():
                param.requires_grad = True
    else:
        # For regular model
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
    
    total_params, trainable_params = count_parameters(model)
    trainable_percentage = (trainable_params / total_params) * 100
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_percentage:.2f}%")
    
    return model.to(device)