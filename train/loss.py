import torch
import torch.nn.functional as F
# attention_maps:[batch_size*numheads,sequence_length,numtokens]
def get_map_loss(attention_maps, batch):
    ori_B = len(batch["parsing_masks"])
    if ori_B == 0:
        return torch.tensor(0.0, device=attention_maps.device)
        
    num_heads = attention_maps.size(0) // ori_B
    H = W = int((attention_maps.size(1))**0.5)
    attention_maps = attention_maps.view(ori_B, num_heads, H, W, -1)

    total_loss = []
    for batch_idx in range(ori_B):
        sample_masks = batch["parsing_masks"][batch_idx]
        sample_tokens = batch["parsing_token_indices"][batch_idx]
        
        for mask_idx, (mask, tokens) in enumerate(zip(sample_masks, sample_tokens)):
            # --- PATCH 1: Skip if tokens list is empty ---
            if not tokens:
                continue

            mask = F.interpolate(mask.unsqueeze(0), size=(H, W), mode='bilinear').squeeze(0)
            mask = (mask > 0.5).float().to(attention_maps.device)
            
            token_attn = attention_maps[batch_idx, :, :, :, tokens]
            attn_agg = token_attn.mean(dim=-1, keepdim=True)
            combined_attn = torch.cat([token_attn, attn_agg], dim=-1)
            
            mask_target = mask.unsqueeze(-1)
            mask_target = mask_target.expand(num_heads, H, W, len(tokens) + 1)
            
            # --- PATCH 2: Upcast to float32 for stable loss calculation ---
            # This is the most critical fix.
            loss = (combined_attn.float() - mask_target.float()).pow(2).mean(dim=(1, 2, 3))
            
            total_loss.append(loss.mean())

    if not total_loss:
        return torch.tensor(0.0, device=attention_maps.device)
    
    return torch.stack(total_loss).mean()

def compute_total_loss(attention_maps_list, batch):
    total_loss = torch.tensor(0.0, device=batch['images'].device)
    for idx, attn_maps in enumerate(attention_maps_list):
        layer_loss = get_map_loss(attn_maps, batch)
        
        # --- PATCH 3: Check for both isnan and isinf ---
        if layer_loss is not None and not torch.isnan(layer_loss) and not torch.isinf(layer_loss):
            layer_weight = 1.0
            total_loss += layer_weight * layer_loss
    
    return total_loss