import torch

def generate_square_subsequent_mask(sz,device):
    mask = (
        torch.triu(torch.ones((sz, sz), device=device)) == 1
    ).transpose(0, 1)

    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))

    return mask


def create_mask(tgt, pad_idx):
    """
    tgt shape: (N, L)
    """

    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device=tgt.device)
    # changing the type here from bool to float32 to get rid of the torch warning
    tgt_padding_mask = (tgt == pad_idx).to(dtype=tgt_mask.dtype)

    return tgt_mask, tgt_padding_mask


def compute_dynamic_cfg_vars(cfg,tokenizer):
    
    cfg.model.tokenizer.pad_idx = tokenizer.PAD_code
    cfg.model.tokenizer.max_len = cfg.model.tokenizer.n_vertices*2+2
    cfg.model.tokenizer.generation_steps = cfg.model.tokenizer.n_vertices*2+1
    cfg.model.encoder.num_patches = int((cfg.model.encoder.input_size // cfg.model.encoder.patch_size) ** 2)
    