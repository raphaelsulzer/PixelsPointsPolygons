def get_model(cfg,**kwargs):
    
    if cfg.model.name == 'pix2poly':
        return get_pix2poly_model(cfg,kwargs["tokenizer"])
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented")
        


def get_pix2poly_model(cfg,tokenizer):
    
    from .pix2poly import EncoderDecoder, ImageEncoder, LiDAREncoder, MultiEncoder, Decoder

    if cfg.use_images and cfg.use_lidar:
        encoder = MultiEncoder(cfg)
    elif cfg.use_images:
        encoder = ImageEncoder(cfg)
    elif cfg.use_lidar: 
        encoder = LiDAREncoder(cfg)
    else:
        raise ValueError("At least one of use_image or use_lidar must be True")
    
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        encoder_len=cfg.model.encoder.num_patches,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_len=cfg.model.tokenizer.max_len,
        pad_idx=cfg.model.tokenizer.pad_idx,
    )
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        cfg=cfg
    )
    model.to(cfg.device)
    
    return model
