
import torch  # pyright: ignore[reportMissingImports]
import safetensors.torch  # pyright: ignore[reportMissingImports]
import os, sys
OUT = sys.argv[1]
torch.manual_seed(42)
def L(n, k, scale=0.02): return torch.randn(n, k) * scale
# fp16 variant for the giant TE linears: real text-encoder checkpoints ship
# fp16/fp8, and fp16 keeps the fixtures small enough for the conversion battery
# while keeping the REAL TE dims (T5-XXL 4096/10240, Qwen2-0.5B 896/4864,
# Gemma-2-2B 2304/9216).
def L16(n, k, scale=0.02): return (torch.randn(n, k) * scale).to(torch.float16)

def make_sdxl():
    sd = {}; p = "model.diffusion_model."
    sd[p+"input_blocks.0.0.weight"] = torch.randn(320, 4, 3, 3) * 0.1
    sd[p+"input_blocks.0.0.bias"] = torch.randn(320) * 0.01
    for b in range(2):
        pre = f"{p}input_blocks.{b+1}.0.transformer_blocks.0."
        # attn1 keeps the real SDXL K=320: K % 256 != 0, so it must pass
        # through (CUDA ConvRot is 256-only); attn2 K=2048 quantizes.
        for a in ("attn1", "attn2"):
            sd[pre+a+".to_q.weight"] = L(320, 320 if a=="attn1" else 2048)
            sd[pre+a+".to_k.weight"] = L(320, 320 if a=="attn1" else 2048)
            sd[pre+a+".to_v.weight"] = L(320, 320 if a=="attn1" else 2048)
            sd[pre+a+".to_out.0.weight"] = L(320, 320)
        sd[pre+"ff.net.0.proj.weight"] = L(1280, 320, 0.01)
        sd[pre+"ff.net.2.weight"] = L(320, 1280, 0.01)
        sd[pre+"norm1.weight"] = torch.randn(320) * 0.1
    sd[p+"time_embed.0.weight"] = L(320, 320)
    sd[p+"out.2.weight"] = L(4, 320) * 0.1
    sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"] = L(49408, 768, 0.02)
    # text-encoder blocks (TE quantization test battery): CLIP-L under
    # conditioner.embedders.0. (HF naming transformer.text_model...) and
    # CLIP-G under conditioner.embedders.1.model. (open_clip naming
    # transformer.resblocks...), both real SDXL naming.
    _clip_text_blocks(sd, "conditioner.embedders.0.", blocks=1,
                      hidden=768, ffn=3072, vocab=49408)
    _clip_oc_text_blocks(sd, "conditioner.embedders.1.model.", blocks=1,
                         hidden=1280, ffn=5120)
    sd["first_stage_model.encoder.conv_in.weight"] = torch.randn(128, 3, 3, 3) * 0.1
    return sd

def make_sd15():
    sd = {}; p = "model.diffusion_model."
    sd[p+"input_blocks.0.0.weight"] = torch.randn(320, 4, 3, 3) * 0.1
    sd[p+"input_blocks.1.0.transformer_blocks.0.attn1.q.weight"] = torch.randn(320, 320, 1, 1) * 0.1
    sd[p+"input_blocks.1.0.transformer_blocks.0.attn1.k.weight"] = torch.randn(320, 320, 1, 1) * 0.1
    sd[p+"input_blocks.1.0.transformer_blocks.0.attn2.to_k.weight"] = L(320, 768)
    sd[p+"input_blocks.1.0.transformer_blocks.0.ff.net.0.proj.weight"] = L(2560, 320, 0.01)
    sd[p+"input_blocks.1.0.transformer_blocks.0.ff.net.2.weight"] = L(320, 2560, 0.01)
    sd[p+"time_embed.0.weight"] = L(320, 320)
    sd[p+"out.2.weight"] = L(4, 320) * 0.1
    # text-encoder (cond_stage_model) CLIP-L block, real SD1.5 naming:
    # cond_stage_model.transformer.text_model... + cond_stage_model.text_projection
    _clip_text_blocks(sd, "cond_stage_model.", blocks=1,
                      hidden=768, ffn=3072, vocab=49408)
    return sd

# ---------------------------------------------------------------------------
# Text-encoder fixture builders (TE quantization test battery).
#
# Naming follows the ComfyUI-native TE conventions from the TE quantization
# plan: T5-style `encoder.blocks.N.layer.M.{SelfAttention,DenseReluDense}`, CLIP
# style `transformer.text_model.encoder.layers.N.self_attn.*` (+ mlp fc1/fc2),
# and llama-style `model.layers.N.self_attn.*` / `model.layers.N.mlp.*`. Keys
# that must stay passthrough are included deliberately: token/positional
# embeddings, final_layer_norm, text_projection, T5 shared.weight,
# encoder.relative_attention_bias, T5 decoder.*, llama embed_tokens / model.norm
# / lm_head.
# ---------------------------------------------------------------------------

def _t5_encoder_block(sd, prefix, b, hidden, ffn):
    # ComfyUI/HF T5 files use SINGULAR `encoder.block.N.` (verified against
    # comfy/text_encoders/t5.py and real t5xxl files; ComfyUI detects T5 via
    # encoder.block.23.layer.1.DenseReluDense.wi_1.weight).
    pre = f"{prefix}encoder.block.{b}.layer."
    for proj in ("q", "k", "v"):
        sd[pre + f"0.SelfAttention.{proj}.weight"] = L16(hidden, hidden)
    sd[pre + "0.SelfAttention.o.weight"] = L16(hidden, hidden)
    sd[pre + "1.DenseReluDense.wi_0.weight"] = L16(ffn, hidden)
    sd[pre + "1.DenseReluDense.wi_1.weight"] = L16(ffn, hidden)
    sd[pre + "1.DenseReluDense.wo.weight"] = L16(hidden, ffn)
    sd[pre + "0.layer_norm.weight"] = torch.randn(hidden) * 0.1
    sd[pre + "1.layer_norm.weight"] = torch.randn(hidden) * 0.1


def _t5_decoder_block(sd, prefix, b, hidden, ffn):
    """T5 decoder block: kept to prove decoder.* stays passthrough."""
    pre = f"{prefix}decoder.block.{b}.layer."
    for proj in ("q", "k", "v"):
        sd[pre + f"0.SelfAttention.{proj}.weight"] = L16(hidden, hidden)
    sd[pre + "0.SelfAttention.o.weight"] = L16(hidden, hidden)
    for proj in ("q", "k", "v"):
        sd[pre + f"1.EncDecAttention.{proj}.weight"] = L16(hidden, hidden)
    sd[pre + "1.EncDecAttention.o.weight"] = L16(hidden, hidden)
    sd[pre + "2.DenseReluDense.wi_0.weight"] = L16(ffn, hidden)
    sd[pre + "2.DenseReluDense.wo.weight"] = L16(hidden, ffn)
    sd[pre + "0.layer_norm.weight"] = torch.randn(hidden) * 0.1
    sd[pre + "1.layer_norm.weight"] = torch.randn(hidden) * 0.1
    sd[pre + "2.layer_norm.weight"] = torch.randn(hidden) * 0.1


def _t5_te_blocks(sd, prefix="", blocks=1, hidden=4096, ffn=10240, vocab=2048,
                  with_decoder=True):
    for b in range(blocks):
        _t5_encoder_block(sd, prefix, b, hidden, ffn)
    if with_decoder:
        _t5_decoder_block(sd, prefix, 0, hidden, ffn)
    sd[prefix + "shared.weight"] = L16(vocab, hidden)
    sd[prefix + "encoder.final_layer_norm.weight"] = torch.randn(hidden) * 0.1
    sd[prefix + "encoder.relative_attention_bias.weight"] = torch.randn(32, 64) * 0.1


def _clip_text_block(sd, prefix, b, hidden, ffn):
    pre = f"{prefix}transformer.text_model.encoder.layers.{b}."
    for proj in ("q_proj", "k_proj", "v_proj"):
        sd[pre + f"self_attn.{proj}.weight"] = L(hidden, hidden)
    sd[pre + "self_attn.out_proj.weight"] = L(hidden, hidden)
    sd[pre + "mlp.fc1.weight"] = L(ffn, hidden)
    sd[pre + "mlp.fc2.weight"] = L(hidden, ffn)
    sd[pre + "layer_norm1.weight"] = torch.randn(hidden) * 0.1
    sd[pre + "layer_norm2.weight"] = torch.randn(hidden) * 0.1


def _clip_text_blocks(sd, prefix, blocks=1, hidden=768, ffn=3072, vocab=49408,
                      max_pos=77, with_projection=True):
    for b in range(blocks):
        _clip_text_block(sd, prefix, b, hidden, ffn)
    sd[prefix + "transformer.text_model.embeddings.token_embedding.weight"] = \
        L16(vocab, hidden)
    sd[prefix + "transformer.text_model.embeddings.position_embedding.weight"] = \
        L(max_pos, hidden)
    sd[prefix + "transformer.text_model.final_layer_norm.weight"] = torch.randn(hidden) * 0.1
    if with_projection:
        sd[prefix + "text_projection.weight"] = L(hidden, hidden)


def _clip_oc_text_block(sd, prefix, b, hidden, ffn):
    """open_clip-style block (CLIP-G in SDXL): transformer.resblocks.N.
    naming with mlp.c_fc/c_proj, verified against sd_xl_base_1.0 keys
    (conditioner.embedders.1.model.transformer.resblocks...)."""
    pre = f"{prefix}transformer.resblocks.{b}."
    for proj in ("q_proj", "k_proj", "v_proj"):
        sd[pre + f"attn.{proj}.weight"] = L(hidden, hidden)
    sd[pre + "attn.out_proj.weight"] = L(hidden, hidden)
    sd[pre + "mlp.c_fc.weight"] = L(ffn, hidden)
    sd[pre + "mlp.c_proj.weight"] = L(hidden, ffn)
    sd[pre + "ln_1.weight"] = torch.randn(hidden) * 0.1
    sd[pre + "ln_2.weight"] = torch.randn(hidden) * 0.1


def _clip_oc_text_blocks(sd, prefix, blocks=1, hidden=1280, ffn=5120,
                         with_projection=True):
    for b in range(blocks):
        _clip_oc_text_block(sd, prefix, b, hidden, ffn)
    sd[prefix + "transformer.final_layernorm.weight"] = torch.randn(hidden) * 0.1
    sd[prefix + "transformer.positional_embedding.weight"] = L(77, hidden)
    if with_projection:
        sd[prefix + "text_projection.weight"] = L(hidden, hidden)


def _llama_te_block(sd, prefix, b, hidden, ffn):
    pre = f"{prefix}model.layers.{b}."
    for proj in ("q_proj", "k_proj", "v_proj"):
        sd[pre + f"self_attn.{proj}.weight"] = L(hidden, hidden)
    sd[pre + "self_attn.o_proj.weight"] = L(hidden, hidden)
    sd[pre + "mlp.gate_proj.weight"] = L(ffn, hidden)
    sd[pre + "mlp.up_proj.weight"] = L(ffn, hidden)
    sd[pre + "mlp.down_proj.weight"] = L(hidden, ffn)
    sd[pre + "input_layernorm.weight"] = torch.randn(hidden) * 0.1
    sd[pre + "post_attention_layernorm.weight"] = torch.randn(hidden) * 0.1


def _llama_te_blocks(sd, prefix="", blocks=1, hidden=896, ffn=4864, vocab=8192,
                     with_lm_head=True):
    for b in range(blocks):
        _llama_te_block(sd, prefix, b, hidden, ffn)
    sd[prefix + "model.embed_tokens.weight"] = L16(vocab, hidden)
    sd[prefix + "model.norm.weight"] = torch.randn(hidden) * 0.1
    if with_lm_head:
        sd[prefix + "lm_head.weight"] = L16(vocab, hidden)


def make_t5xxl_te():
    """Standalone T5-XXL text-encoder fixture (real dims 4096/10240)."""
    sd = {}
    _t5_te_blocks(sd, "", blocks=1, hidden=4096, ffn=10240, vocab=2048)
    return sd


def make_clip_l_te():
    """Standalone CLIP-L text-encoder fixture (real dims 768/3072)."""
    sd = {}
    _clip_text_blocks(sd, "", blocks=1, hidden=768, ffn=3072, vocab=49408)
    return sd


def make_llama_qwen05b_te():
    """Standalone Qwen2-0.5B-style TE fixture (hidden 896, ffn 4864).
    K=896 is W4A8-ineligible (896 % 256 != 0), exercising the INT8/W4A4
    fallback path for TE weights."""
    sd = {}
    _llama_te_blocks(sd, "", blocks=1, hidden=896, ffn=4864, vocab=8192)
    return sd


def make_llama_gemma2b_te():
    """Standalone Gemma-2-2B-style TE fixture (hidden 2304, ffn 9216)."""
    sd = {}
    _llama_te_blocks(sd, "", blocks=1, hidden=2304, ffn=9216, vocab=8192)
    return sd


def make_flux():
    sd = {}; p = "model.diffusion_model."
    for b in range(2):
        sd[p+f"double_blocks.{b}.img_attn.qkv.weight"] = L(768, 768, 0.01)
        sd[p+f"double_blocks.{b}.img_attn.proj.weight"] = L(768, 768, 0.01)
        sd[p+f"double_blocks.{b}.txt_attn.qkv.weight"] = L(768, 1024, 0.01)
        sd[p+f"double_blocks.{b}.txt_attn.proj.weight"] = L(768, 768, 0.01)
        sd[p+f"double_blocks.{b}.img_mlp.w1.weight"] = L(3072, 768, 0.005)
        sd[p+f"double_blocks.{b}.img_mlp.w2.weight"] = L(768, 3072, 0.005)
        sd[p+f"double_blocks.{b}.img_attn.norm.key_norm.weight"] = torch.randn(768) * 0.1
        sd[p+f"single_blocks.{b}.linear1.weight"] = L(2304, 2304, 0.005)
        sd[p+f"single_blocks.{b}.linear2.weight"] = L(2304, 2304, 0.005)
    sd[p+"img_in.weight"] = L(768, 768, 0.01)
    sd[p+"txt_in.weight"] = L(768, 1024, 0.01)
    sd[p+"vector_in.in_layer.weight"] = L(768, 256, 0.05)
    sd[p+"guidance_in.in_layer.weight"] = L(768, 256, 0.05)
    sd[p+"final_layer.linear.weight"] = L(768, 768, 0.01)
    # text-encoder blocks (TE quantization test battery): one T5-XXL encoder
    # block + one decoder block under t5xxl. (decoder.* must stay passthrough)
    # and one CLIP-L block under clip_l. (clip_l.transformer.text_model...)
    _t5_te_blocks(sd, "t5xxl.", blocks=1, hidden=4096, ffn=10240, vocab=2048)
    _clip_text_blocks(sd, "clip_l.", blocks=1, hidden=768,
                      ffn=3072, vocab=49408)
    return sd

def make_wan():
    sd = {}; p = "model.diffusion_model."
    for b in range(2):
        for attn in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                sd[p+f"blocks.{b}.{attn}.{proj}.weight"] = L(768, 768)
            sd[p+f"blocks.{b}.{attn}.norm_q.weight"] = torch.randn(768) * 0.1
        # real WAN FFN dims: K=384 / K=2240 are not divisible by 256 and pass through
        sd[p+f"blocks.{b}.ffn.0.weight"] = L(2240, 384, 0.005)
        sd[p+f"blocks.{b}.ffn.2.weight"] = L(384, 2240, 0.005)
        sd[p+f"blocks.{b}.modulation"] = torch.randn(1, 6, 384) * 0.1
    sd[p+"patch_embedding.weight"] = L(384, 64, 0.02)
    sd[p+"text_embedding.weight"] = L(384, 1024, 0.01)
    sd[p+"time_embedding.0.weight"] = L(384, 384)
    sd[p+"head.modulation"] = torch.randn(1, 6, 384) * 0.1
    sd[p+"final_linear.weight"] = L(384, 384, 0.01)
    return sd

def make_minimax_h3():
    sd = {}
    for b in range(2):
        sd[f"blocks.{b}.attn.qkv_proj.weight"] = L(2304, 768, 0.005)
        sd[f"blocks.{b}.attn.out_proj.weight"] = L(768, 768, 0.01)
        sd[f"blocks.{b}.attn.q_norm.weight"] = torch.randn(128) * 0.1
        sd[f"blocks.{b}.attn.k_norm.weight"] = torch.randn(128) * 0.1
        # ffn_hidden_size = fc1.out // 2 = 1152; fc2 = Linear(ffn, hidden) -> K=1152.
        # K=1152 is not divisible by 256, so fc2 passes through at original precision.
        sd[f"blocks.{b}.mlp.fc1.weight"] = L(2304, 768, 0.005)
        sd[f"blocks.{b}.mlp.fc2.weight"] = L(768, 1152, 0.005)
        sd[f"blocks.{b}.adaln_proj.linear.weight"] = L(13824, 8, 0.05)
        sd[f"blocks.{b}.adaln_proj.linear.bias"] = torch.randn(13824) * 0.01
        sd[f"blocks.{b}.norm1.weight"] = torch.randn(768) * 0.1
    # latents_dim = 64//4 = 16 -> video_patch_dim = 16*16 = 256; audio_latents_dim = 256
    sd["video_patch_proj.weight"] = L(768, 256, 0.02)
    sd["audio_patch_proj.weight"] = L(768, 256, 0.02)
    sd["condition_proj.weight"] = L(768, 1024, 0.01)
    sd["final_layer.video_out.weight"] = L(64, 768, 0.01)
    sd["final_layer.audio_out.weight"] = L(256, 768, 0.01)
    sd["final_layer.norm.weight"] = torch.randn(768) * 0.1
    sd["adaln_t_table"] = torch.randn(1025, 8) * 0.1
    sd["rope.inv_freq"] = torch.randn(16) * 0.1
    sd["token_refiner.blocks.0.attn.qkv_proj.weight"] = L(2304, 768, 0.005)
    sd["token_refiner.blocks.0.attn.out_proj.weight"] = L(768, 768, 0.01)
    sd["token_refiner.blocks.0.mlp.fc1.weight"] = L(2304, 768, 0.005)
    sd["token_refiner.blocks.0.mlp.fc2.weight"] = L(768, 1152, 0.005)  # K%256!=0: passthrough
    return sd

def make_zimage():
    """Z-Image shaped fixture using the real tensor naming from
    comfy/ldm/lumina/model.py (attention.qkv/out, feed_forward.w1/w2/w3,
    context_refiner, noise_refiner, adaLN_modulation)."""
    sd = {}; p = "model.diffusion_model."
    for b in range(3):
        pre = p + f"layers.{b}."
        sd[pre + "attention.qkv.weight"] = L(1152, 768)
        sd[pre + "attention.out.weight"] = L(384, 768)
        sd[pre + "feed_forward.w1.weight"] = L(1024, 768, 0.01)
        sd[pre + "feed_forward.w2.weight"] = L(384, 1024, 0.01)
        sd[pre + "feed_forward.w3.weight"] = L(1024, 768, 0.01)
        sd[pre + "adaLN_modulation.0.weight"] = L(1536, 64)
        sd[pre + "attention_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "attention.q_norm.weight"] = torch.randn(128) * 0.1
    for b in range(1):
        pre = p + f"context_refiner.{b}."
        sd[pre + "attention.qkv.weight"] = L(1152, 768)
        sd[pre + "attention.out.weight"] = L(384, 768)
        sd[pre + "feed_forward.w1.weight"] = L(1024, 768, 0.01)
        sd[pre + "feed_forward.w2.weight"] = L(384, 1024, 0.01)
        sd[pre + "feed_forward.w3.weight"] = L(1024, 768, 0.01)
    sd[p + "cap_embedder.1.weight"] = L(384, 256)
    sd[p + "x_embedder.weight"] = L(384, 64)
    sd[p + "t_embedder.mlp.0.weight"] = L(256, 64)
    sd[p + "final_layer.adaLN_modulation.1.weight"] = L(384, 64)
    sd[p + "final_layer.linear.weight"] = L(16, 384, 0.01)  # K=384: K%256!=0, passthrough
    sd[p + "cap_pad_token"] = torch.randn(1, 384)
    return sd


def make_hydit():
    sd = {}; p = "model.diffusion_model."
    for b in range(2):
        sd[p+f"blocks.{b}.attn.qkv.weight"] = L(768, 768, 0.01)
        sd[p+f"blocks.{b}.attn.proj.weight"] = L(384, 768, 0.01)
        sd[p+f"blocks.{b}.mlp.fc1.weight"] = L(1536, 768, 0.005)
        sd[p+f"blocks.{b}.mlp.fc2.weight"] = L(384, 1536, 0.005)
    sd[p+"mlp_t5.0.weight"] = L(384, 1024, 0.01)
    sd[p+"x_embedder.proj.weight"] = L(384, 64, 0.02)
    sd[p+"extra_embedder.0.weight"] = L(384, 3968, 0.01)
    sd[p+"final_layer.linear.weight"] = L(64, 384, 0.01)
    return sd

def _og2_refiner(name):
    """One OmniGen2/Boogu refiner block: attn + feed_forward (real naming)."""
    sd = {}
    pre = f"{name}.0."
    for proj in ("to_q", "to_k", "to_v"):
        sd[pre + "attn." + proj + ".weight"] = L(768, 768)
    sd[pre + "attn.to_out.0.weight"] = L(384, 768)
    for i in (1, 2, 3):
        sd[pre + f"feed_forward.linear_{i}.weight"] = L(768, 768)
    sd[pre + "attn.norm_k.weight"] = torch.randn(384) * 0.1
    sd[pre + "attn.norm_q.weight"] = torch.randn(384) * 0.1
    sd[pre + "ffn_norm1.weight"] = torch.randn(384) * 0.1
    sd[pre + "ffn_norm2.weight"] = torch.randn(384) * 0.1
    sd[pre + "norm1.weight"] = torch.randn(384) * 0.1
    sd[pre + "norm2.weight"] = torch.randn(384) * 0.1
    return sd


def _og2_embedders():
    """Shared OmniGen2/Boogu embedders (real naming, prefix-less)."""
    sd = {}
    sd["x_embedder.weight"] = L(384, 64)
    sd["x_embedder.bias"] = torch.randn(384) * 0.01
    sd["ref_image_patch_embedder.weight"] = L(384, 64)
    sd["ref_image_patch_embedder.bias"] = torch.randn(384) * 0.01
    sd["image_index_embedding"] = torch.randn(5, 384) * 0.1
    sd["time_caption_embed.timestep_embedder.linear_1.weight"] = L(256, 64)
    sd["time_caption_embed.timestep_embedder.linear_1.bias"] = torch.randn(256) * 0.01
    sd["time_caption_embed.timestep_embedder.linear_2.weight"] = L(384, 256)
    sd["time_caption_embed.timestep_embedder.linear_2.bias"] = torch.randn(384) * 0.01
    sd["time_caption_embed.caption_embedder.0.weight"] = L(384, 1024)
    sd["time_caption_embed.caption_embedder.1.weight"] = L(384, 384)  # K%256!=0: passthrough
    sd["time_caption_embed.caption_embedder.1.bias"] = torch.randn(384) * 0.01
    sd["norm_out.linear_1.weight"] = L(384, 64)
    sd["norm_out.linear_1.bias"] = torch.randn(384) * 0.01
    sd["norm_out.linear_2.weight"] = L(64, 384)
    sd["norm_out.linear_2.bias"] = torch.randn(64) * 0.01
    return sd


def make_boogu():
    """Boogu-Image-0.1 shaped fixture using the real state-dict naming
    (Comfy-Org repack, prefix-less): double/single stream layers with
    img_self_attn / img_instruct_attn.processor / img_feed_forward /
    instruct_feed_forward and feed_forward.linear_N refiners."""
    sd = _og2_embedders()
    for b in range(2):
        pre = f"double_stream_layers.{b}."
        for proj in ("to_q", "to_k", "to_v"):
            sd[pre + "img_self_attn." + proj + ".weight"] = L(768, 768)
            sd[pre + "img_instruct_attn.processor.img_" + proj + ".weight"] = L(768, 768)
            sd[pre + "img_instruct_attn.processor.instruct_" + proj + ".weight"] = L(768, 768)
        sd[pre + "img_self_attn.to_out.0.weight"] = L(768, 768)
        sd[pre + "img_instruct_attn.to_out.0.weight"] = L(768, 768)
        sd[pre + "img_instruct_attn.processor.img_out.weight"] = L(384, 768)
        sd[pre + "img_instruct_attn.processor.instruct_out.weight"] = L(384, 768)
        for ffn in ("img_feed_forward", "instruct_feed_forward"):
            for i in (1, 2, 3):
                sd[pre + f"{ffn}.linear_{i}.weight"] = L(768, 768)
        for mod in ("img_norm1", "img_norm2", "img_norm3", "instruct_norm1", "instruct_norm2"):
            sd[pre + f"{mod}.linear.weight"] = L(1536, 64)
            sd[pre + f"{mod}.linear.bias"] = torch.randn(1536) * 0.01
            sd[pre + f"{mod}.norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_self_attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_self_attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_instruct_attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_instruct_attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_self_attn_norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "instruct_attn_norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_attn_norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_ffn_norm2.weight"] = torch.randn(384) * 0.1
        sd[pre + "instruct_ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "instruct_ffn_norm2.weight"] = torch.randn(384) * 0.1
    for b in range(2):
        pre = f"single_stream_layers.{b}."
        for proj in ("to_q", "to_k", "to_v"):
            sd[pre + "attn." + proj + ".weight"] = L(768, 768)
        sd[pre + "attn.to_out.0.weight"] = L(384, 768)
        for i in (1, 2, 3):
            sd[pre + f"feed_forward.linear_{i}.weight"] = L(768, 768)
        sd[pre + "norm1.linear.weight"] = L(1536, 64)
        sd[pre + "norm1.linear.bias"] = torch.randn(1536) * 0.01
        sd[pre + "norm1.norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "norm2.weight"] = torch.randn(384) * 0.1
        sd[pre + "attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm2.weight"] = torch.randn(384) * 0.1
    for name in ("context_refiner", "noise_refiner", "ref_image_refiner"):
        sd.update(_og2_refiner(name))
    return sd


def make_boogu_real():
    """Boogu-Image-0.1-Turbo shaped fixture with the REAL structural widths
    (hidden 3360, FFN expansion 13568) and the full state-dict inventory the
    ComfyUI Boogu loader needs (embedders, norms, modulations). Synthetic data,
    real dimensions (verified against Boogu/Boogu-Image-0.1-Turbo). The
    conversion proves mixed passthrough + W4A8 behavior and the file loads in
    real ComfyUI for the one-step smoke test."""
    sd = {}
    def L(n, k, scale=0.02): return torch.randn(n, k) * scale
    # embedders (real naming, prefix-less)
    sd["x_embedder.weight"] = L(3360, 64)
    sd["x_embedder.bias"] = torch.randn(3360) * 0.01
    sd["ref_image_patch_embedder.weight"] = L(3360, 64)
    sd["ref_image_patch_embedder.bias"] = torch.randn(3360) * 0.01
    sd["image_index_embedding"] = torch.randn(5, 3360) * 0.1
    sd["time_caption_embed.timestep_embedder.linear_1.weight"] = L(256, 64)
    sd["time_caption_embed.timestep_embedder.linear_1.bias"] = torch.randn(256) * 0.01
    sd["time_caption_embed.timestep_embedder.linear_2.weight"] = L(3360, 256)
    sd["time_caption_embed.timestep_embedder.linear_2.bias"] = torch.randn(3360) * 0.01
    sd["time_caption_embed.caption_embedder.0.weight"] = L(4096, 1024)
    sd["time_caption_embed.caption_embedder.1.weight"] = L(3360, 4096)
    sd["time_caption_embed.caption_embedder.1.bias"] = torch.randn(3360) * 0.01
    sd["norm_out.linear_1.weight"] = L(3360, 3360)
    sd["norm_out.linear_1.bias"] = torch.randn(3360) * 0.01
    sd["norm_out.linear_2.weight"] = L(64, 3360)
    sd["norm_out.linear_2.bias"] = torch.randn(64) * 0.01
    for b in range(4):
        pre = f"double_stream_layers.{b}."
        for proj in ("to_q", "to_k", "to_v"):
            sd[pre + "img_self_attn." + proj + ".weight"] = L(384, 3360)
            sd[pre + "img_instruct_attn.processor.img_" + proj + ".weight"] = L(384, 3360)
            sd[pre + "img_instruct_attn.processor.instruct_" + proj + ".weight"] = L(384, 3360)
        sd[pre + "img_self_attn.to_out.0.weight"] = L(384, 3360)
        sd[pre + "img_instruct_attn.to_out.0.weight"] = L(384, 3360)
        sd[pre + "img_instruct_attn.processor.img_out.weight"] = L(384, 3360)
        sd[pre + "img_instruct_attn.processor.instruct_out.weight"] = L(384, 3360)
        for ffn in ("img_feed_forward", "instruct_feed_forward"):
            for i in (1, 2, 3):
                sd[pre + f"{ffn}.linear_{i}.weight"] = L(384, 3360 if i != 2 else 13568)
        for mod in ("img_norm1", "img_norm2", "img_norm3", "instruct_norm1", "instruct_norm2"):
            sd[pre + f"{mod}.linear.weight"] = L(1536, 64)
            sd[pre + f"{mod}.linear.bias"] = torch.randn(1536) * 0.01
            sd[pre + f"{mod}.norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_self_attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_self_attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_instruct_attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_instruct_attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_self_attn_norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "instruct_attn_norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_attn_norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "img_ffn_norm2.weight"] = torch.randn(384) * 0.1
        sd[pre + "instruct_ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "instruct_ffn_norm2.weight"] = torch.randn(384) * 0.1
    for b in range(8):
        pre = f"single_stream_layers.{b}."
        for proj in ("to_q", "to_k", "to_v"):
            sd[pre + "attn." + proj + ".weight"] = L(384, 3360)
        sd[pre + "attn.to_out.0.weight"] = L(384, 3360)
        for i in (1, 2, 3):
            sd[pre + f"feed_forward.linear_{i}.weight"] = L(384, 3360 if i != 2 else 13568)
        sd[pre + "norm1.linear.weight"] = L(1536, 64)
        sd[pre + "norm1.linear.bias"] = torch.randn(1536) * 0.01
        sd[pre + "norm1.norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "norm2.weight"] = torch.randn(384) * 0.1
        sd[pre + "attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm2.weight"] = torch.randn(384) * 0.1
    for name in ("context_refiner", "noise_refiner", "ref_image_refiner"):
        pre = f"{name}.0."
        for proj in ("to_q", "to_k", "to_v"):
            sd[pre + "attn." + proj + ".weight"] = L(384, 3360)
        sd[pre + "attn.to_out.0.weight"] = L(384, 3360)
        for i in (1, 2, 3):
            sd[pre + f"feed_forward.linear_{i}.weight"] = L(384, 3360 if i != 2 else 13568)
        sd[pre + "attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm2.weight"] = torch.randn(384) * 0.1
        sd[pre + "norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "norm2.weight"] = torch.randn(384) * 0.1
    return sd


def make_omnigen2():
    """OmniGen2 shaped fixture using the real state-dict naming (BAAI/OmniGen2,
    prefix-less): layers.N.attn / feed_forward plus the refiners."""
    sd = _og2_embedders()
    for b in range(2):
        pre = f"layers.{b}."
        for proj in ("to_q", "to_k", "to_v"):
            sd[pre + "attn." + proj + ".weight"] = L(768, 768)
        sd[pre + "attn.to_out.0.weight"] = L(384, 768)
        for i in (1, 2, 3):
            sd[pre + f"feed_forward.linear_{i}.weight"] = L(768, 768)
        sd[pre + "norm1.linear.weight"] = L(1536, 64)
        sd[pre + "norm1.linear.bias"] = torch.randn(1536) * 0.01
        sd[pre + "norm1.norm.weight"] = torch.randn(384) * 0.1
        sd[pre + "norm2.weight"] = torch.randn(384) * 0.1
        sd[pre + "attn.norm_k.weight"] = torch.randn(384) * 0.1
        sd[pre + "attn.norm_q.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm1.weight"] = torch.randn(384) * 0.1
        sd[pre + "ffn_norm2.weight"] = torch.randn(384) * 0.1
    for name in ("context_refiner", "noise_refiner", "ref_image_refiner"):
        sd.update(_og2_refiner(name))
    return sd


def make_mmdit_sd3():
    sd = {}; p = "model.diffusion_model."
    for b in range(2):
        sd[p+f"joint_blocks.{b}.x_block.attn.qkv.weight"] = L(1152, 768, 0.005)
        sd[p+f"joint_blocks.{b}.x_block.attn.proj.weight"] = L(384, 768, 0.01)
        sd[p+f"joint_blocks.{b}.context_block.attn.qkv.weight"] = L(1152, 1024, 0.005)
        sd[p+f"joint_blocks.{b}.context_block.attn.proj.weight"] = L(384, 768, 0.01)
        sd[p+f"joint_blocks.{b}.x_block.mlp.fc1.weight"] = L(1536, 768, 0.005)
        sd[p+f"joint_blocks.{b}.x_block.mlp.fc2.weight"] = L(384, 1536, 0.005)
    sd[p+"x_embedder.proj.weight"] = L(384, 64, 0.02)
    sd[p+"y_embedder.mlp.0.weight"] = L(384, 2816, 0.01)
    sd[p+"final_layer.linear.weight"] = L(64, 384, 0.01)
    return sd

makers = {
    "sdxl": make_sdxl, "sd15": make_sd15, "flux": make_flux, "wan": make_wan,
    "minimax_h3": make_minimax_h3, "hydit": make_hydit, "mmdit_sd3": make_mmdit_sd3,
    "zimage": make_zimage, "boogu": make_boogu, "boogu_real": make_boogu_real,
    "omnigen2": make_omnigen2,
    "t5xxl": make_t5xxl_te, "clip_l": make_clip_l_te,
    "llama_qwen05b": make_llama_qwen05b_te, "llama_gemma2b": make_llama_gemma2b_te,
}
key = os.path.basename(OUT).split("_fixture")[0]
sd = makers[key]()
safetensors.torch.save_file(sd, OUT, metadata={"fixture": key})
print(f"wrote {OUT}: {len(sd)} tensors, {os.path.getsize(OUT)//1024} KiB")
