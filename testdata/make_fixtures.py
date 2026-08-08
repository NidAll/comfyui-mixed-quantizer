
import torch, safetensors.torch, os, sys
OUT = sys.argv[1]
torch.manual_seed(42)
def L(n, k, scale=0.02): return torch.randn(n, k) * scale

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
}
key = os.path.basename(OUT).split("_fixture")[0]
sd = makers[key]()
safetensors.torch.save_file(sd, OUT, metadata={"fixture": key})
print(f"wrote {OUT}: {len(sd)} tensors, {os.path.getsize(OUT)//1024} KiB")
