
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
                sd[p+f"blocks.{b}.{attn}.{proj}.weight"] = L(384, 384)
            sd[p+f"blocks.{b}.{attn}.norm_q.weight"] = torch.randn(384) * 0.1
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
        sd[f"blocks.{b}.mlp.fc1.weight"] = L(2304, 768, 0.005)
        sd[f"blocks.{b}.mlp.fc2.weight"] = L(768, 2304, 0.005)
        sd[f"blocks.{b}.adaln_proj.linear.weight"] = L(2304, 8, 0.05)
        sd[f"blocks.{b}.adaln_proj.linear.bias"] = torch.randn(2304) * 0.01
        sd[f"blocks.{b}.norm1.weight"] = torch.randn(768) * 0.1
    sd["video_patch_proj.weight"] = L(768, 64, 0.02)
    sd["audio_patch_proj.weight"] = L(768, 64, 0.02)
    sd["condition_proj.weight"] = L(768, 1024, 0.01)
    sd["final_layer.video_out.weight"] = L(64, 768, 0.01)
    sd["final_layer.audio_out.weight"] = L(256, 768, 0.01)
    sd["final_layer.norm.weight"] = torch.randn(768) * 0.1
    sd["adaln_t_table"] = torch.randn(1025, 8) * 0.1
    sd["rope.inv_freq"] = torch.randn(16) * 0.1
    sd["token_refiner.blocks.0.attn.qkv_proj.weight"] = L(2304, 768, 0.005)
    sd["token_refiner.blocks.0.mlp.fc1.weight"] = L(2304, 768, 0.005)
    return sd

def make_hydit():
    sd = {}; p = "model.diffusion_model."
    for b in range(2):
        sd[p+f"blocks.{b}.attn.qkv.weight"] = L(384, 384, 0.01)
        sd[p+f"blocks.{b}.attn.proj.weight"] = L(384, 384, 0.01)
        sd[p+f"blocks.{b}.mlp.fc1.weight"] = L(1536, 384, 0.005)
        sd[p+f"blocks.{b}.mlp.fc2.weight"] = L(384, 1536, 0.005)
    sd[p+"mlp_t5.0.weight"] = L(384, 1024, 0.01)
    sd[p+"x_embedder.proj.weight"] = L(384, 64, 0.02)
    sd[p+"extra_embedder.0.weight"] = L(384, 3968, 0.01)
    sd[p+"final_layer.linear.weight"] = L(64, 384, 0.01)
    return sd

def make_mmdit_sd3():
    sd = {}; p = "model.diffusion_model."
    for b in range(2):
        sd[p+f"joint_blocks.{b}.x_block.attn.qkv.weight"] = L(1152, 384, 0.005)
        sd[p+f"joint_blocks.{b}.x_block.attn.proj.weight"] = L(384, 384, 0.01)
        sd[p+f"joint_blocks.{b}.context_block.attn.qkv.weight"] = L(1152, 1024, 0.005)
        sd[p+f"joint_blocks.{b}.context_block.attn.proj.weight"] = L(384, 384, 0.01)
        sd[p+f"joint_blocks.{b}.x_block.mlp.fc1.weight"] = L(1536, 384, 0.005)
        sd[p+f"joint_blocks.{b}.x_block.mlp.fc2.weight"] = L(384, 1536, 0.005)
    sd[p+"x_embedder.proj.weight"] = L(384, 64, 0.02)
    sd[p+"y_embedder.mlp.0.weight"] = L(384, 2816, 0.01)
    sd[p+"final_layer.linear.weight"] = L(64, 384, 0.01)
    return sd

makers = {
    "sdxl": make_sdxl, "sd15": make_sd15, "flux": make_flux, "wan": make_wan,
    "minimax_h3": make_minimax_h3, "hydit": make_hydit, "mmdit_sd3": make_mmdit_sd3,
}
key = os.path.basename(OUT).split("_fixture")[0]
sd = makers[key]()
safetensors.torch.save_file(sd, OUT, metadata={"fixture": key})
print(f"wrote {OUT}: {len(sd)} tensors, {os.path.getsize(OUT)//1024} KiB")
