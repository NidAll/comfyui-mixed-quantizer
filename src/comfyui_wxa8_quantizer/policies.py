"""Architecture family registry and checkpoint detection."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import dataclasses
import re
from comfyui_wxa8_quantizer.errors import UnknownArchitectureError
from comfyui_wxa8_quantizer.io import CheckpointInfo
from comfyui_wxa8_quantizer.utils import flatten_regex
@dataclass(frozen=True)
class FamilyPolicy:
    family: str
    comfyui_classes: Tuple[str, ...]
    detect_primary: Tuple[str, ...]
    detect_hints: Tuple[str, ...] = ()
    quantize: Tuple[str, ...] = ()
    keep: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    group_size: int = 16
    min_weight_numel: int = 4096
    max_rel_l2: float = 0.25
    min_cosine: float = 0.95
    runtime_status: str = "experimental"
    notes: str = ""

    def quantize_re(self) -> re.Pattern:
        return flatten_regex(self.quantize)

    def keep_re(self) -> re.Pattern:
        return flatten_regex(self.keep)

    def exclude_re(self) -> re.Pattern:
        return flatten_regex(self.exclude)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

UNIVERSAL_EXCLUDE = (
    r"(^|\.)(norm|norm1|norm2|norm3|ln\w*|layer_norm|rms_norm|final_norm|final_layer_norm|"
    r"q_norm|k_norm|query_norm|key_norm|pre_norm|post_norm|adaln_norm|norm_added_q|"
    r"prenorm|input_layernorm|self_attn_norm|post_attention_layernorm|"
    r"ln_final|ln1|ln2|ln_pre|ln_post|emb_norm|token_norm|txt_norm|ffn_norm1|ffn_norm2)"
    r"\.(weight|bias|scale|shift)$",
    r"(^|\.)(pos_embed|pos_embedding|positional_encoding|position_ids|"
    r"emb_pos|pos_emb|rotary_pos_emb|rope|inv_freq|freqs_cis|emb_tokens|"
    r"embed_positions|patch_embedding|patch_embed|adaln_t_table|cap_pad_token|"
    r"__x0__|__sequential__|memory_tokens|timestep_features)(\.|$)",
    r"(^|\.)(time_embed|t_embedder|timestep_embedder|time_embedder|time_embeddings|"
    r"ofs_embedding|fps_embedding|style_embedding|cond_embedding|guidance_in|"
    r"vector_in|txt_in|img_in|input_embedder|x_embedder|y_embedder|context_embedder|"
    r"patch_embedder|patchify_proj|video_patch_proj|audio_patch_proj|condition_proj|"
    r"cond_proj|cond_embed|input_proj|img_emb|text_proj|caption_proj|t5_yproj|"
    r"final_layer|output_layer|head|out_layer|final_linear|"
    r"audio_out|video_out|linear_fc2|to_gate_logits|genre_embedder|speaker_embedder|"
    r"lyric_proj|text_embedding|ref_image_patch_embedder|ofs_embedding_linear_1|"
    r"ofs_embedding_linear_2|time_embedding_linear_1|time_embedding_linear_2|"
    r"adaln_proj|adaln_modulation|adaLN_modulation|adaln_single|time_caption_embed|extra_embedder|"
    r"text_embedder|label_emb|clip_txt_mapper|clip_img_mapper|clip_mapper|"
    r"clip_txt_pooled_mapper|cond_type_embedding|distilled_guidance_layer|"
    r"control_adapter|ref_conv|latent_in|cond_in|cam_out_layer|repo_layers|"
    r"content_map|gate_map|final_map|input_layer|cam_enc|cam_dec|"
    r"visual_embeddings|time_embeddings|ofs_embedding_linear|patch_embedding_mask|"
    r"patch_embedding_pose|patch_embedding_global|emb|embed|embedding|"
    r"condition_embedder|adaln_curve|llm_cond_proj|input_embedder|pos_embed_proj|"
    r"visual_transformer_blocks|text_transformer_blocks|"
    r"encoder|decoder|lyric_encoder|ssl_|vocoder|first_stage|cond_stage)(\.|$)",
)

UNET_ATTN_Q = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight",
)

UNET_ATTN_K = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight",
)

UNET_FF = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
)

UNET_TIME_EMBED = (r"(^|\.)time_embed\.(0|2)\.weight$",)

UNET_LABEL_EMB = (r"(^|\.)label_emb\.0\.0\.weight$",)

def _sd_unet_policy(family: str, classes: Tuple[str, ...], notes: str = "") -> FamilyPolicy:
    return FamilyPolicy(
        family=family, comfyui_classes=classes,
        detect_primary=(),
        detect_hints=(),
        quantize=UNET_ATTN_Q + UNET_FF,
        keep=UNET_TIME_EMBED + UNET_LABEL_EMB + (
            r"(^|\.)(out\.\d+|output_blocks\.\d+\.\d+\.conv|input_blocks\.\d+\.\d+\.conv)\.weight$",
        ),
        exclude=UNIVERSAL_EXCLUDE,
        runtime_status="experimental",
        notes=notes,
    )

REGISTRY_ORDER: List[str] = []

REGISTRY: Dict[str, FamilyPolicy] = {}

def _register(policy: FamilyPolicy) -> None:
    REGISTRY[policy.family] = policy
    REGISTRY_ORDER.append(policy.family)

_register(FamilyPolicy(
    family="mmdit_sd3",
    comfyui_classes=("SD3",),
    detect_primary=("joint_blocks.0.context_block.attn.qkv.weight",),
    detect_hints=("x_embedder.proj.weight", "final_layer.linear.weight", "y_embedder.mlp.0.weight"),
    quantize=(
        r"joint_blocks\.\d+\.(x_block|context_block)\.attn\.(qkv|proj)\.weight$",
        r"joint_blocks\.\d+\.(x_block|context_block)\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|y_embedder|context_embedder|final_layer|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="SD3 / SD3.5 MMDiT joint-block family (also covers sd3.5 medium/large).",
))

_register(FamilyPolicy(
    family="stable_cascade",
    comfyui_classes=("Stable_Cascade_C", "Stable_Cascade_B"),
    detect_primary=("clf.1.weight",),
    detect_hints=("clip_txt_mapper.weight", "clip_mapper.weight", "clip_img_mapper.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"blocks\.\d+\.ff\.(0|2)\.(proj)?\.?weight$",
        r"(^|\.)mapper\.weight$",
    ),
    keep=(r"(^|\.)(clip_txt_mapper|clip_img_mapper|clip_mapper|clip_txt_pooled_mapper)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Würstchen stage B/C DiT.",
))

_register(FamilyPolicy(
    family="stable_audio",
    comfyui_classes=("StableAudio", "StableAudio3"),
    detect_primary=("transformer.rotary_pos_emb.inv_freq",),
    detect_hints=("to_global_embed.0.weight", "to_cond_embed.0.weight", "to_timestep_embed.0.weight"),
    quantize=(
        r"transformer\.layers\.\d+\.self_attn\.(to_qkv|to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer\.layers\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"transformer\.layers\.\d+\.(to_local_embed|to_global_embed|to_cond_embed|to_timestep_embed)\.\d+\.weight$",
    ),
    keep=(r"(^|\.)(to_global_embed|to_cond_embed|to_timestep_embed|postprocess_conv|project_in|project_out|transformer\.project_in|transformer\.project_out|global_cond_embedder)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Stable Audio 1/3 DiT (audio).",
))

_register(FamilyPolicy(
    family="aura_flow",
    comfyui_classes=("AuraFlow",),
    detect_primary=("double_layers.0.attn.w1q.weight",),
    detect_hints=("single_layers.0.attn.w1q.weight", "cond_seq_linear.weight", "positional_encoding"),
    quantize=(
        r"(double_layers|single_layers)\.\d+\.attn\.(w1q|w1k|w1v|w2|o_proj|w2q|w2k|w2v|w1o|w2o)\.weight$",
        r"(double_layers|single_layers)\.\d+\.mlp\.(c_fc1|c_fc2|c_proj)\.weight$",
    ),
    keep=(r"(^|\.)(cond_seq_linear|init_x_linear|final_linear|positional_encoding)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="hydit",
    comfyui_classes=("HunyuanDiT", "HunyuanDiT1"),
    detect_primary=("mlp_t5.0.weight",),
    detect_hints=("blocks.0.attn.qkv.weight", "x_embedder.proj.weight", "extra_embedder.0.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|extra_embedder|text_embedder|final_layer|pos_embed|mlp_t5)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="hunyuan_video",
    comfyui_classes=("HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V",
                     "HunyuanImage21", "HunyuanImage21Refiner", "HunyuanVideo15",
                     "HunyuanVideo15_SR_Distilled"),
    detect_primary=("txt_in.individual_token_refiner.blocks.0.norm1.weight",),
    detect_hints=("img_in.proj.weight", "final_layer.linear.weight",
                  "double_blocks.0.attn.qkv.weight", "txt_in.input_embedder.weight"),
    quantize=(
        r"double_blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"single_blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
        r"single_blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|fps_embedding|style_embedding|"
          r"cond_embedding|txt_std|txt_emb|final_layer|individual_token_refiner|"
          r"byt5_in|time_r_in|vision_in|cond_type_embedding|time_embed|extra_embedder|"
          r"audio_embed|adaln_)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="HunyuanVideo 1.x + HunyuanImage 2.1 families share this DiT structure.",
))

_register(FamilyPolicy(
    family="flux",
    comfyui_classes=("Flux", "FluxInpaint", "FluxSchnell", "LongCatImage"),
    detect_primary=("double_blocks.0.img_attn.norm.key_norm.weight",
                    "double_blocks.0.img_attn.norm.key_norm.scale"),
    detect_hints=("img_in.weight", "txt_in.weight", "single_blocks.0.linear1.weight",
                  "guidance_in.in_layer.weight", "vector_in.in_layer.weight"),
    quantize=(
        r"double_blocks\.\d+\.img_attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.txt_attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.img_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"double_blocks\.\d+\.txt_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"single_blocks\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|time_text_embed|txt_embed|"
          r"final_layer|distilled_guidance_layer|txt_norm)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="verified",
    notes="Flux family; runtime verified via the reference W4A8 pipeline (same "
          "double_blocks/single_blocks structure as MiniMax H3).",
))

_register(FamilyPolicy(
    family="flux2",
    comfyui_classes=("Flux2",),
    detect_primary=("double_stream_modulation_img.lin.weight",),
    detect_hints=("double_stream_layers.0.img_attn.qkv.weight", "img_in.weight"),
    quantize=(
        r"double_stream_layers\.\d+\.img_attn\.(qkv|proj)\.weight$",
        r"double_stream_layers\.\d+\.txt_attn\.(qkv|proj)\.weight$",
        r"double_stream_layers\.\d+\.img_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"double_stream_layers\.\d+\.txt_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"single_stream_layers\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="chroma",
    comfyui_classes=("Chroma", "ChromaRadiance"),
    detect_primary=("distilled_guidance_layer.norms.0.weight",),
    detect_hints=("distilled_guidance_layer.0.norms.0.weight", "nerf_blocks.0.norm.weight"),
    quantize=(
        r"(distilled_guidance_layer|double_blocks|single_blocks|nerf_blocks)\.\d+\.\w+\.(qkv|proj|w1|w2|linear1|linear2)\.weight$",
        r"(distilled_guidance_layer|double_blocks|single_blocks|nerf_blocks)\.\d+\.(img_attn|txt_attn|attn)\.(qkv|proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|final_layer|nerf_final_layer|nerf_embedder|img_in_patch)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="mochi",
    comfyui_classes=("GenmoMochi",),
    detect_primary=("t5_yproj.weight",),
    detect_hints=("time_blocks.0.attn.qkv_x.weight", "patch_embed.proj.weight"),
    quantize=(
        r"(time_blocks|t5_blocks)\.\d+\.attn\.(qkv_x|qkv_y|proj_x|proj_y)\.weight$",
        r"(time_blocks|t5_blocks)\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(patch_embed|t5_yproj|final_layer|cond_embedder|timestep_embedder|pos_embed|mod)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="minimax_h3",
    comfyui_classes=("MiniMaxH3",),
    detect_primary=("video_patch_proj.weight", "audio_patch_proj.weight"),
    detect_hints=("blocks.0.attn.qkv_proj.weight", "final_layer.video_out.weight",
                  "adaln_t_table", "rope.inv_freq"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv_proj|out_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(video_patch_proj|audio_patch_proj|condition_proj|final_layer|"
          r"adaln_proj|adaln_t_table|token_refiner|time_embedder|rope)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="verified",
    notes="Reference family: the Kijai W4A8 test model quantizes exactly "
          "attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2 per block.",
))

_register(FamilyPolicy(
    family="ltxv",
    comfyui_classes=("LTXV", "LTXAV"),
    detect_primary=("adaln_single.emb.timestep_embedder.linear_1.bias",),
    detect_hints=("transformer_blocks.0.attn2.to_k.weight", "audio_adaln_single.linear.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.(0|2)\.weight$",
    ),
    keep=(r"(^|\.)(patchify_proj|time_embed|cond_proj|caption_proj|proj_out|adaln_single|"
          r"audio_adaln_single|pos_embed|rope)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="ace_step",
    comfyui_classes=("ACEStep", "ACEStep15"),
    detect_primary=("genre_embedder.weight",),
    detect_hints=("encoder.lyric_encoder.layers.0.input_layernorm.weight",
                  "decoder.layers.0.self_attn.q_proj.weight"),
    quantize=(
        r"(encoder|decoder)\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"(encoder|decoder)\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(genre_embedder|speaker_embedder|lyric_proj|ssl_|enc|dec)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="ACE-Step music diffusion (audio).",
))

_register(FamilyPolicy(
    family="pixart",
    comfyui_classes=("PixArtAlpha", "PixArtSigma"),
    detect_primary=("t_block.1.weight",),
    detect_hints=("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                  "y_embedder.y_embedding", "ar_embedder.mlp.0.weight"),
    quantize=(
        r"blocks\.\d+\.attn[12]\.(qkv|to_q|to_k|to_v|proj|to_out\.0)\.weight$",
        r"blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|y_embedder|t_block|pos_embed|final_layer|csize_embedder|"
          r"ar_embedder|pe_interpolation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="cosmos",
    comfyui_classes=("CosmosT2V", "CosmosI2V"),
    detect_primary=("blocks.block0.blocks.0.block.attn.to_q.0.weight",),
    detect_hints=("x_embedder.proj.1.weight", "adaln_lora"),
    quantize=(
        r"blocks\.block\d+\.blocks\.\d+\.block\.attn\.(to_q|to_k|to_v|to_out)\.\d+\.weight$",
        r"blocks\.block\d+\.blocks\.\d+\.block\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        r"blocks\.block\d+\.blocks\.\d+\.block\.cross_attn\.(to_q|to_k|to_v|to_out)\.\d+\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|adaln|pos_emb|final_layer|patch_embed|cond_embed|"
          r"cross_attn_norm|norm)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="cosmos_predict2",
    comfyui_classes=("CosmosT2IPredict2", "CosmosI2VPredict2"),
    detect_primary=("blocks.0.mlp.layer1.weight",),
    detect_hints=("x_embedder.proj.1.weight",),
    quantize=(
        r"blocks\.\d+\.attn\.(q_proj|k_proj|v_proj|output_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(layer1|layer2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|adaln|pos_emb|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="anima",
    comfyui_classes=("Anima",),
    detect_primary=("__x0__",),
    detect_hints=("layers.0.attn.q_proj.weight",),
    quantize=(
        r"layers\.\d+\.attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"layers\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|cond_embed|final_layer|modulation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="lumina2",
    comfyui_classes=("Lumina2", "ZImage", "ZImagePixelSpace"),
    detect_primary=("cap_embedder.1.weight",),
    detect_hints=("noise_refiner.0.attention.k_norm.weight", "layers.0.attn.qkv.weight",
                  "layers.0.attention.qkv.weight", "dec_net.cond_embed.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(qkv|o_proj|proj)\.weight$",
        r"layers\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        # real Lumina2 / Z-Image naming (comfy/ldm/lumina/model.py):
        # JointTransformerBlock.attention.{qkv,out}, FeedForward.{w1,w2,w3}
        r"(layers|context_refiner|noise_refiner)\.\d+\.attention\.(qkv|out)\.weight$",
        r"(layers|context_refiner|noise_refiner)\.\d+\.feed_forward\.(w1|w2|w3)\.weight$",
    ),
    keep=(r"(^|\.)(cap_embedder|clip_text_pooled_proj|siglip_embedder|x_embedder|t_embedder|"
          r"cond_embed|final_layer|dec_net|pos_embed|adaLN_modulation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="pixeldit",
    comfyui_classes=("PixelDiTT2I", "PiD"),
    detect_primary=("core.pixel_embedder.proj.weight", "lq_proj.latent_proj.0.weight"),
    detect_hints=("cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight",
                  "x_embedder.proj.weight"),
    quantize=(
        r"core\.(blocks|transformer_blocks)\.\d+\.(attn|attention)\.(qkv|to_q|to_k|to_v|proj|to_out\.0)\.weight$",
        r"core\.(blocks|transformer_blocks)\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(core\.pixel_embedder|cap_embedder|noise_refiner|x_embedder|t_embedder|"
          r"final_layer|pos_embed|lq_proj)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="wan",
    comfyui_classes=("WAN21_T2V", "WAN21_CausalAR_T2V", "WAN21_I2V", "WAN21_FunControl2V",
                     "WAN21_Camera", "WAN22_Camera", "WAN21_Vace", "WAN21_HuMo",
                     "WAN22_S2V", "WAN22_Animate", "WAN22_T2V", "WAN_Animate2", "WAN21_FlowRVS",
                     "WAN21_SCAIL", "WAN21_SCAIL2", "WAN22_WanDancer"),
    detect_primary=("head.modulation",),
    detect_hints=("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight",
                  "patch_embedding.weight", "txt_embedding.weight"),
    quantize=(
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)\.weight$",
        r"blocks\.\d+\.feed_forward\.(w1|w2)\.weight$",
        r"blocks\.\d+\.ffn\.(0|2)\.weight$",
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q_img|k_img|v_img|o_img)\.weight$",
    ),
    keep=(r"(^|\.)(patch_embedding|text_embedding|time_embedding|time_projection|"
          r"final_linear|head|before_proj|after_proj|audio_proj|cond_embedding|"
          r"vace_patch_embedding|control_adapter|img_emb|face_adapter|latent_in|"
          r"patch_embedding_mask|patch_embedding_pose|patch_embedding_global)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="hunyuan3d",
    comfyui_classes=("Hunyuan3Dv2", "Hunyuan3Dv2_1", "Hunyuan3Dv2mini"),
    detect_primary=("latent_in.weight",),
    detect_hints=("x_embedder.weight", "cond_in.weight", "blocks.0.attn.to_q.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0|qkv|proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(latent_in|cond_in|x_embedder|t_embedder|final_layer|pos_embed|mod)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="triposplat",
    comfyui_classes=("TripoSplat",),
    detect_primary=("cam_out_layer.weight",),
    detect_hints=("repo_layers.0.final_map.weight", "cond_embedder.weight"),
    quantize=(
        r"(cam_enc|cam_dec|repo_layers)\.\d+\.(qkv|to_q|to_k|to_v|proj|fc1|fc2|w1|w2|final_map|content_map|gate_map)\.weight$",
    ),
    keep=(r"(^|\.)(cond_embedder|input_layer|out_layer|cam_out_layer|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="hidream",
    comfyui_classes=("HiDream", "HiDreamO1"),
    detect_primary=("t_embedder1.mlp.0.weight",),
    detect_hints=("x_embedder.proj1.weight", "caption_projection.0.linear.weight"),
    quantize=(
        r"(visual|text)_transformer_blocks\.\d+\.(attn|attention)\.(qkv|to_q|to_k|to_v|proj|to_out\.0)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        r"double_stream_blocks\.\d+\.(attn|mlp)\.\w+\.weight$",
        r"single_stream_blocks\.\d+\.(attn|mlp)\.\w+\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|caption_projection|time_embed|pos_embed|final_layer|cond_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="seedvr2",
    comfyui_classes=("SeedVR2",),
    detect_primary=("cap_embedder.1.weight",),
    detect_hints=("noise_refiner.0.attention.k_norm.weight",
                  "x_embedder.proj.1.weight", "lq_proj.gate_modules.0.content_proj.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
        r"noise_refiner\.\d+\.attention\.(qkv|proj)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|cap_embedder|noise_refiner|t_embedder|final_layer|pos_embed|lq_proj)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="SeedVR2 shares Lumina2-like signatures; disambiguated by hints.",
))

_register(FamilyPolicy(
    family="omnigen2",
    comfyui_classes=("Omnigen2",),
    detect_primary=("time_caption_embed.timestep_embedder.linear_1.bias",
                    "layers.0.attn.to_q.weight"),
    detect_hints=("layers.0.feed_forward.linear_1.weight", "context_refiner.0.attn.to_q.weight",
                  "ref_image_patch_embedder.weight", "x_embedder.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"layers\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
    ),
    keep=(r"(^|\.)(norm_out|image_index_embedding)\.", r"norm\d+\.linear\.(weight|bias)$"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="boogu",
    comfyui_classes=("Boogu",),
    detect_primary=("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
                    "double_stream_layers.0.img_self_attn.to_q.weight",
                    "double_stream_layers.0.img_feed_forward.linear_1.weight"),
    detect_hints=("single_stream_layers.0.attn.to_q.weight",
                  "single_stream_layers.0.feed_forward.linear_1.weight",
                  "context_refiner.0.attn.to_q.weight",
                  "ref_image_patch_embedder.weight", "x_embedder.weight"),
    quantize=(
        r"double_stream_layers\.\d+\.(img_self_attn|img_instruct_attn)\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"double_stream_layers\.\d+\.img_instruct_attn\.processor\.(img_to_q|img_to_k|img_to_v|img_out|instruct_to_q|instruct_to_k|instruct_to_v|instruct_out)\.weight$",
        r"double_stream_layers\.\d+\.(img_feed_forward|instruct_feed_forward)\.(linear_1|linear_2|linear_3)\.weight$",
        r"single_stream_layers\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"single_stream_layers\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
    ),
    keep=(r"(^|\.)(norm_out|image_index_embedding|time_caption_embed)\.", r"norm\d+\.linear\.(weight|bias)$"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Real Boogu-Image-0.1 naming (double/single_stream_layers, img_instruct_attn.processor).",
))

_register(FamilyPolicy(
    family="lens",
    comfyui_classes=("Lens",),
    detect_primary=("transformer_blocks.0.attn.norm_added_q.weight",),
    detect_hints=("transformer_blocks.0.img_mlp.w1.weight", "img_in.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="mage_flow",
    comfyui_classes=("MageFlow",),
    detect_primary=("txt_norm.weight",),
    detect_hints=("proj_out.weight", "transformer_blocks.0.img_attn.qkv.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Disambiguated from qwen_image by txt_norm dim 2560 / proj_out 128.",
))

_register(FamilyPolicy(
    family="qwen_image",
    comfyui_classes=("QwenImage",),
    detect_primary=("txt_norm.weight",),
    detect_hints=("proj_out.weight", "transformer_blocks.0.attn.to_q.weight",
                  "img_in.weight", "time_text_embed.addition_t_embedding.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp|mlp)\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer|"
          r"time_caption_embed|add_k_proj|add_q_proj|add_v_proj|to_add_out)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="ideogram4",
    comfyui_classes=("Ideogram4",),
    detect_primary=("embed_image_indicator.weight",),
    detect_hints=("input_proj.weight", "layers.0.attn.qkv.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(qkv|o|proj)\.weight$",
        r"layers\.\d+\.mlp\.(mlp_in|mlp_out|w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(input_proj|embed_image_indicator|adaln_proj|adaln_modulation|"
          r"llm_cond_proj|time_embed|pos_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="krea2",
    comfyui_classes=("Krea2",),
    detect_primary=("txtfusion.projector.weight",),
    detect_hints=("txtfusion.layerwise_blocks.0.prenorm.scale",
                  "blocks.0.attn.wq.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(wq|wk|wv|wo|gate)\.weight$",
        r"blocks\.\d+\.mlp\.(gate|up|down)\.weight$",
        r"txtfusion\.(layerwise_blocks|refiner_blocks)\.\d+\.(attn|mlp)\.\w+\.weight$",
    ),
    keep=(r"(^|\.)(first|projector|txtfusion\.projector|input_proj|time_embed|"
          r"pos_embed|final_layer|adaln)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Real Kroma v0.2 (Krea2 fine-tune) naming: blocks.N.attn.wq/wk/wv/wo/gate, "
          "blocks.N.mlp.gate/up/down, txtfusion layerwise+refiner blocks. Verified "
          "against the published kroma-v0.2-turbo checkpoint header (K 6144/16384/"
          "2560/6912 are all ConvRot-256 compatible).",
))

_register(FamilyPolicy(
    family="kandinsky5",
    comfyui_classes=("Kandinsky5", "Kandinsky5Image"),
    detect_primary=("visual_transformer_blocks.0.cross_attention.key_norm.weight",),
    detect_hints=("visual_embeddings.in_layer.weight", "text_transformer_blocks.0.self_attention.q_norm.weight"),
    quantize=(
        r"(visual|text)_transformer_blocks\.\d+\.(self_attention|cross_attention)\.(to_query|to_key|to_value|to_out\.0)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.ff\.(in_layer|out_layer)\.weight$",
    ),
    keep=(r"(^|\.)(visual_embeddings|time_embeddings|in_layer|out_layer|head|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="cogvideox",
    comfyui_classes=("CogVideoX_T2V", "CogVideoX_I2V", "CogVideoX_Inpaint"),
    detect_primary=("blocks.0.norm1.linear.weight",),
    detect_hints=("patch_embed.proj.weight", "transformer_blocks.0.attn1.to_q.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
    ),
    keep=(r"(^|\.)(patch_embed|time_embed|text_proj|proj_out|final_layer|ofs_embedding|"
          r"norm1\.linear|cond_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="ernie_image",
    comfyui_classes=("ErnieImage",),
    detect_primary=("layers.0.mlp.linear_fc2.weight",),
    detect_hints=("text_proj.weight", "visual_transformer_blocks.0.feed_forward.in_layer.weight"),
    quantize=(
        r"layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"layers\.\d+\.mlp\.(linear_fc1|linear_fc2)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.(self_attention|cross_attention)\.(to_query|to_key|to_value|to_out\.0)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.feed_forward\.(in_layer|out_layer)\.weight$",
    ),
    keep=(r"(^|\.)(text_proj|visual_embeddings|time_embeddings|patch_embed|final_linear|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(_sd_unet_policy(
    "sd15", ("SD15", "SD15_instructpix2pix", "Stable_Zero123"),
    "SD1.5 UNet (context_dim 768). Attention q/k/v/out are 1x1 convs in the "
    "non-linear-attention layout; only feed-forward linears plus linear-attention "
    "variants are quantized."))

_register(_sd_unet_policy(
    "sd20", ("SD20", "SD21UnclipL", "SD21UnclipH", "SD_X4Upscaler", "LotusD"),
    "SD2.x UNet (context_dim 1024, linear attention)."))

_register(_sd_unet_policy(
    "sdxl", ("SDXL", "SSD1B", "Segmind_Vega", "KOALA_700M", "KOALA_1B",
             "SDXL_instructpix2pix"),
    "SDXL / SSD-1B / Segmind-Vega / KOALA UNets (context_dim 2048)."))

_register(_sd_unet_policy(
    "sdxl_refiner", ("SDXLRefiner",),
    "SDXL refiner UNet (model_channels 384, context_dim 1280)."))

_register(_sd_unet_policy(
    "svd", ("SVD_img2vid", "SV3D_u", "SV3D_p"),
    "SVD / SV3D video UNets (temporal attention)."))

_register(FamilyPolicy(
    family="joyimage",
    comfyui_classes=("JoyImage",),
    detect_primary=("double_blocks.0.attn.img_attn_qkv.weight",
                    "double_blocks.0.attn.img_attn_q_norm.weight"),
    detect_hints=("condition_embedder.time_embedder.linear_1.weight", "img_in.weight"),
    quantize=(
        r"double_blocks\.\d+\.attn\.(img_attn_qkv|img_attn_proj|txt_attn_qkv|txt_attn_proj)\.weight$",
        r"double_blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|condition_embedder|time_embedder|proj_out|"
          r"final_layer|pos_embed|adaln)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

for _fam, _cls in (("rt_detr_v4", ("RT_DETR_v4",)),
                   ("depth_anything3", ("DepthAnything3",)),
                   ("sam3", ("SAM3", "SAM31"))):
    _register(FamilyPolicy(
        family=_fam, comfyui_classes=_cls,
        detect_primary=(
            "encoder.pan_blocks.1.cv4.conv.weight",
            "backbone.embeddings.patch_embeddings.projection.weight",
            "backbone.encoder.layer.0.attention.q_norm.weight",
        ),
        detect_hints=(),
        quantize=(), keep=(), exclude=UNIVERSAL_EXCLUDE,
        runtime_status="unsupported",
        notes="Perception model: ComfyUI loads it through its own node, not through "
              "the mixed-precision (quantized) loader; W4A8 output would not "
              "be consumable. Conversion refused unless --architecture forces it.",
    ))

def family_names() -> List[str]:
    return list(REGISTRY_ORDER)

def get_family(name: str) -> FamilyPolicy:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    for family, policy in REGISTRY.items():
        aliases = (family,) + policy.comfyui_classes
        if any(re.sub(r"[^a-z0-9]", "", alias.lower()) == normalized
               for alias in aliases):
            return policy
    raise UnknownArchitectureError(
        f"unknown architecture {name!r}; use --list-architectures")

UNET_PREFIX_CANDIDATES = ("model.diffusion_model.", "model.model.", "net.")

def unet_prefix_from_keys(keys: Iterable[str]) -> str:
    counts = {c: 0 for c in UNET_PREFIX_CANDIDATES}
    for k in keys:
        for c in UNET_PREFIX_CANDIDATES:
            if k.startswith(c):
                counts[c] += 1
                break
    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    return "model."

@dataclass
class DetectionResult:
    architecture: str
    confidence: str                 # high | medium | low
    policy: FamilyPolicy
    unet_prefix: str
    evidence: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    competing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    classifier_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "confidence": self.confidence,
            "unet_prefix": self.unet_prefix,
            "evidence": self.evidence,
            "hints": self.hints,
            "competing": self.competing,
            "warnings": self.warnings,
            "classifier_info": self.classifier_info,
        }

def _match_signatures(keys: Iterable[str], prefix: str,
                      signatures: Sequence[str]) -> List[str]:
    """Return the signature keys that appear as substrings of some state-dict key
    (with the unet prefix stripped)."""
# SPDX-License-Identifier: Apache-2.0
    stripped = [k[len(prefix):] if k.startswith(prefix) else k for k in keys]
    found = []
    for sig in signatures:
        for k in stripped:
            if sig in k:
                found.append(sig)
                break
    return found

def detect_architecture(info: CheckpointInfo, override: Optional[str] = None,
                        shape_lookup: Optional[Callable[[str], Optional[Tuple[int, ...]]]] = None,
                        ) -> DetectionResult:
    """Detect the model architecture from checkpoint structure alone."""
    keys = list(info.key_set())
    prefix = unet_prefix_from_keys(keys)
    if override:
        policy = get_family(override)
        return DetectionResult(
            architecture=policy.family, confidence="high",
            policy=policy, unet_prefix=prefix,
            evidence=[f"user override --architecture {override}"],
            warnings=["architecture supplied by the user; detection was not used"])
    if shape_lookup is None:
        shape_lookup = lambda name: None  # noqa: E731

    # ---- structural families first (mirroring ComfyUI's branch order) ----
    stripped = [k[len(prefix):] if k.startswith(prefix) else k for k in keys]

    def has(*subs: str) -> bool:
        for s in subs:
            if not any(s in k for k in stripped):
                return False
        return True

    def any_of(*subs: str) -> bool:
        return any(s in k for k in stripped for s in subs)

    candidates: List[Tuple[str, List[str], List[str]]] = []  # (family, evidence, hints)

    # 1. mmdit (SD3 / SD3.5)
    if has("joint_blocks.0.context_block.attn.qkv.weight", "x_embedder.proj.weight"):
        candidates.append(("mmdit_sd3", ["joint_blocks.0.context_block.attn.qkv.weight",
                                         "x_embedder.proj.weight"],
                           [s for s in ("final_layer.linear.weight", "y_embedder.mlp.0.weight",
                                        "context_embedder.weight") if has(s)]))
    # 2. stable cascade
    if has("clf.1.weight"):
        candidates.append(("stable_cascade", ["clf.1.weight"],
                           [s for s in ("clip_txt_mapper.weight", "clip_mapper.weight",
                                        "clip_img_mapper.weight") if has(s)]))
    # 3. stable audio
    if has("transformer.rotary_pos_emb.inv_freq"):
        candidates.append(("stable_audio", ["transformer.rotary_pos_emb.inv_freq"],
                           [s for s in ("to_global_embed.0.weight", "to_timestep_embed.0.weight") if has(s)]))
    # 4. aura flow
    if has("double_layers.0.attn.w1q.weight"):
        candidates.append(("aura_flow", ["double_layers.0.attn.w1q.weight"],
                           [s for s in ("single_layers.0.attn.w1q.weight", "cond_seq_linear.weight") if has(s)]))
    # 5. hydit
    if has("mlp_t5.0.weight"):
        candidates.append(("hydit", ["mlp_t5.0.weight"],
                           [s for s in ("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                                        "extra_embedder.0.weight") if has(s)]))
    # 6. hunyuan video / image
    if has("txt_in.individual_token_refiner.blocks.0.norm1.weight"):
        candidates.append(("hunyuan_video", ["txt_in.individual_token_refiner.blocks.0.norm1.weight"],
                           [s for s in ("img_in.proj.weight", "final_layer.linear.weight",
                                        "double_blocks.0.attn.qkv.weight") if has(s)]))
    # 7. flux / chroma / flux2
    if any_of("double_blocks.0.img_attn.norm.key_norm.weight",
              "double_blocks.0.img_attn.norm.key_norm.scale"):
        if has("double_stream_modulation_img.lin.weight"):
            candidates.append(("flux2", ["double_stream_modulation_img.lin.weight"],
                               [s for s in ("double_stream_layers.0.img_attn.qkv.weight",) if has(s)]))
        elif any_of("distilled_guidance_layer.norms.0.weight",
                    "distilled_guidance_layer.0.norms.0.weight"):
            candidates.append(("chroma", ["distilled_guidance_layer.norms.0.weight"],
                               [s for s in ("nerf_blocks.0.norm.weight", "img_in.weight") if has(s)]))
        else:
            candidates.append(("flux", ["double_blocks.0.img_attn.norm.key_norm.weight"],
                               [s for s in ("img_in.weight", "txt_in.weight",
                                            "single_blocks.0.linear1.weight") if has(s)]))
    # 8. mochi
    if has("t5_yproj.weight"):
        candidates.append(("mochi", ["t5_yproj.weight"],
                           [s for s in ("time_blocks.0.attn.qkv_x.weight",) if has(s)]))
    # 9. minimax h3 (checked before ltxv, like ComfyUI)
    if has("video_patch_proj.weight", "audio_patch_proj.weight"):
        candidates.append(("minimax_h3", ["video_patch_proj.weight", "audio_patch_proj.weight"],
                           [s for s in ("blocks.0.attn.qkv_proj.weight", "final_layer.video_out.weight",
                                        "adaln_t_table") if has(s)]))
    # 10. ltxv / ltxav
    if has("adaln_single.emb.timestep_embedder.linear_1.bias"):
        candidates.append(("ltxv", ["adaln_single.emb.timestep_embedder.linear_1.bias"],
                           [s for s in ("transformer_blocks.0.attn2.to_k.weight",
                                        "audio_adaln_single.linear.weight") if has(s)]))
    # 11. ace-step
    if has("genre_embedder.weight"):
        candidates.append(("ace_step", ["genre_embedder.weight"],
                           [s for s in ("encoder.lyric_encoder.layers.0.input_layernorm.weight",) if has(s)]))
    # 12. pixart
    if has("t_block.1.weight"):
        candidates.append(("pixart", ["t_block.1.weight"],
                           [s for s in ("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                                        "y_embedder.y_embedding") if has(s)]))
    # 13. cosmos
    if has("blocks.block0.blocks.0.block.attn.to_q.0.weight"):
        candidates.append(("cosmos", ["blocks.block0.blocks.0.block.attn.to_q.0.weight"],
                           [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 14. PiD (checked before PixelDiT)
    if has("lq_proj.latent_proj.0.weight"):
        candidates.append(("pixeldit", ["lq_proj.latent_proj.0.weight"],
                           [s for s in ("lq_proj.gate_modules.0.content_proj.weight",
                                        "lq_proj.pit_head.weight") if has(s)]))
    # 15. PixelDiT T2I
    if has("core.pixel_embedder.proj.weight"):
        candidates.append(("pixeldit", ["core.pixel_embedder.proj.weight"],
                           [s for s in ("cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight") if has(s)]))
    # 16. lumina2 / zimage
    if has("cap_embedder.1.weight") and has("noise_refiner.0.attention.k_norm.weight"):
        candidates.append(("lumina2", ["cap_embedder.1.weight"],
                           [s for s in ("layers.0.attn.qkv.weight", "cap_pad_token",
                                        "dec_net.cond_embed.weight") if has(s)]))
    # 17. cogvideox
    if has("blocks.0.norm1.linear.weight"):
        candidates.append(("cogvideox", ["blocks.0.norm1.linear.weight"],
                           [s for s in ("patch_embed.proj.weight", "transformer_blocks.0.attn1.to_q.weight") if has(s)]))
    # 18. wan
    if has("head.modulation"):
        candidates.append(("wan", ["head.modulation"],
                           [s for s in ("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight",
                                        "patch_embedding.weight") if has(s)]))
    # 19. seedvr2 (must be checked before generic lumina2-ish catches; ComfyUI order)
    if any_of("blocks.35.mlp.vid.proj_out.weight", "blocks.35.mlp.all.proj_in_gate.weight",
              "blocks.31.mlp.all.proj_in_gate.weight"):
        candidates.append(("seedvr2", ["blocks.35.mlp.vid.proj_out.weight"
                                       if has("blocks.35.mlp.vid.proj_out.weight")
                                       else "blocks.35.mlp.all.proj_in_gate.weight"],
                           [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 20. cosmos predict2 / anima
    if has("blocks.0.mlp.layer1.weight"):
        if has("__x0__"):
            candidates.append(("anima", ["blocks.0.mlp.layer1.weight", "__x0__"],
                               [s for s in ("layers.0.attn.q_proj.weight",) if has(s)]))
        else:
            candidates.append(("cosmos_predict2", ["blocks.0.mlp.layer1.weight"],
                               [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 21. boogu (checked before omnigen2; both share the embedder skeleton)
    if has("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
           "double_stream_layers.0.img_self_attn.to_q.weight",
           "double_stream_layers.0.img_feed_forward.linear_1.weight"):
        candidates.append(("boogu",
                           ["double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
                            "double_stream_layers.0.img_self_attn.to_q.weight",
                            "double_stream_layers.0.img_feed_forward.linear_1.weight"],
                           [s for s in ("single_stream_layers.0.attn.to_q.weight",
                                        "single_stream_layers.0.feed_forward.linear_1.weight",
                                        "context_refiner.0.attn.to_q.weight",
                                        "ref_image_patch_embedder.weight") if has(s)]))
    # 22. omnigen2
    if has("time_caption_embed.timestep_embedder.linear_1.bias", "layers.0.attn.to_q.weight"):
        candidates.append(("omnigen2",
                           ["time_caption_embed.timestep_embedder.linear_1.bias",
                            "layers.0.attn.to_q.weight"],
                           [s for s in ("layers.0.feed_forward.linear_1.weight",
                                        "context_refiner.0.attn.to_q.weight",
                                        "ref_image_patch_embedder.weight",
                                        "x_embedder.weight") if has(s)]))
    # 23. lens
    if has("transformer_blocks.0.attn.norm_added_q.weight",
           "transformer_blocks.0.img_mlp.w1.weight"):
        candidates.append(("lens", ["transformer_blocks.0.attn.norm_added_q.weight"],
                           [s for s in ("img_in.weight", "proj_out.weight") if has(s)]))
    # 24. mage flow (shape-disambiguated from qwen image)
    if has("txt_norm.weight") and has("proj_out.weight"):
        tn_shape = shape_lookup(prefix + "txt_norm.weight")
        po_shape = shape_lookup(prefix + "proj_out.weight")
        if (tn_shape is not None and len(tn_shape) == 1 and tn_shape[0] == 2560
                and po_shape is not None and len(po_shape) == 1 and po_shape[0] == 128):
            candidates.append(("mage_flow", ["txt_norm.weight", "proj_out.weight"],
                               [s for s in ("transformer_blocks.0.img_attn.qkv.weight",) if has(s)]))
    # 25. qwen image
    if has("txt_norm.weight") and has("proj_out.weight") and not any(c[0] == "mage_flow" for c in candidates):
        candidates.append(("qwen_image", ["txt_norm.weight", "proj_out.weight"],
                           [s for s in ("img_in.weight", "transformer_blocks.0.attn.to_q.weight",
                                        "time_text_embed.addition_t_embedding.weight") if has(s)]))
    # 26. ideogram4
    if has("embed_image_indicator.weight"):
        candidates.append(("ideogram4", ["embed_image_indicator.weight"],
                           [s for s in ("input_proj.weight", "layers.0.attn.qkv.weight") if has(s)]))
    # 27. krea2
    if has("txtfusion.projector.weight"):
        candidates.append(("krea2", ["txtfusion.projector.weight"],
                           [s for s in ("txtfusion.layerwise_blocks.0.prenorm.scale",
                                        "layers.0.attn.wq.weight") if has(s)]))
    # 28. kandinsky5
    if has("visual_transformer_blocks.0.cross_attention.key_norm.weight"):
        candidates.append(("kandinsky5", ["visual_transformer_blocks.0.cross_attention.key_norm.weight"],
                           [s for s in ("visual_embeddings.in_layer.weight",) if has(s)]))
    # 29. ace 1.5 (music)
    if has("encoder.lyric_encoder.layers.0.input_layernorm.weight") and not any(c[0] == "ace_step" for c in candidates):
        candidates.append(("ace_step", ["encoder.lyric_encoder.layers.0.input_layernorm.weight"],
                           [s for s in ("decoder.layers.0.self_attn.q_proj.weight",) if has(s)]))
    # 30. RT-DETR / DepthAnything / SAM3 (perception: unsupported)
    if has("encoder.pan_blocks.1.cv4.conv.weight"):
        candidates.append(("rt_detr_v4", ["encoder.pan_blocks.1.cv4.conv.weight"], []))
    if has("backbone.embeddings.patch_embeddings.projection.weight"):
        candidates.append(("depth_anything3", ["backbone.embeddings.patch_embeddings.projection.weight"],
                           [s for s in ("head.scratch.refinenet1.out_conv.weight",) if has(s)]))
    if has("backbone.encoder.layer.0.attention.q_norm.weight") or has("backbone.encoder.layer.0.attention.self.query.weight"):
        candidates.append(("sam3", ["backbone.encoder.layer.0.attention.q_norm.weight"],
                           [s for s in ("head.projects.0.weight",) if has(s)]))
    # 31. ernie image
    if has("layers.0.mlp.linear_fc2.weight"):
        candidates.append(("ernie_image", ["layers.0.mlp.linear_fc2.weight"],
                           [s for s in ("text_proj.weight",) if has(s)]))
    # 32. classic SD UNets
    sd_family = None
    if "input_blocks.0.0.weight" in stripped:
        mc_shape = shape_lookup(prefix + "input_blocks.0.0.weight")
        model_channels = int(mc_shape[0]) if mc_shape else None
        # context dim from the first transformer block's cross-attention to_k
        ctx_dim = None
        lin_attn = False
        # real checkpoints place the transformer sub-block at index 1
        # (input_blocks.1.1 / 2.1 / middle_block.1); the synthetic fixtures
        # use index 0. Probe the real layouts first, then fall back.
        for probe in ("input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight",
                      "middle_block.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.1.0.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.1.0.transformer_blocks.0.attn2.q.weight",
                      "input_blocks.2.0.transformer_blocks.0.attn2.to_k.weight"):
            shp = shape_lookup(prefix + probe)
            if shp is not None:
                ctx_dim = int(shp[1]) if len(shp) == 2 else None
                lin_attn = probe.endswith("to_k.weight")
                break
        in_ch = None
        in_shp = shape_lookup(prefix + "input_blocks.0.0.weight")
        if in_shp is not None:
            in_ch = int(in_shp[1])
        temporal = any(("time_stack" in k or "temporal_transformer" in k) for k in stripped)
        has_label_emb = has("label_emb.0.0.weight")
        # Local config is supporting evidence only.  Tensor shapes remain the
        # primary signal, but a standard HF field can resolve a missing probe.
        cfg_ctx = info.config.get("cross_attention_dim") or info.config.get("context_dim")
        if ctx_dim is None and isinstance(cfg_ctx, int):
            ctx_dim = int(cfg_ctx)
        if in_ch is None and isinstance(info.config.get("in_channels"), int):
            in_ch = int(info.config["in_channels"])
        classifier_info = {"model_channels": model_channels, "context_dim": ctx_dim,
                           "in_channels": in_ch, "linear_attention": lin_attn,
                           "temporal": temporal, "label_emb": has_label_emb,
                           "config_keys": sorted(info.config.keys())[:32]}
        if model_channels == 256:
            sd_family = "sd20"  # SD_X4Upscaler
        elif ctx_dim == 2048:
            sd_family = "sdxl"
        elif ctx_dim == 1280:
            sd_family = "sdxl_refiner"
        elif ctx_dim == 1024:
            sd_family = "svd" if (temporal or in_ch == 8) else "sd20"
        elif ctx_dim == 768:
            sd_family = "sd15"
        elif model_channels == 384:
            sd_family = "sdxl_refiner"
        if sd_family is not None:
            candidates.append((sd_family, ["input_blocks.0.0.weight"],
                               ["config.cross_attention_dim"]
                               if cfg_ctx is not None else []))

    # ---- score candidates ----
    best: Optional[Tuple[str, int, List[str], List[str]]] = None
    competing = []
    for fam, ev, hints in candidates:
        score = 2 * len(ev) + len(hints)
        if best is None or score > best[1]:
            best = (fam, score, ev, hints)
            competing = [fam]
        elif score == best[1] and fam != best[0]:
            competing.append(fam)

    if best is None:
        # try generic signature matching against the registry for anything missed
        for fam in REGISTRY_ORDER:
            pol = REGISTRY[fam]
            ev = _match_signatures(keys, prefix, pol.detect_primary)
            if ev:
                hints = _match_signatures(keys, prefix, pol.detect_hints)
                score = 2 * len(ev) + len(hints)
                if best is None or score > best[1]:
                    best = (fam, score, ev, hints)
                    competing = [fam]
                elif score == best[1] and fam != best[0]:
                    competing.append(fam)

    if best is None:
        raise UnknownArchitectureError(
            "could not identify the model architecture from checkpoint structure. "
            "Use --architecture NAME (see --list-architectures) to supply one "
            "explicitly, or --inspect to review the checkpoint contents.")

    fam, score, ev, hints = best
    competing = [c for c in competing if c != fam]
    if competing:
        choices = ", ".join([fam] + sorted(set(competing)))
        raise UnknownArchitectureError(
            f"architecture detection is ambiguous between: {choices}. "
            "Pass --architecture NAME after inspecting the checkpoint; refusing "
            "to guess.")
    policy = REGISTRY[fam]
    if len(ev) >= 2:
        confidence = "high"
    elif len(ev) == 1 and len(hints) >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    warnings = []
    if confidence == "low":
        warnings.append("low detection confidence; consider --architecture override")
    if policy.runtime_status == "unsupported":
        warnings.append(
            f"architecture {fam!r} has no ComfyUI quantized-loading path; conversion "
            "would not be consumable by ComfyUI (refusing unless forced)")

    result = DetectionResult(
        architecture=fam, confidence=confidence, policy=policy,
        unet_prefix=prefix, evidence=ev, hints=hints,
        competing=competing, warnings=warnings,
        classifier_info=locals().get("classifier_info", {}))
    return result
