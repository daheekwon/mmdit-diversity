"""Hugging Face pipelines and the per-model attention adapters (SD3.5 Medium, FLUX.1-dev, CogVideoX-2B, FLUX.2-klein-base-4B).

Everything model-specific lives here: loading, prompt encoding, padding-token positions, global block numbers and
where q / k / v are intercepted.
  SD3.5: joint sequence [image | text], text = CLIP 77 + T5 256 tokens, batch rows [uncond | cond], blocks 0-23.
  FLUX:  joint sequence [text | image], text = T5 256 tokens, no CFG rows, blocks 0-18 double + 19-56 single.
  CogVideoX: joint sequence [text | video], text = T5 226 tokens (no mask), one q / k / v projection for both streams,
         batch rows [uncond | cond], blocks 0-29. "Image" in the method means the video tokens (all frames).
  klein: joint sequence [text | image], text = Qwen3 chat template padded to 512 slots (no attention mask, so the padding
         slots are attended), true CFG as two transformer calls per step (conditional first, then the empty prompt),
         blocks 0-4 double + 5-24 single (single blocks take [text | image] as one stream), no q / k / v biases.
"""
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import diffusers.models.transformers.transformer_flux as flux
from diffusers.models.embeddings import apply_rotary_emb

N_BLOCKS = dict(sd35=24, flux=57, cogvideox=30, klein=25)
VIDEO = ('cogvideox',)
CFG_CALLS = dict(klein=2)          # transformer calls per denoising step (others: 1; SD3.5 / CogVideoX batch the CFG rows)
KLEIN_DOUBLE = 5
CLIP_LEN = 77


def load(mcfg, gpu):
    from diffusers import CogVideoXPipeline, FluxPipeline, StableDiffusion3Pipeline
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if mcfg.name == 'klein':
        from diffusers import Flux2KleinPipeline as cls
    else:
        cls = dict(sd35=StableDiffusion3Pipeline, flux=FluxPipeline, cogvideox=CogVideoXPipeline)[mcfg.name]
    extra = dict(cache_dir=mcfg.cache_dir) if mcfg.get('cache_dir') else {}
    pipe = cls.from_pretrained(mcfg.hf_id, revision=mcfg.revision, local_files_only=mcfg.local_files_only,
                               torch_dtype=getattr(torch, mcfg.dtype), **extra)
    if mcfg.name == 'klein':
        tr = pipe.transformer
        if pipe.config.is_distilled or len(tr.transformer_blocks) != KLEIN_DOUBLE or len(tr.single_transformer_blocks) != 20:
            raise ValueError('unexpected FLUX.2-klein layout (expected the undistilled base: 5 double + 20 single blocks)')
    if mcfg.get('vae_tiling'):                 # video VAE: decode one video at a time, in spatial tiles
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    if mcfg.offload:
        pipe.enable_model_cpu_offload(gpu_id=gpu)
    else:
        pipe.to(f'cuda:{gpu}')
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.inference_mode()
def encode(pipe, mcfg, prompt):
    if mcfg.name == 'sd35':
        pe, ne, pp, npool = pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, device=pipe._execution_device,
                                               do_classifier_free_guidance=True, max_sequence_length=mcfg.max_sequence_length)
        return dict(prompt_embeds=pe, negative_prompt_embeds=ne, pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=npool)
    if mcfg.name == 'cogvideox':
        pe, ne = pipe.encode_prompt(prompt=prompt, negative_prompt=None, do_classifier_free_guidance=True,
                                    num_videos_per_prompt=1, max_sequence_length=mcfg.max_sequence_length,
                                    device=pipe._execution_device)
        return dict(prompt_embeds=pe, negative_prompt_embeds=ne)
    if mcfg.name == 'klein':           # the pipeline's own CFG negative is the empty prompt; encoded once, like the prompt
        kw = dict(device=pipe._execution_device, num_images_per_prompt=1, max_sequence_length=mcfg.max_sequence_length)
        pe, _ = pipe.encode_prompt(prompt=prompt, **kw)
        ne, _ = pipe.encode_prompt(prompt='', **kw)
        return dict(prompt_embeds=pe, negative_prompt_embeds=ne)
    pe, pp, _ =pipe.encode_prompt(prompt=prompt, prompt_2=None, device=pipe._execution_device, num_images_per_prompt=1,
                                   max_sequence_length=mcfg.max_sequence_length)
    return dict(prompt_embeds=pe, pooled_prompt_embeds=pp)


class _Stop(Exception):
    pass


def generate(pipe, mcfg, emb, seeds, ctrl=None, last_step=None):
    """One prompt, one image per seed. The initial noise comes from a CPU generator per seed, so an image does not
    depend on the batch it was made in. last_step: stop after this denoising step (basis capture) and return None.
    Video models return one uint8 array [frames, H, W, 3] per seed (the rounding of diffusers' PIL output)."""
    rep = {k: v.repeat(len(seeds), *[1]*(v.ndim-1)) for k, v in emb.items()}

    def callback(p, i, t, kw):
        if not torch.isfinite(kw['latents']).all():
            raise FloatingPointError(f'nonfinite latents at step {i}')
        if last_step is not None and i >= last_step:
            raise _Stop
        return kw
    common = dict(num_inference_steps=mcfg.steps, guidance_scale=mcfg.guidance, max_sequence_length=mcfg.max_sequence_length,
                  generator=[torch.Generator('cpu').manual_seed(int(s)) for s in seeds],
                  callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'])
    with torch.inference_mode(), (ctrl.install(pipe) if ctrl is not None else nullcontext()):
        try:
            if mcfg.name in VIDEO:
                frames = pipe(**rep, height=mcfg.height, width=mcfg.width, num_frames=mcfg.num_frames,
                              use_dynamic_cfg=mcfg.dynamic_cfg, output_type='np', **common).frames
                images = [(f*255).round().astype(np.uint8) for f in frames]
            else:
                images = pipe(**rep, height=mcfg.resolution, width=mcfg.resolution, **common).images
        except _Stop:
            return None
        if ctrl is not None:
            ctrl.verify()
    return images


def pad_index(pipe, model, prompt, max_len=256):
    """Padding-token positions inside the text part of the joint sequence, from the conditional prompt.
    SD3.5: CLIP tokens after the first EOS + T5 pad tokens (the same positions are used in both CFG rows)."""
    if model == 'sd35':
        ci = pipe.tokenizer(prompt, padding='max_length', max_length=CLIP_LEN, truncation=True)['input_ids']
        first_eos = ci.index(pipe.tokenizer.eos_token_id)
        ti = pipe.tokenizer_3(prompt, padding='max_length', max_length=max_len, truncation=True,
                              add_special_tokens=True)['input_ids']
        pos = list(range(first_eos+1, CLIP_LEN))+[CLIP_LEN+i for i, t in enumerate(ti) if t == pipe.tokenizer_3.pad_token_id]
    elif model == 'klein':                     # Qwen3 slots after the chat-template text, exactly as the pipeline tokenizes
        tok = pipe.tokenizer                   # (the empty-prompt CFG call uses the same positions: all padding there too)
        text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        mask = tok(text, padding='max_length', truncation=True, max_length=max_len)['attention_mask']
        pos = [i for i, m in enumerate(mask) if m == 0]
    elif model == 'cogvideox':                 # T5 pad tokens of the 226-token text (CogVideoX: same in both CFG rows)
        ids = pipe.tokenizer(prompt, padding='max_length', max_length=pipe.transformer.config.max_text_seq_length,
                             truncation=True, add_special_tokens=True)['input_ids']
        pos = [i for i, t in enumerate(ids) if t == pipe.tokenizer.pad_token_id]
    else:
        ids = pipe.tokenizer_2(prompt, padding='max_length', max_length=max_len, truncation=True)['input_ids']
        pos = [i for i, t in enumerate(ids) if t == pipe.tokenizer_2.pad_token_id]
    return torch.tensor(pos, dtype=torch.long)


def blocks(pipe, model):
    tr = pipe.transformer
    return list(tr.transformer_blocks)+(list(tr.single_transformer_blocks) if model in ('flux', 'klein') else [])


def attentions(pipe, model):
    """The joint-attention module of every block, in global block order."""
    return [b.attn1 if model == 'cogvideox' else b.attn for b in blocks(pipe, model)]


def n_image_tokens(mcfg):
    return (mcfg.resolution//16)**2


def frame_tokens(mcfg):
    """Video tokens per latent frame (CogVideoX: patch 2 on the VAE's 8x latents, (H / 16) x (W / 16)); the video rows of
    the joint sequence are frame-major (diffusers CogVideoXPatchEmbed: [frames x height x width])."""
    if mcfg.name != 'cogvideox':
        raise ValueError(mcfg.name)
    return (mcfg.height//16)*(mcfg.width//16)


def n_text_tokens(mcfg):
    """Text length of the joint sequence (FLUX.2-klein only: Qwen3 slots, always padded to max_sequence_length)."""
    if mcfg.name != 'klein':
        raise ValueError(mcfg.name)
    return int(mcfg.max_sequence_length)


def value_bias(attn):
    """to_v bias of a joint attention, or None (FLUX.2: no biases; its single blocks fuse q / k / v / MLP-in)."""
    to_v = getattr(attn, 'to_v', None)
    return None if to_v is None else to_v.bias


def value_dim(attn):
    return attn.inner_dim


def set_processors(pipe, model, ctrl):
    """Replace every joint-attention processor; returns a function that restores the native ones."""
    proc = dict(sd35=_SD35Processor, flux=_FluxProcessor, cogvideox=_CogVideoXProcessor, klein=_KleinProcessor)[model]
    natives = []
    for index, attn in enumerate(attentions(pipe, model)):
        natives.append((attn, attn.processor))
        attn.set_processor(proc(attn.processor, index, ctrl))

    def restore():
        for attn, p in natives:
            attn.set_processor(p)
    return restore


class _SD35Processor:
    """diffusers JointAttnProcessor2_0, with the image V rows and the attention call routed through the controller."""

    def __init__(self, native, index, ctrl):
        self.native, self.index, self.ctrl = native, index, ctrl

    @torch.no_grad()
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        act = self.ctrl.active(self.index)
        if not act:
            return self.native(attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs)
        B, n = hidden_states.shape[:2]
        q, k = attn.to_q(hidden_states), attn.to_k(hidden_states)
        v = self.ctrl.image_value(self.index, attn.to_v(hidden_states), attn.to_v.bias, act)
        hd = k.shape[-1]//attn.heads
        sh = lambda x: x.view(B, -1, attn.heads, hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        tq, tk, tv = (sh(p(encoder_hidden_states)) for p in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
        if attn.norm_added_q is not None:
            tq = attn.norm_added_q(tq)
        if attn.norm_added_k is not None:
            tk = attn.norm_added_k(tk)
        q, k, v = torch.cat([q, tq], 2), torch.cat([k, tk], 2), torch.cat([v, tv], 2)
        out = self.ctrl.attend(q, k, v, img=slice(0, n), txt=slice(n, None), act=act)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads*hd).to(q.dtype)
        hs, ehs = out[:, :n], out[:, n:]
        if not attn.context_pre_only:
            ehs = attn.to_add_out(ehs)
        return attn.to_out[1](attn.to_out[0](hs)), ehs


class _FluxProcessor:
    """The native FLUX processor, with its attention dispatch ([B, S, H, D], sequence [text | image]) intercepted."""

    def __init__(self, native, index, ctrl):
        self.native, self.index, self.ctrl = native, index, ctrl
        self.n_img = n_image_tokens(ctrl.cfg.model)

    @torch.no_grad()
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kw):
        act = self.ctrl.active(self.index)
        if not act:
            return self.native(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kw)
        original, ctrl, index, n_img = flux.dispatch_attention_fn, self.ctrl, self.index, self.n_img

        def dispatch(q, k, v, **options):
            nt = q.shape[1]-n_img
            if 'capture' in act or 'v_edit' in act:
                vi = ctrl.image_value(index, v[:, nt:].flatten(2), attn.to_v.bias, act)
                if 'v_edit' in act:
                    v = v.clone()
                    v[:, nt:] = vi.unflatten(-1, v.shape[2:])
            if not ctrl.wants_attend(act):
                return original(q, k, v, **options)
            q, k, v = (x.permute(0, 2, 1, 3) for x in (q, k, v))
            return ctrl.attend(q, k, v, img=slice(nt, None), txt=slice(0, nt), act=act).permute(0, 2, 1, 3)
        flux.dispatch_attention_fn = dispatch
        try:
            return self.native(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kw)
        finally:
            flux.dispatch_attention_fn = original


class _CogVideoXProcessor:
    """diffusers CogVideoXAttnProcessor2_0 (one q / k / v projection for [text | video], q / k LayerNorm, RoPE on the
    video rows when the model has it), with the video V rows and the attention call routed through the controller."""

    def __init__(self, native, index, ctrl):
        self.native, self.index, self.ctrl = native, index, ctrl

    @torch.no_grad()
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, image_rotary_emb=None):
        act = self.ctrl.active(self.index)
        if not act:
            return self.native(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        if attention_mask is not None:
            raise NotImplementedError('CogVideoX attention mask')
        nt = encoder_hidden_states.shape[1]
        x = torch.cat([encoder_hidden_states, hidden_states], 1)
        B = x.shape[0]
        q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        if 'capture' in act or 'v_edit' in act:
            vi = self.ctrl.image_value(self.index, v[:, nt:], attn.to_v.bias, act)
            if 'v_edit' in act:
                v = torch.cat([v[:, :nt], vi], 1)
        hd = k.shape[-1]//attn.heads
        sh = lambda t: t.view(B, -1, attn.heads, hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if image_rotary_emb is not None:
            q[:, :, nt:] = apply_rotary_emb(q[:, :, nt:], image_rotary_emb)
            if not attn.is_cross_attention:
                k[:, :, nt:] = apply_rotary_emb(k[:, :, nt:], image_rotary_emb)
        out = self.ctrl.attend(q, k, v, img=slice(nt, None), txt=slice(0, nt), act=act)
        out = attn.to_out[1](attn.to_out[0](out.transpose(1, 2).reshape(B, -1, attn.heads*hd).to(q.dtype)))
        return out[:, nt:], out[:, :nt]


class _KleinProcessor:
    """The native FLUX.2 processors (Flux2AttnProcessor in double blocks, Flux2ParallelSelfAttnProcessor in single
    blocks), with their attention dispatch ([B, S, H, D] after q / k RMSNorm and RoPE, sequence [text | image])
    intercepted, as in _FluxProcessor. FLUX.2 has no q / k / v biases, so the image V rows are already bias-free."""

    def __init__(self, native, index, ctrl):
        import diffusers.models.transformers.transformer_flux2 as flux2
        want = flux2.Flux2ParallelSelfAttnProcessor if index >= KLEIN_DOUBLE else flux2.Flux2AttnProcessor
        if type(native) is not want:
            raise ValueError(f'block {index}: unsupported processor {type(native).__name__}')
        self.native, self.index, self.ctrl, self.mod = native, index, ctrl, flux2
        self.n_img, self.n_txt = n_image_tokens(ctrl.cfg.model), n_text_tokens(ctrl.cfg.model)

    @torch.no_grad()
    def __call__(self, attn, *args, **kw):
        act = self.ctrl.active(self.index)
        if not act:
            return self.native(attn, *args, **kw)
        mod, ctrl, index, nt = self.mod, self.ctrl, self.index, self.n_txt
        original = mod.dispatch_attention_fn

        def dispatch(q, k, v, **options):
            if q.shape[1] != nt+self.n_img:
                raise RuntimeError(f'joint length {q.shape[1]}, expected {nt}+{self.n_img}')
            if options.get('attn_mask') is not None:
                raise NotImplementedError('FLUX.2 attention mask')
            if 'capture' in act or 'v_edit' in act:
                vi = ctrl.image_value(index, v[:, nt:].flatten(2), None, act)
                if 'v_edit' in act:
                    v = v.clone()
                    v[:, nt:] = vi.unflatten(-1, v.shape[2:])
            if not ctrl.wants_attend(act):
                return original(q, k, v, **options)
            q, k, v = (x.permute(0, 2, 1, 3) for x in (q, k, v))
            return ctrl.attend(q, k, v, img=slice(nt, None), txt=slice(0, nt), act=act).permute(0, 2, 1, 3)
        mod.dispatch_attention_fn = dispatch
        try:
            return self.native(attn, *args, **kw)
        finally:
            mod.dispatch_attention_fn = original
