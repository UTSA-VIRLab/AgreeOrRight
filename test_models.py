#!/usr/bin/env python3
"""Quick test: load candidate models and verify generation + logit extraction."""
import os, sys, torch
os.environ['HF_HOME'] = '/raid/den365/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/raid/den365/hf_cache/hub'
from PIL import Image

img = Image.new('RGB', (224, 224), color='gray')

def test_model(name, load_fn, gen_fn=None):
    print(f"\n{'='*60}")
    print(f"Testing {name}...")
    try:
        model, proc, device = load_fn()
        model.eval()
        print(f"  Loaded on {device}")

        if gen_fn:
            text, n_scores, has_logits, logit_shape = gen_fn(model, proc, device)
        else:
            prompt = "Describe this medical image."
            inputs = proc(images=img, text=prompt, return_tensors="pt").to(device)
            print(f"  Input keys: {list(inputs.keys())}")

            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=30, do_sample=False,
                    return_dict_in_generate=True, output_scores=True, output_logits=True
                )

            gen_ids = out.sequences[0][inputs['input_ids'].shape[-1]:]
            text = proc.decode(gen_ids, skip_special_tokens=True)
            n_scores = len(out.scores) if out.scores else 0
            has_logits = hasattr(out, 'logits') and out.logits is not None
            logit_shape = out.logits[0].shape if has_logits else (out.scores[0].shape if n_scores > 0 else None)

        print(f"  Generated: {repr(text[:100])}")
        print(f"  Scores: {n_scores}, Logits: {has_logits}")
        if logit_shape:
            print(f"  Shape: {logit_shape}")
        print(f"  SUCCESS")
        del model, proc
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        return False

# ---- CheXagent ----
def load_chexagent():
    from transformers import AutoModelForCausalLM, AutoProcessor

    # Patch get_head_mask (removed in transformers 5.x) before loading
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError
        return [None] * num_hidden_layers

    proc = AutoProcessor.from_pretrained('StanfordAIMI/CheXagent-8b', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        'StanfordAIMI/CheXagent-8b', trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='cuda:6', low_cpu_mem_usage=True
    )
    # Patch get_head_mask on QFormer instance after loading
    if hasattr(model, 'qformer') and not hasattr(model.qformer, 'get_head_mask'):
        model.qformer.get_head_mask = lambda head_mask, num_hidden_layers, **kw: [None] * num_hidden_layers
    return model, proc, 'cuda:6'

# ---- Phi-3.5-Vision ----
def load_phi35():
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, DynamicCache

    # Patch DynamicCache for transformers 5.x compat with Phi-3.5 custom code
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, 'get_max_length'):
        DynamicCache.get_max_length = lambda self: getattr(self, 'max_cache_len', None)

    config = AutoConfig.from_pretrained('microsoft/Phi-3.5-vision-instruct', trust_remote_code=True)
    config._attn_implementation = 'eager'
    config._attn_implementation_internal = 'eager'
    proc = AutoProcessor.from_pretrained('microsoft/Phi-3.5-vision-instruct', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct', trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16, device_map='cuda:7', low_cpu_mem_usage=True,
    )
    return model, proc, 'cuda:7'

def gen_phi35(model, proc, device):
    """Phi-3.5-Vision uses chat template with <|image_1|> placeholder."""
    messages = [
        {"role": "user", "content": "<|image_1|>\nDescribe this medical image."},
    ]
    prompt = proc.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(prompt, [img], return_tensors="pt").to(device)
    print(f"  Input keys: {list(inputs.keys())}")

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            return_dict_in_generate=True, output_scores=True, output_logits=True
        )

    gen_ids = out.sequences[0][inputs['input_ids'].shape[-1]:]
    text = proc.decode(gen_ids, skip_special_tokens=True)
    n_scores = len(out.scores) if out.scores else 0
    has_logits = hasattr(out, 'logits') and out.logits is not None
    logit_shape = out.logits[0].shape if has_logits else (out.scores[0].shape if n_scores > 0 else None)
    return text, n_scores, has_logits, logit_shape

# ---- IDEFICS2 ----
def load_idefics2():
    from transformers import Idefics2ForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained('HuggingFaceM4/idefics2-8b')
    model = Idefics2ForConditionalGeneration.from_pretrained(
        'HuggingFaceM4/idefics2-8b',
        torch_dtype=torch.bfloat16, device_map='cuda:5', low_cpu_mem_usage=True,
        attn_implementation='eager',
    )
    return model, proc, 'cuda:5'

def gen_idefics2(model, proc, device):
    """IDEFICS2 needs apply_chat_template with images."""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this medical image."},
        ]},
    ]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[img], return_tensors="pt").to(device)
    print(f"  Input keys: {list(inputs.keys())}")

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            return_dict_in_generate=True, output_scores=True, output_logits=True
        )

    gen_ids = out.sequences[0][inputs['input_ids'].shape[-1]:]
    text = proc.decode(gen_ids, skip_special_tokens=True)
    n_scores = len(out.scores) if out.scores else 0
    has_logits = hasattr(out, 'logits') and out.logits is not None
    logit_shape = out.logits[0].shape if has_logits else (out.scores[0].shape if n_scores > 0 else None)
    return text, n_scores, has_logits, logit_shape

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if which in ("chexagent", "all"):
        results["CheXagent"] = test_model("CheXagent-8b", load_chexagent)
    if which in ("phi35", "all"):
        results["Phi-3.5-Vision"] = test_model("Phi-3.5-Vision", load_phi35, gen_phi35)
    if which in ("idefics2", "all"):
        results["IDEFICS2"] = test_model("IDEFICS2-8b", load_idefics2, gen_idefics2)
    print(f"\n{'='*60}")
    print("Results:", results)
