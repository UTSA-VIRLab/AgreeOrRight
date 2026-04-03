"""
Unified model loader for multi-model experiments.

Supports 7 VLMs with a common inference API:
  - llava-hf/llava-1.5-7b-hf          (General)
  - Qwen/Qwen3-VL-8B-Instruct         (General)
  - HuggingFaceM4/idefics2-8b         (General)
  - chaoyinshe/llava-med-v1.5-mistral-7b-hf  (Medical, HF-converted)
  - StanfordAIMI/CheXagent-8b          (Medical, CXR specialist)
  - JZPeterPan/MedVLM-R1              (Medical, Qwen2-VL based)
  - google/medgemma-4b-it              (Medical, Google)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

os.environ.setdefault("HF_HOME", "/raid/den365/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/raid/den365/hf_cache/hub")

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Model registry
# ------------------------------------------------------------------ #

MODEL_REGISTRY = {
    "llava-1.5": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "family": "llava",
        "dtype": "float16",
        "short_name": "LLaVA-1.5",
    },
    "qwen3-vl": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen3",
        "dtype": "bfloat16",
        "short_name": "Qwen3-VL",
    },
    "llava-med": {
        "model_id": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "family": "llava",
        "dtype": "bfloat16",
        "short_name": "LLaVA-Med",
    },
    "medvlm-r1": {
        "model_id": "JZPeterPan/MedVLM-R1",
        "family": "qwen2vl",
        "dtype": "bfloat16",
        "short_name": "MedVLM-R1",
    },
    "medgemma": {
        "model_id": "google/medgemma-4b-it",
        "family": "gemma3",
        "dtype": "bfloat16",
        "short_name": "MedGemma",
    },
    "chexagent": {
        "model_id": "StanfordAIMI/CheXagent-8b",
        "family": "chexagent",
        "dtype": "bfloat16",
        "short_name": "CheXagent",
    },
    "idefics2": {
        "model_id": "HuggingFaceM4/idefics2-8b",
        "family": "idefics2",
        "dtype": "bfloat16",
        "short_name": "IDEFICS2",
    },
}


@dataclass
class VLMWrapper:
    """Unified wrapper for vision-language model inference."""
    model: object
    processor: object
    model_key: str
    family: str
    device: torch.device
    dtype: torch.dtype

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        output_scores: bool = False,
    ):
        """
        Generate text from image + prompt.

        Returns:
            dict with keys:
              - text: generated text string
              - scores: tuple of (vocab_size,) logit tensors per step (if output_scores)
              - generated_ids: token IDs of generated text
        """
        if self.family == "llava":
            return self._generate_llava(
                image, prompt, max_new_tokens, temperature, do_sample, output_scores
            )
        elif self.family in ("qwen3", "qwen2vl"):
            return self._generate_qwen3(
                image, prompt, max_new_tokens, temperature, do_sample, output_scores
            )
        elif self.family == "gemma3":
            return self._generate_gemma3(
                image, prompt, max_new_tokens, temperature, do_sample, output_scores
            )
        elif self.family == "chexagent":
            return self._generate_chexagent(
                image, prompt, max_new_tokens, temperature, do_sample, output_scores
            )
        elif self.family == "idefics2":
            return self._generate_idefics2(
                image, prompt, max_new_tokens, temperature, do_sample, output_scores
            )
        else:
            raise ValueError(f"Unknown model family: {self.family}")

    # ------------------------------------------------------------------ #
    # LLaVA-style (llava-1.5 and llava-med)
    # ------------------------------------------------------------------ #
    def _generate_llava(self, image, prompt, max_new_tokens, temperature, do_sample, output_scores):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            output_logits=output_scores,  # raw logits for L-VASE
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        # Prefer raw logits over processed scores for L-VASE
        logits = getattr(output, "logits", None) if output_scores else None
        return {
            "text": text,
            "scores": logits if logits else (output.scores if output_scores else None),
            "generated_ids": generated_ids,
        }

    # ------------------------------------------------------------------ #
    # Qwen3-VL
    # ------------------------------------------------------------------ #
    def _generate_qwen3(self, image, prompt, max_new_tokens, temperature, do_sample, output_scores):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            output_logits=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        logits = getattr(output, "logits", None) if output_scores else None
        return {
            "text": text,
            "scores": logits if logits else (output.scores if output_scores else None),
            "generated_ids": generated_ids,
        }

    # ------------------------------------------------------------------ #
    # MedGemma (Gemma3)
    # ------------------------------------------------------------------ #
    def _generate_gemma3(self, image, prompt, max_new_tokens, temperature, do_sample, output_scores):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            output_logits=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        logits = getattr(output, "logits", None) if output_scores else None
        return {
            "text": text,
            "scores": logits if logits else (output.scores if output_scores else None),
            "generated_ids": generated_ids,
        }

    # ------------------------------------------------------------------ #
    # CheXagent (BLIP-2 based, radiology specialist)
    # ------------------------------------------------------------------ #
    def _generate_chexagent(self, image, prompt, max_new_tokens, temperature, do_sample, output_scores):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            output_logits=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        logits = getattr(output, "logits", None) if output_scores else None
        return {
            "text": text,
            "scores": logits if logits else (output.scores if output_scores else None),
            "generated_ids": generated_ids,
        }

    # ------------------------------------------------------------------ #
    # IDEFICS2 (HuggingFace multimodal)
    # ------------------------------------------------------------------ #
    def _generate_idefics2(self, image, prompt, max_new_tokens, temperature, do_sample, output_scores):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            output_logits=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        logits = getattr(output, "logits", None) if output_scores else None
        return {
            "text": text,
            "scores": logits if logits else (output.scores if output_scores else None),
            "generated_ids": generated_ids,
        }

    # ------------------------------------------------------------------ #
    # Multi-turn generation (for VIPER sycophancy)
    # ------------------------------------------------------------------ #
    def generate_multiturn(
        self,
        image: Image.Image,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        do_sample: bool = True,
        output_scores: bool = False,
    ):
        """
        Generate response in a multi-turn conversation context.

        messages: list of {"role": "user"|"assistant", "content": str or list}
        """
        if self.family == "llava":
            return self._multiturn_llava(image, messages, max_new_tokens, temperature, do_sample, output_scores)
        elif self.family == "qwen3":
            return self._multiturn_chat_template(image, messages, max_new_tokens, temperature, do_sample, output_scores)
        elif self.family == "qwen2vl":
            return self._multiturn_chat_template(image, messages, max_new_tokens, temperature, do_sample, output_scores)
        elif self.family == "gemma3":
            return self._multiturn_chat_template(image, messages, max_new_tokens, temperature, do_sample, output_scores)
        elif self.family == "chexagent":
            return self._multiturn_chexagent(image, messages, max_new_tokens, temperature, do_sample, output_scores)
        elif self.family == "idefics2":
            return self._multiturn_idefics2(image, messages, max_new_tokens, temperature, do_sample, output_scores)
        else:
            raise ValueError(f"Unknown model family: {self.family}")

    def _multiturn_llava(self, image, messages, max_new_tokens, temperature, do_sample, output_scores):
        # Build conversation with image in first user message
        conversation = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                if i == 0:
                    content = [{"type": "image"}, {"type": "text", "text": content}]
                else:
                    content = [{"type": "text", "text": content}]
            conversation.append({"role": msg["role"], "content": content})

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return {
            "text": text,
            "scores": output.scores if output_scores else None,
            "generated_ids": generated_ids,
        }

    def _multiturn_chat_template(self, image, messages, max_new_tokens, temperature, do_sample, output_scores):
        # Qwen3 and Gemma3 use apply_chat_template with messages
        conversation = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                if i == 0:
                    content = [
                        {"type": "image", "image": image},
                        {"type": "text", "text": content},
                    ]
                else:
                    content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # Already structured content — ensure image is PIL for first msg
                new_content = []
                for item in content:
                    if item.get("type") == "image" and "image" not in item:
                        new_content.append({"type": "image", "image": image})
                    else:
                        new_content.append(item)
                content = new_content
            conversation.append({"role": msg["role"], "content": content})

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return {
            "text": text,
            "scores": output.scores if output_scores else None,
            "generated_ids": generated_ids,
        }

    def _multiturn_chexagent(self, image, messages, max_new_tokens, temperature, do_sample, output_scores):
        # CheXagent doesn't have native multi-turn; concatenate conversation
        conv_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
            if role == "user":
                conv_parts.append(f"User: {content}")
            else:
                conv_parts.append(f"Assistant: {content}")
        prompt = "\n".join(conv_parts) + "\nAssistant:"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return {
            "text": text,
            "scores": output.scores if output_scores else None,
            "generated_ids": generated_ids,
        }

    def _multiturn_idefics2(self, image, messages, max_new_tokens, temperature, do_sample, output_scores):
        # Build conversation with image in first user message
        conversation = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                if i == 0:
                    content = [
                        {"type": "image"},
                        {"type": "text", "text": content},
                    ]
                else:
                    content = [{"type": "text", "text": content}]
            conversation.append({"role": msg["role"], "content": content})

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=output_scores,
            return_dict_in_generate=True,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output.sequences[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return {
            "text": text,
            "scores": output.scores if output_scores else None,
            "generated_ids": generated_ids,
        }


# ------------------------------------------------------------------ #
# Loader function
# ------------------------------------------------------------------ #

def load_model(model_key: str, device: Optional[str] = None) -> VLMWrapper:
    """
    Load a VLM by its registry key.

    Args:
        model_key: one of 'llava-1.5', 'qwen3-vl', 'llava-med', 'chexagent', 'medgemma'
        device: e.g. 'cuda:0'. If None, uses device_map='auto'.

    Returns:
        VLMWrapper with unified generate() API.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}. Choose from {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_key]
    model_id = info["model_id"]
    family = info["family"]
    dtype = getattr(torch, info["dtype"])

    log.info(f"Loading {info['short_name']} ({model_id}) on {device or 'auto'}...")

    if family == "llava":
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device or "auto", low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    elif family == "qwen3":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device or "auto", low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    elif family == "qwen2vl":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device or "auto", low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    elif family == "gemma3":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device or "auto", low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    elif family == "chexagent":
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Patch get_head_mask (removed in transformers 5.x)
        def _get_head_mask(self, head_mask, num_hidden_layers, **kw):
            return [None] * num_hidden_layers

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=dtype, device_map=device or "auto", low_cpu_mem_usage=True,
        )
        # Apply patch to QFormer instance
        if hasattr(model, 'qformer') and not hasattr(model.qformer, 'get_head_mask'):
            model.qformer.get_head_mask = lambda head_mask, num_hidden_layers, **kw: [None] * num_hidden_layers

    elif family == "idefics2":
        from transformers import Idefics2ForConditionalGeneration, AutoProcessor

        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device or "auto", low_cpu_mem_usage=True,
            attn_implementation='eager',
        )
        processor = AutoProcessor.from_pretrained(model_id)

    else:
        raise ValueError(f"Unknown family: {family}")

    model.eval()

    # Resolve device
    if hasattr(model, "device"):
        resolved_device = model.device
    else:
        resolved_device = torch.device(device or "cuda")

    log.info(f"  {info['short_name']} loaded on {resolved_device}")

    return VLMWrapper(
        model=model,
        processor=processor,
        model_key=model_key,
        family=family,
        device=resolved_device,
        dtype=dtype,
    )
