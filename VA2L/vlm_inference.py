import os
# This pipeline uses PyTorch-only inference; avoid importing TensorFlow backends.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from PIL import Image
import torch
import time


class VLMInference:
    """Local vision-language inference for robot task intent (Qwen or Gemma)."""

    DEFAULT_MAX_NEW_TOKENS = 96

    def __init__(
        self,
        model: str = "qwen",
        model_size: str = "base",
        model_id: str = None,
        device: str = "cuda:0",
        precision: str = "auto",
    ):
        self.model = model.lower()
        self.model_size = model_size.lower()
        self.device = device
        self.precision = precision.lower()
        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS

        if self.model_size not in {"base", "small"}:
            raise ValueError(f"Unknown model_size: {model_size}. Use 'base' or 'small'.")

        if model_id is None:
            if self.model == "qwen" and self.model_size == "base":
                model_id = "Qwen/Qwen3-VL-4B-Instruct"
            elif self.model == "qwen" and self.model_size == "small":
                model_id = "Qwen/Qwen3-VL-2B-Instruct"
            elif self.model == "gemma" and self.model_size == "base":
                model_id = "google/gemma-3-4b-it"
            elif self.model == "gemma" and self.model_size == "small":
                model_id = "google/gemma-3-1b-it"
            else:
                raise ValueError(f"Unknown model: {model}. Use 'qwen' or 'gemma'.")
        
        self.model_id = model_id
        self.is_multimodal = self.model == "qwen"
        self.torch_dtype = self._resolve_torch_dtype()

        print(f"Loading model: {model_id} (type: {self.model}, size: {self.model_size})")
        print(f"Precision: {self.precision} -> {self.torch_dtype}")

        if self.is_multimodal:
            self.model_obj = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map={"": device},
                dtype=self.torch_dtype,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": device},
                dtype=self.torch_dtype,
            ).eval()
            self.processor = AutoTokenizer.from_pretrained(model_id)

        first_param_device = next(self.model_obj.parameters()).device
        print(f"Model loaded successfully on device: {first_param_device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    def _resolve_torch_dtype(self):
        if self.precision == "fp32":
            return torch.float32
        if self.precision == "fp16":
            return torch.float16 if str(self.device).startswith("cuda") else torch.float32
        if self.precision == "bf16":
            return torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        if self.precision == "auto":
            return torch.float16 if str(self.device).startswith("cuda") else torch.float32
        raise ValueError("Unknown precision: {0}. Use 'auto', 'fp16', 'bf16', or 'fp32'.".format(self.precision))


    def infer(self, image: Image.Image, instruction: str) -> str:
        """
        Infer robot task intent from image with trajectory overlay and instruction prompt.

        Args:
            image: PIL Image with overlaid trajectory and action arrow.
            instruction: Text instruction describing the image overlays.

        Returns:
            Decoded inference result describing inferred task intent.
        """
        concise_instruction = instruction.strip() + "\n\nPlease answer in one short sentence only. No bullets."

        if self.is_multimodal:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": concise_instruction},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model_obj.device)
        else:
            messages = [
                {
                    "role": "user",
                    "content": concise_instruction,
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model_obj.device)

        input_len = inputs["input_ids"].shape[-1]

        start_time = time.perf_counter()
        with torch.inference_mode():
            generation = self.model_obj.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
            generation = generation[0][input_len:]
        elapsed = time.perf_counter() - start_time

        if generation.shape[-1] >= self.max_new_tokens:
            print(
                f"Warning: output reached max_new_tokens={self.max_new_tokens}. "
                "The result may be truncated. Increase VLMInference.DEFAULT_MAX_NEW_TOKENS for longer answers."
            )

        result = self.processor.decode(generation, skip_special_tokens=True)
        # print(f"VLM inference time: {elapsed:.3f} s")
        return result


def infer_task_intent(
    image: Image.Image,
    instruction: str,
    model: str = "qwen",
    model_size: str = "base",
    model_id: str = None,
    device: str = "cuda:0",
    precision: str = "auto",
) -> str:
    """Standalone function for single inference call."""
    vlm = VLMInference(
        model=model,
        model_size=model_size,
        model_id=model_id,
        device=device,
        precision=precision,
    )
    return vlm.infer(image, instruction)
