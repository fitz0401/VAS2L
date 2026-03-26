from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import time


class VLMInference:
    """Local vision-language inference for robot task intent (Qwen or Gemma)."""

    def __init__(self, model: str = "qwen", model_id: str = None, device: str = "cuda:0"):
        self.model = model.lower()
        if model_id is None:
            if self.model == "qwen":
                model_id = "Qwen/Qwen3-VL-4B-Instruct"
            elif self.model == "gemma":
                model_id = "google/gemma-3-4b-it"
            else:
                raise ValueError(f"Unknown model: {model}. Use 'qwen' or 'gemma'.")
        
        self.model_id = model_id
        self.device = device
        print(f"Loading model: {model_id} (type: {self.model})")
        self.model_obj = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map={"": device}
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Model loaded successfully")


    def infer(self, image: Image.Image, instruction: str, max_tokens: int = 150) -> str:
        """
        Infer robot task intent from image with trajectory overlay and instruction prompt.

        Args:
            image: PIL Image with overlaid trajectory and action arrow.
            instruction: Text instruction describing the image overlays.
            max_tokens: Maximum tokens to generate.

        Returns:
            Decoded inference result describing inferred task intent.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
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

        input_len = inputs["input_ids"].shape[-1]

        start_time = time.perf_counter()
        with torch.inference_mode():
            generation = self.model_obj.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
            generation = generation[0][input_len:]
        elapsed = time.perf_counter() - start_time

        result = self.processor.decode(generation, skip_special_tokens=True)
        print(f"VLM inference time: {elapsed:.3f} s")
        return result


def infer_task_intent(
    image: Image.Image,
    instruction: str,
    model: str = "qwen",
    model_id: str = None,
    device: str = "cuda:0",
) -> str:
    """Standalone function for single inference call."""
    vlm = VLMInference(model=model, model_id=model_id, device=device)
    return vlm.infer(image, instruction)
