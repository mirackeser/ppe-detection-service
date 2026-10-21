"""Optional local advisory VLM. Free-text output never overrides PPE policy."""
from dataclasses import asdict, dataclass
from time import perf_counter

from PIL import Image

PROMPT = ("Describe the safety equipment visible on this person. "
          "Is a hard hat visible? Is a high-visibility safety vest visible? "
          "If an item is obscured or unclear, say unknown. Do not assume missing equipment.")


@dataclass
class Review:
    model: str
    device: str
    prompt: str
    text: str
    latency_ms: float
    advisory_only: bool = True
    revision: str | None = None

    def to_dict(self):
        return asdict(self)


class SmolVLMReviewer:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu",
                 cache_dir="models/huggingface"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model_id, self.device = model_id, device
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir,
                                                       trust_remote_code=False, token=False,
                                                       local_files_only=True)
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, cache_dir=cache_dir, dtype=dtype,
            trust_remote_code=False, token=False, use_safetensors=True,
            local_files_only=True, _attn_implementation="eager").to(device).eval()

    def review(self, image: Image.Image) -> Review:
        import torch
        def sync():
            if self.device == "mps":
                torch.mps.synchronize()
            elif self.device.startswith("cuda"):
                torch.cuda.synchronize()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        sync()
        start = perf_counter()
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt")
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        text = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        sync()
        return Review(self.model_id, self.device, PROMPT, text, (perf_counter() - start) * 1000,
                      revision=getattr(self.model.config, "_commit_hash", None))
