import importlib
import torch

MODEL_ID = "lerobot/smolvla_base"
REVISION = None

# ---------------------------------------------------------
# Import SmolVLA
# ---------------------------------------------------------

for mod_name in (
    "lerobot.policies.smolvla.modeling_smolvla",
    "lerobot.common.policies.smolvla.modeling_smolvla",
):
    try:
        mod = importlib.import_module(mod_name)
        SmolVLAPolicy = getattr(mod, "SmolVLAPolicy")
        print(f"Imported SmolVLAPolicy from: {mod_name}")
        break
    except (ImportError, AttributeError):
        continue
else:
    raise ImportError("Could not find SmolVLAPolicy in expected lerobot paths.")


# ---------------------------------------------------------
# Check CUDA
# ---------------------------------------------------------

assert torch.cuda.is_available(), "CUDA is not available"

device = torch.device("cuda")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

torch.cuda.reset_peak_memory_stats(device)

kwargs = {"revision": REVISION} if REVISION else {}

policy = SmolVLAPolicy.from_pretrained(
    MODEL_ID,
    **kwargs,
)

policy = policy.to(device=device, dtype=dtype).eval()

load_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)


# ---------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------

trainable_params = sum(
    p.numel() for p in policy.parameters() if p.requires_grad
)

frozen_params = sum(
    p.numel() for p in policy.parameters() if not p.requires_grad
)

total_params = trainable_params + frozen_params

print(f"\nTotal params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params:    {frozen_params:,}")
print(f"Peak VRAM after load ({dtype}): {load_peak_mb:.1f} MB")


# ---------------------------------------------------------
# Print configuration
# ---------------------------------------------------------

print("\n--- policy.config ---")
print(policy.config)

input_features = getattr(policy.config, "input_features", None)
output_features = getattr(policy.config, "output_features", None)

print("\n--- input_features ---")
print(input_features)

print("\n--- output_features ---")
print(output_features)


# ---------------------------------------------------------
# Build dummy observation batch
# ---------------------------------------------------------

batch = {}

if isinstance(input_features, dict):
    for key, feat in input_features.items():

        shape = getattr(feat, "shape", None)

        if shape is None and isinstance(feat, dict):
            shape = feat.get("shape")

        if shape is not None:
            batch[key] = torch.zeros(
                (1, *shape),
                device=device,
                dtype=dtype,
            )


# ---------------------------------------------------------
# Add language instruction
# ---------------------------------------------------------

prompt = "insert the peg into the round hole"

tokenizer = policy.model.vlm_with_expert.processor.tokenizer

tokens = tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=policy.config.tokenizer_max_length,
)

batch["observation.language.tokens"] = tokens["input_ids"].to(device)

batch["observation.language.attention_mask"] = (
    tokens["attention_mask"].to(device).bool()
)

# Make sure every floating-point tensor matches the model dtype.
# Integer (token ids) and bool (attention mask) tensors are left alone.
batch = {
    k: (v.to(dtype) if torch.is_floating_point(v) else v)
    for k, v in batch.items()
}

print(
    "\nDummy batch keys, shapes & dtypes:",
    {
        k: (tuple(v.shape), str(v.dtype)) if hasattr(v, "shape") else type(v)
        for k, v in batch.items()
    },
)


# ---------------------------------------------------------
# Dummy forward pass
# ---------------------------------------------------------
# The policy creates some tensors internally in fp32 (e.g. the initial
# denoising noise). Autocast casts them per-op so they don't clash with
# the bf16/fp16 weights.

if hasattr(policy, "reset"):
    policy.reset()  # clear the internal action queue

torch.cuda.reset_peak_memory_stats(device)

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
    if hasattr(policy, "predict_action_chunk"):
        out = policy.predict_action_chunk(batch)  # full (1, chunk_size, action_dim)
    elif hasattr(policy, "select_action"):
        out = policy.select_action(batch)
    else:
        out = policy(batch)

torch.cuda.synchronize(device)
fwd_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

print(f"\nPeak VRAM during dummy forward pass: {fwd_peak_mb:.1f} MB")

if hasattr(out, "shape"):
    print("Output shape:", tuple(out.shape))

print("\nOK: SmolVLA loaded and completed a dummy forward pass.")