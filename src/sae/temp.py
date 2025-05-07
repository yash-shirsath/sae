from datasets import (
    IterableDataset,
    load_from_disk,
    Features,
    Value,
    Sequence,
    Array2D,
)
from torch.utils.data import DataLoader
import torch

# -----------------------------------------------------------------------------
# 0) your prompt list, model, tokenizer, and target layer
# -----------------------------------------------------------------------------
prompts = [...]  # e.g. 10 000 prompts in a Python list

model.eval()
tokenizer  # assume already initialized
layer = model.base_model.encoder.layer[-1].output
batch_size = 32


# -----------------------------------------------------------------------------
# 1) batched activation generator
# -----------------------------------------------------------------------------
def batched_activation_generator():
    storage = {}

    def hook_fn(module, inp, out):
        storage["act"] = out.detach()

    handle = layer.register_forward_hook(hook_fn)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # tokenize the batch with padding -> (B, L)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(model.device)

        # run model; hook_fn will stash the tensor in storage["act"]
        _ = model(**inputs)

        # pull out and clear
        act = storage.pop("act")  # Tensor shape (B, L, H)
        vecs = act[:, 0, :].cpu().numpy()  # → (B, H) e.g. [CLS] token

        # yield one “example” that is really a batch
        yield {
            "prompts": batch_prompts,  # list[str], length B
            "activations": vecs,  # numpy float32 array (B, H)
        }

    handle.remove()


# -----------------------------------------------------------------------------
# 2) wrap in IterableDataset and save to disk
# -----------------------------------------------------------------------------
ds = IterableDataset.from_generator(
    batched_activation_generator,
    features=Features(
        {
            "prompts": Sequence(Value("string")),  # variable-length list
            "activations": Array2D(
                shape=(None, model.config.hidden_size), dtype="float32"
            ),  # 2D float32 (B, H)
        }
    ),
)

# this will shard your stream into ∼400 MB Arrow files under “activations_ds/”
ds.save_to_disk("activations_ds/")

# -----------------------------------------------------------------------------
# 3) later: reload with memory-map and zero-copy Torch tensors
# -----------------------------------------------------------------------------
ds2 = load_from_disk("activations_ds/", keep_in_memory=False)
ds2 = ds2.with_format("torch", columns=["activations"])

loader = DataLoader(
    ds2,
    batch_size=None,  # each “example” is already a batch
    shuffle=False,  # IterableDataset doesn’t support built-in shuffle
    num_workers=4,
)

# -----------------------------------------------------------------------------
# 4) training loop
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(3):
    # if you want to reshuffle each epoch, you could random.shuffle(prompts) here
    for batch in loader:
        prompts_batch = batch["prompts"]  # Python list of B strings
        acts: torch.Tensor = batch["activations"]  # shape [B, H]

        # e.g. feed them back in, or your own head:
        outputs = model(inputs_embeds=acts)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch} done.")
