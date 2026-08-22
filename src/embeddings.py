import os
import time

import numpy as np
import torch


def load_remoteclip(checkpoint_path, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"RemoteCLIP checkpoint did not load cleanly: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    return model, preprocess


def load_clip(device, model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def encode_with_remoteclip(crops, model, preprocess, device, batch_size=16, progress_every=10, log=print):
    """crops: list of (PIL.Image, label). Returns (features [N, D] float32, labels [N])."""
    feats, labels = [], []
    batch_imgs, batch_labels = [], []
    n_done, t0 = 0, time.time()

    def flush():
        nonlocal n_done
        with torch.no_grad():
            t = torch.stack(batch_imgs).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.extend(batch_labels)
        n_done += len(batch_imgs)
        if log is not None and (n_done // batch_size) % progress_every == 0:
            elapsed = time.time() - t0
            log(f"  [remoteclip] {n_done}/{len(crops)} crops, {n_done/elapsed:.1f} crops/s, {elapsed:.0f}s elapsed", flush=True)

    for crop, label in crops:
        batch_imgs.append(preprocess(crop))
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        flush()

    return np.vstack(feats), np.array(labels)


def encode_with_clip(crops, model, processor, device, batch_size=16, progress_every=10, log=print):
    """crops: list of (PIL.Image, label). Returns (features [N, D] float32, labels [N])."""
    feats, labels = [], []
    batch_imgs, batch_labels = [], []
    n_done, t0 = 0, time.time()

    def flush():
        nonlocal n_done
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.extend(batch_labels)
        n_done += len(batch_imgs)
        if log is not None and (n_done // batch_size) % progress_every == 0:
            elapsed = time.time() - t0
            log(f"  [clip] {n_done}/{len(crops)} crops, {n_done/elapsed:.1f} crops/s, {elapsed:.0f}s elapsed", flush=True)

    for crop, label in crops:
        batch_imgs.append(crop)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        flush()

    return np.vstack(feats), np.array(labels)


class CropPromptDataset(torch.utils.data.Dataset):
    """Wraps (PIL.Image, label) crops as (image, text-prompt) pairs for CLIP contrastive fine-tuning."""

    def __init__(self, crops, processor, prompt_template="a photo of a {}"):
        self.crops = crops
        self.processor = processor
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop, label = self.crops[idx]
        prompt = self.prompt_template.format(label)
        inputs = self.processor(
            text=[prompt], images=crop, return_tensors="pt",
            padding="max_length", truncation=True, max_length=77,
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


def clip_contrastive_loss(logits_per_image):
    """Standard CLIP image/text contrastive loss -- diagonal entries are the correct pairs."""
    import torch.nn.functional as F

    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_image.T, labels)
    return (loss_i + loss_t) / 2


def finetune_clip_on_crops(
    crops, model, processor, device,
    num_epochs=3, batch_size=8, lr=1e-5,
    unfreeze_patterns=(
        "vision_model.encoder.layers.11", "vision_model.encoder.layers.10",
        "vision_model.post_layernorm", "visual_projection",
    ),
    prompt_template="a photo of a {}",
    checkpoint_dir=None, checkpoint_prefix="finetuned_clip",
    progress_every=20, log=print,
):
    """Partial contrastive fine-tune of a CLIPModel on (crop, label) pairs.

    Only parameters whose name contains one of unfreeze_patterns are trained (default: last
    2 vision-encoder layers + post-layernorm + visual projection). Saves a checkpoint after
    every epoch if checkpoint_dir is given. Returns the model in eval() mode.
    """
    for name, param in model.named_parameters():
        param.requires_grad = any(p in name for p in unfreeze_patterns)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if log is not None:
        log(f"trainable params: {trainable:,} / {total:,}", flush=True)

    dataset = CropPromptDataset(crops, processor, prompt_template)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(dataloader) * num_epochs))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            loss = clip_contrastive_loss(outputs.logits_per_image)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if log is not None and (step + 1) % progress_every == 0:
                elapsed = time.time() - t0
                done = (step + 1) * batch_size
                rate = done / elapsed
                remaining_steps = len(dataloader) - (step + 1)
                eta_this_epoch = remaining_steps * batch_size / rate if rate > 0 else float("nan")
                log(
                    f"  epoch {epoch+1}/{num_epochs} step {step+1}/{len(dataloader)} "
                    f"loss={loss.item():.4f} avg_loss={total_loss/(step+1):.4f} "
                    f"{rate:.1f} crops/s elapsed={elapsed:.0f}s eta_this_epoch={eta_this_epoch:.0f}s",
                    flush=True,
                )

        avg_loss = total_loss / max(1, len(dataloader))
        epoch_time = time.time() - t0
        if log is not None:
            log(f"epoch {epoch+1}/{num_epochs} done. avg_loss={avg_loss:.4f} time={epoch_time:.0f}s", flush=True)

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            if log is not None:
                log(f"  saved checkpoint: {ckpt_path}", flush=True)

    model.eval()
    return model
