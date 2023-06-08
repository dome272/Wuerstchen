import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import numpy as np
import wandb
import os
import shutil
from transformers import AutoTokenizer, CLIPTextModel
import webdataset as wds
from webdataset.handlers import warn_and_continue
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from vqgan import VQModel
from modules import Paella, sample, EfficientNetEncoder, Wrapper
from utils import WebdatasetFilter, transforms, effnet_preprocess, identity
import transformers
from transformers.utils import is_torch_bf16_available, is_torch_tf32_available
transformers.utils.logging.set_verbosity_error()

# PARAMETERS
updates = 1500000
warmup_updates = 10000
ema_start = 5000
ema_every = 100
ema_beta = 0.9
batch_size = 384
grad_accum_steps = 1
max_iters = updates * grad_accum_steps
print_every = 1000 * grad_accum_steps
extra_ckpt_every = 10000 * grad_accum_steps
lr = 1e-4
generate_new_wandb_id = False

dataset_path = ""
run_name = "Würstchen-Paella-v4-512-CLIP-text"
output_path = f"output/würstchen/{run_name}"
os.makedirs(output_path, exist_ok=True)
checkpoint_dir = f"models/würstchen/"
checkpoint_path = os.path.join(checkpoint_dir, run_name, "model.pt")
os.makedirs(os.path.join(checkpoint_dir, run_name), exist_ok=True)

wandv_project = ""
wandv_entity = ""
wandb_run_name = run_name


def ddp_setup(rank, world_size, n_node, node_id):  # <--- DDP
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "33751"
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rank + node_id * world_size, world_size=world_size * n_node,
        init_method="file:///mnt/nvme/home/dome/src/würstchen/dist_file4",
    )
    print(f"[GPU {rank + node_id * world_size}] READY")


def train(gpu_id, world_size, n_nodes):
    node_id = int(os.environ["SLURM_PROCID"])
    main_node = gpu_id == 0 and node_id == 0
    ddp_setup(gpu_id, world_size, n_nodes, node_id)  # <--- DDP
    device = torch.device(gpu_id)

    # only ampere gpu architecture allows these
    _float16_dtype = torch.float16 if not is_torch_bf16_available() else torch.bfloat16
    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- PREPARE DATASET ---
    dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=warn_and_continue
    ).select(
        WebdatasetFilter(min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)
    ).shuffle(690, handler=warn_and_continue).decode(
        "pilrgb", handler=warn_and_continue
    ).to_tuple(
        "jpg", "txt", handler=warn_and_continue
    ).map_tuple(
        transforms, identity, handler=warn_and_continue
    )

    real_batch_size = batch_size // (world_size * n_nodes * grad_accum_steps)

    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)

    if main_node:
        print("REAL BATCH SIZE / DEVICE:", real_batch_size)

    # --- PREPARE MODELS ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) if os.path.exists(checkpoint_path) else None
    except RuntimeError as e:
        if os.path.exists(f"{checkpoint_path}.bak"):
            os.remove(checkpoint_path)
            shutil.copyfile(f"{checkpoint_path}.bak", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            raise e

    # - vqmodel -
    vqmodel = VQModel().to(device)
    vqmodel.load_state_dict(torch.load("models/vqgan_f4_v1_500k.pt", map_location=device)['state_dict'])
    vqmodel.eval().requires_grad_(False)

    # - CLIP text encoder
    clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(
        device).eval().requires_grad_(False)
    clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    # - Paella Model as generator - 
    generator = Paella(byt5_embd=1024).to(device)
    if checkpoint is not None:
        generator.load_state_dict(checkpoint['state_dict'])

    # - EfficientNet -
    effnet = EfficientNetEncoder(effnet="efficientnet_v2_l").to(device)
    if checkpoint is not None:
        if "effnet_state_dict" in checkpoint:
            effnet.load_state_dict(checkpoint['effnet_state_dict'])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Wrapper(effnet, generator, device=device).to(device))

    model = DDP(model, device_ids=[gpu_id], output_device=device)  # <--- DDP

    # - SETUP WANDB - 
    if main_node:
        print("Num trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        if checkpoint is not None and not generate_new_wandb_id:
            run_id = checkpoint['wandb_run_id']
        else:
            run_id = wandb.util.generate_id()
        wandb.init(project=wandv_project, name=wandb_run_name, entity=wandv_entity, id=run_id, resume="allow")

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))  # eps=1e-4
    # optimizer = Lion(model.parameters(), lr=lr / 3) # eps=1e-4
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_updates)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Failed loading optimizer, skipping...")
        scheduler.last_epoch = checkpoint['scheduler_last_step']
    scaler = torch.cuda.amp.GradScaler()
    if checkpoint is not None and 'grad_scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    start_iter = 1
    grad_norm = torch.tensor(0, device=device)
    if checkpoint is not None:
        start_iter = checkpoint['scheduler_last_step'] * grad_accum_steps + 1
        if main_node:  # <--- DDP
            print("RESUMING TRAINING FROM ITER ", start_iter)

    skipped = 0
    loss_adjusted = 0.

    if checkpoint is not None:
        del checkpoint  # cleanup memory
        torch.cuda.empty_cache()

        # -------------- START TRAINING --------------
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, max_iters + 1)) if (main_node) else range(start_iter, max_iters + 1)  # <--- DDP
    model.train()
    for it in pbar:
        images, captions = next(dataloader_iterator)
        images = images.to(device)

        with torch.cuda.amp.autocast(dtype=_float16_dtype), torch.no_grad():
            if np.random.rand() < 0.05:  # 90% of the time, drop the CLIP text embeddings (indepentently)
                clip_captions = [''] * len(captions)  # 5% of the time drop all the captions
            else:
                clip_captions = captions
            clip_tokens = clip_tokenizer(clip_captions, truncation=True, padding="max_length",
                                         max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
            clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

            t = (1 - torch.rand(images.size(0), device=device)).mul(1.08).add(0.001).clamp(0.001, 1.0)
            latents = vqmodel.encode(images)[2]
            noised_latents, mask = model.module.generator.add_noise(latents, t)
            loss_weight = model.module.generator.get_loss_weight(t, mask)

            effnet_preproc = effnet_preprocess(images)

        with torch.cuda.amp.autocast(dtype=_float16_dtype):
            pred = model(noised_latents, t, effnet_preproc, clip_text_embeddings)
            loss = criterion(pred, latents)
            loss = ((loss * loss_weight).sum(dim=[1, 2]) / loss_weight.sum(dim=[1, 2])).mean()
            loss_adjusted = loss / grad_accum_steps

        acc = (pred.argmax(1) == latents).float()
        acc = acc.mean()
        if not torch.isnan(loss_adjusted):
            if it % grad_accum_steps == 0 or it == max_iters:
                loss_adjusted.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                with model.no_sync():
                    loss_adjusted.backward()
        else:
            print(f"Encountered NaN loss in iteration {it}.")
            skipped += 1

        if main_node:  # <--- DDP
            pbar.set_postfix({
                'bs': images.size(0),
                'loss': loss_adjusted.item(),
                'acc': acc.item(),
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
                'skipped': skipped,
            })
            wandb.log({
                'loss': loss_adjusted.item(),
                'acc': acc.item(),
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })

        if main_node and (it == 1 or it % print_every == 0 or it == max_iters):  # <--- DDP
            # if main_node:
            print(f"ITER {it}/{max_iters} - loss {loss_adjusted}")

            if it % extra_ckpt_every == 0:
                torch.save({
                    'state_dict': model.module.generator.state_dict(),
                    'effnet_state_dict': model.module.effnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_last_step': scheduler.last_epoch,
                    'iter': it,
                    'grad_scaler_state_dict': scaler.state_dict(),
                    'wandb_run_id': run_id,
                }, os.path.join(checkpoint_dir, run_name, f"model_{it}.pt"))
            torch.save({
                'state_dict': model.module.generator.state_dict(),
                'effnet_state_dict': model.module.effnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_last_step': scheduler.last_epoch,
                'iter': it,
                'grad_scaler_state_dict': scaler.state_dict(),
                'wandb_run_id': run_id,
            }, checkpoint_path)

            model.eval()
            images, captions = next(dataloader_iterator)
            while images.size(0) < 8:
                _images, _captions = next(dataloader_iterator)
                images = torch.cat([images, _images], dim=0)
                captions += _captions
            images, captions = images[:8].to(device), captions[:8]
            with torch.no_grad():
                # CLIP stuff
                clip_tokens = clip_tokenizer(captions, truncation=True, padding="max_length",
                                             max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

                clip_tokens_uncond = clip_tokenizer([""] * len(captions), truncation=True, padding="max_length",
                                                    max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(
                    device)
                clip_embeddings_uncond = clip_model(**clip_tokens_uncond).last_hidden_state
                # ---

                # Efficientnet stuff
                effnet_embeddings = model.module.effnet(effnet_preprocess(images))
                effnet_embeddings_uncond = torch.zeros_like(effnet_embeddings)
                # ---

                t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
                latents = vqmodel.encode(images)[2]
                noised_latents, mask = model.module.generator.add_noise(latents, t)
                pred = model.module.generator(noised_latents, t, effnet_embeddings, clip_text_embeddings)
                pred_tokens = pred.div(0.1).softmax(dim=1).permute(0, 2, 3, 1) @ vqmodel.vquantizer.codebook.weight.data
                pred_tokens = vqmodel.vquantizer.forward(pred_tokens, dim=-1)[-1]
                sampled = sample(model.module.generator, {'effnet': effnet_embeddings, 'byt5': clip_text_embeddings},
                                 (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                 unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                       'byt5': clip_embeddings_uncond})
                sampled_noimg = sample(model.module.generator,
                                       {'effnet': effnet_embeddings, 'byt5': clip_text_embeddings},
                                       (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                       unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                             'byt5': clip_embeddings_uncond})

                noised_images = vqmodel.decode_indices(noised_latents).clamp(0, 1)
                pred_images = vqmodel.decode_indices(pred_tokens).clamp(0, 1)
                sampled_images = vqmodel.decode_indices(sampled).clamp(0, 1)
                sampled_images_noimg = vqmodel.decode_indices(sampled_noimg).clamp(0, 1)
            model.train()

            torchvision.utils.save_image(torch.cat([
                torch.cat([i for i in images.cpu()], dim=-1),
                torch.cat([i for i in noised_images.cpu()], dim=-1),
                torch.cat([i for i in pred_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images_noimg.cpu()], dim=-1),
            ], dim=-2), f'{output_path}/{it:06d}.jpg')

            log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_noimg[i])] + [
                wandb.Image(images[i])] for i in range(len(images))]
            log_table = wandb.Table(data=log_data, columns=["Captions", "Sampled", "Sampled noimg", "Orig"])
            wandb.log({"Log": log_table})

    destroy_process_group()  # <--- DDP


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    n_node = 1
    mp.spawn(train, args=(world_size, n_node), nprocs=world_size)
