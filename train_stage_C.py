import os
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import numpy as np
import wandb
import shutil
from transformers import AutoTokenizer, CLIPTextModel
import webdataset as wds
from webdataset.handlers import warn_and_continue
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchtools.utils import Diffuzz
from vqgan import VQModel
from modules import Paella, sample, EfficientNetEncoder, Prior
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
batch_size = 20 * 8 * 8
grad_accum_steps = 1
max_iters = updates * grad_accum_steps
print_every = 2000 * grad_accum_steps
extra_ckpt_every = 10000 * grad_accum_steps
lr = 1e-4  # 1e-4

dataset_path = ""
run_name = "Würstchen-Prior-CLIP-Text-v2"
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
        init_method="file:///mnt/nvme/home/dome/src/würstchen/dist_file69",
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
    ).shuffle(44, handler=warn_and_continue).decode(
        "pilrgb", handler=warn_and_continue
    ).to_tuple(
        "jpg", "txt", handler=warn_and_continue
    ).map_tuple(
        transforms, identity, handler=warn_and_continue
    )

    real_batch_size = batch_size // (world_size * n_nodes * grad_accum_steps)
    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=False)

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
    if main_node:
        vqmodel = VQModel().to(device)
        vqmodel.load_state_dict(torch.load("models/vqgan_f4_v1_500k.pt", map_location=device)['state_dict'])
        vqmodel.eval().requires_grad_(False)

    diffuzz = Diffuzz(device=device)

    # - CLIP text encoder
    clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(
        device).eval().requires_grad_(False)
    clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # - EfficientNet -
    pretrained_checkpoint = torch.load("models/model_stage_b.pt", map_location=device)

    effnet = EfficientNetEncoder(effnet="efficientnet_v2_l").to(device)
    effnet.load_state_dict(pretrained_checkpoint['effnet_state_dict'])
    effnet.eval().requires_grad_(False)

    # - Paella Model as generator - 
    if main_node:
        generator = Paella(byt5_embd=1024).to(device)
        generator.load_state_dict(pretrained_checkpoint['state_dict'])
        generator.eval().requires_grad_(False)

    del pretrained_checkpoint

    # - Diffusive Imagination Combinatrainer, a.k.a. Risotto - 
    model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])

    if main_node:  # <--- DDP
        model_ema = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(device).eval().requires_grad_(False)

    # load checkpoints & prepare ddp
    if checkpoint is not None:
        if main_node:  # <--- DDP
            if 'ema_state_dict' in checkpoint:
                model_ema.load_state_dict(checkpoint['ema_state_dict'])
            else:
                model_ema.load_state_dict(model.state_dict())

    # - SETUP WANDB - 
    if main_node:  # <--- DDP
        run_id = checkpoint['wandb_run_id'] if checkpoint is not None else wandb.util.generate_id()
        wandb.init(project=wandv_project, name=wandb_run_name, entity=wandv_entity, id=run_id, resume="allow")

    model = DDP(model, device_ids=[gpu_id], output_device=device)  # <--- DDP

    if main_node:  # <--- DDP
        print("Num trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # eps=1e-4
    # optimizer = StableAdamW(model.parameters(), lr=lr) # eps=1e-4
    # optimizer = Lion(model.parameters(), lr=lr / 3) # eps=1e-4
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_updates)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

    ema_loss = None
    if checkpoint is not None:
        ema_loss = checkpoint['metrics']['ema_loss']

    if checkpoint is not None:
        del checkpoint  # cleanup memory
        torch.cuda.empty_cache()

        # -------------- START TRAINING --------------
    if main_node:
        print("Everything prepared, starting training now....")
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, max_iters + 1)) if (main_node) else range(start_iter, max_iters + 1)  # <--- DDP
    model.train()
    for it in pbar:
        bls = time.time()
        images, captions = next(dataloader_iterator)
        ble = time.time() - bls
        images = images.to(device)

        with torch.no_grad():
            effnet_features = effnet(effnet_preprocess(images))
            with torch.cuda.amp.autocast(dtype=_float16_dtype):
                if np.random.rand() < 0.05:  # 90% of the time, drop the CLIP text embeddings (independently)
                    clip_captions = [''] * len(captions)  # 5% of the time drop all the captions
                else:
                    clip_captions = captions
                clip_tokens = clip_tokenizer(clip_captions, truncation=True, padding="max_length", max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

            t = (1 - torch.rand(images.size(0), device=device)).mul(1.08).add(0.001).clamp(0.001, 1.0)
            noised_embeddings, noise = diffuzz.diffuse(effnet_features, t)

        with torch.cuda.amp.autocast(dtype=_float16_dtype):
            pred_noise = model(noised_embeddings, t, clip_text_embeddings)
            loss = nn.functional.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * diffuzz.p2_weight(t)).mean() / grad_accum_steps

        if it % grad_accum_steps == 0 or it == max_iters:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if main_node and (it % ema_every == 0 or it == max_iters):
                if it < ema_start:
                    model_ema.load_state_dict(model.module.state_dict())
                else:
                    model_ema.update_weights_ema(model.module, beta=ema_beta)
        else:
            with model.no_sync():
                loss_adjusted.backward()

        ema_loss = loss.mean().item() if ema_loss is None else ema_loss * 0.99 + loss.mean().item() * 0.01

        if main_node:
            pbar.set_postfix({
                'bs': images.size(0),
                'batch_loading': ble,
                'loss': loss.mean().item(),
                'loss_adjusted': loss_adjusted.item(),
                'ema_loss': ema_loss,
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })

        if main_node:
            wandb.log({
                'loss': loss.mean().item(),
                'loss_adjusted': loss_adjusted.item(),
                'ema_loss': ema_loss,
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })

        if main_node and (it == 1 or it % print_every == 0 or it == max_iters):  # <--- DDP
            tqdm.write(f"ITER {it}/{max_iters} - loss {ema_loss}")

            if it % extra_ckpt_every == 0:
                torch.save({
                    'state_dict': model.module.state_dict(),
                    'ema_state_dict': model_ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_last_step': scheduler.last_epoch,
                    'iter': it,
                    'metrics': {
                        'ema_loss': ema_loss,
                    },
                    'grad_scaler_state_dict': scaler.state_dict(),
                    'wandb_run_id': run_id,
                }, os.path.join(checkpoint_dir, run_name, f"model_{it}.pt"))

            torch.save({
                'state_dict': model.module.state_dict(),
                'ema_state_dict': model_ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_last_step': scheduler.last_epoch,
                'iter': it,
                'metrics': {
                    'ema_loss': ema_loss,
                },
                'grad_scaler_state_dict': scaler.state_dict(),
                'wandb_run_id': run_id,
            }, checkpoint_path)

            model.eval()
            images, captions = next(dataloader_iterator)
            images, captions = images.to(device), captions
            images = images[:10]
            captions = captions[:10]
            with torch.no_grad():
                clip_tokens = clip_tokenizer(captions, truncation=True, padding="max_length",
                                             max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

                clip_tokens_uncond = clip_tokenizer([''] * len(captions), truncation=True, padding="max_length",
                                                    max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(
                    device)
                clip_text_embeddings_uncond = clip_model(**clip_tokens_uncond).last_hidden_state

                t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
                effnet_features = effnet(effnet_preprocess(images))
                effnet_embeddings_uncond = torch.zeros_like(effnet_features)
                noised_embeddings, noise = diffuzz.diffuse(effnet_features, t)

                with torch.cuda.amp.autocast(dtype=_float16_dtype):
                    pred_noise = model(noised_embeddings, t, clip_text_embeddings)
                    pred = diffuzz.undiffuse(noised_embeddings, t, torch.zeros_like(t), pred_noise)
                    sampled = diffuzz.sample(model.module, {'c': clip_text_embeddings},
                                             unconditional_inputs={"c": clip_text_embeddings_uncond},
                                             shape=effnet_features.shape, cfg=6)[-1]
                    sampled_ema = diffuzz.sample(model_ema, {'c': clip_text_embeddings},
                                                 unconditional_inputs={"c": clip_text_embeddings_uncond},
                                                 shape=effnet_features.shape, cfg=6)[-1]

                    sampled_images = sample(generator, {'effnet': sampled_ema, 'byt5': clip_text_embeddings},
                                            (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                            unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                  'byt5': clip_text_embeddings_uncond})
                    sampled_images_ema = sample(generator, {'effnet': sampled, 'byt5': clip_text_embeddings}, (
                    clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                                unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                      'byt5': clip_text_embeddings_uncond})
                    sampled_images_original = sample(generator,
                                                     {'effnet': effnet_features, 'byt5': clip_text_embeddings}, (
                                                     clip_text_embeddings.size(0), images.size(-2) // 4,
                                                     images.size(-1) // 4),
                                                     unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                           'byt5': clip_text_embeddings_uncond})
                    sampled_pred = sample(generator, {'effnet': pred, 'byt5': clip_text_embeddings},
                                          (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                          unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                'byt5': clip_text_embeddings_uncond})
                    sampled_noised = sample(generator, {'effnet': noised_embeddings, 'byt5': clip_text_embeddings},
                                            (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                            unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                  'byt5': clip_text_embeddings_uncond})

                noised_images = vqmodel.decode_indices(sampled_noised).clamp(0, 1)
                pred_images = vqmodel.decode_indices(sampled_pred).clamp(0, 1)
                sampled_images_original = vqmodel.decode_indices(sampled_images_original).clamp(0, 1)
                sampled_images = vqmodel.decode_indices(sampled_images).clamp(0, 1)
                sampled_images_ema = vqmodel.decode_indices(sampled_images_ema).clamp(0, 1)
            model.train()

            torchvision.utils.save_image(torch.cat([
                torch.cat([i for i in images.cpu()], dim=-1),
                torch.cat([i for i in noised_images.cpu()], dim=-1),
                torch.cat([i for i in pred_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images_ema.cpu()], dim=-1),
                torch.cat([i for i in sampled_images_original.cpu()], dim=-1),
            ], dim=-2), f'{output_path}/{it:06d}.jpg')

            log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_ema[i])] + [
                wandb.Image(sampled_images_original[i])] + [wandb.Image(images[i])] for i in range(len(images))]
            log_table = wandb.Table(data=log_data,
                                    columns=["Captions", "Sampled", "Sampled EMA", "Sampled Original", "Orig"])
            wandb.log({"Log": log_table})
            del clip_tokens, clip_text_embeddings, clip_tokens_uncond, clip_text_embeddings_uncond, t, effnet_features, effnet_embeddings_uncond
            del noised_embeddings, noise, pred_noise, pred, sampled, sampled_ema, sampled_images, sampled_images_ema, sampled_images_original
            del sampled_pred, sampled_noised, noised_images, pred_images, log_data, log_table

    destroy_process_group()  # <--- DDP


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    n_node = 4
    mp.spawn(train, args=(world_size, n_node), nprocs=world_size)  # <--- DDP ;)
