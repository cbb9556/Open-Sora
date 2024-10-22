import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from opensora.utils.lr_scheduler import LinearWarmupLR
from opensora.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    record_time = cfg.get("record_time", False)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    # 根据当前进程的排名设置CUDA设备
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # 设置随机种子以保证实验的可重复性
    set_seed(cfg.get("seed", 1024))

    # 初始化分布式协调器，用于管理分布式训练或推理过程
    coordinator = DistCoordinator()

    # 获取当前设备信息，用于后续的操作或计算
    device = get_current_device()

    # == init exp_dir ==
    # 定义实验的工作空间，初始化实验名称和目录
    exp_name, exp_dir = define_experiment_workspace(cfg)

    # 阻塞所有工作线程，确保在主协调器进行下一步操作时，所有工作线程都处于等待状态
    coordinator.block_all()

    # 如果当前进程为主进程，则执行目录创建和配置保存操作
    if coordinator.is_master():
        # 确保实验目录存在，如果不存在则创建，exist_ok=True表示如果目录已存在则不抛出异常
        os.makedirs(exp_dir, exist_ok=True)

        # 将训练配置保存到实验目录下，确保配置的可追溯性和可复现性
        save_training_config(cfg.to_dict(), exp_dir)

    # 再次阻塞所有工作线程，确保所有进程都同步到同一状态
    coordinator.block_all()

    # == init logger, tensorboard & wandb ==
    # 创建一个日志记录器，用于记录实验过程中的信息
    logger = create_logger(exp_dir)
    # 记录实验目录的创建位置
    logger.info("Experiment directory created at %s", exp_dir)
    # 记录训练配置信息，以便于调试和复现
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    # 如果当前进程是主进程，则创建TensorBoard写入器，用于记录训练过程中的度量信息
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        # 如果配置中启用了wandb，则初始化wandb，用于实验跟踪和可视化
        if cfg.get("wandb", False):
            wandb.init(project="Open-Sora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")

    # == init ColossalAI booster ==
    # 创建ColossalAI插件实例
    # 参数:
    #   plugin: 指定插件类型，默认为'zero2'
    #   dtype: 指定数据类型
    #   grad_clip: 梯度裁剪阈值，默认为0，表示不进行裁剪
    #   sp_size: 并行切分的大小，默认为1，表示不进行切分
    #   reduce_bucket_size_in_m: 梯度规约时的bucket大小，以MB为单位，默认为20MB
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
    )

    # 初始化Booster对象，用于管理和执行深度学习模型的训练过程
    # 参数:
    #   plugin: 指定使用的ColossalAI插件实例
    booster = Booster(plugin=plugin)

    # 设置PyTorch的线程数为1，以减少多线程带来的开销
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    # dataloader 将数据转换为，模型可以处理的对象
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
    if vae is not None:
        input_size = (dataset.num_frames, *dataset.image_size)
        latent_size = vae.get_latent_size(input_size)
        vae_out_channels = vae.out_channels
    else:
        latent_size = (None, None, None)
        vae_out_channels = cfg.get("vae_out_channels", 4)

    # == build diffusion model ==
    # 构建并配置模型
    model = (
        build_module(
            cfg.model,  # 使用配置文件中的模型配置
            MODELS,
            input_size=latent_size,  # 输入大小
            in_channels=vae_out_channels,  # 输入通道数
            caption_channels=text_encoder_output_dim,  # 字幕通道数，即文本编码器的输出维度
            model_max_length=text_encoder_model_max_length,  # 模型最大长度，用于文本编码器
            enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,  # 是否启用序列并行ism，将一个长序列划分为多个子序列，分配到不同的计算单元（如 GPU 中的不同处理核心）上同时进行处理。
        )
        .to(device, dtype)  # 将模型移动到指定设备和数据类型
        .train()  # 将模型设置为训练模式
    )

    # 获取模型的总参数量和可训练参数量
    model_numel, model_numel_trainable = get_model_numel(model)

    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    # 创建模型的深度拷贝，并将其转换为浮点类型，然后移动到指定设备上
    # 这是为了在不改变原始模型的情况下，创建一个用于指数移动平均（EMA）的模型副本
    # EMA 更新模型参数可使模型更快靠近最优解，减少参数振荡。
    ema = deepcopy(model).to(torch.float32).to(device)

    # 禁用EMA模型的梯度计算，因为EMA模型不需要进行反向传播和参数更新
    requires_grad(ema, False)

    # 记录EMA模型中每个参数的形状信息，这可能用于后续的参数更新或检查
    ema_shape_dict = record_model_param_shape(ema)

    # 将EMA模型设置为评估模式，这是因为在训练过程中，EMA模型仅用于参数的指数移动平均计算，不参与实际的训练
    ema.eval()

    # 使用当前模型的参数更新EMA模型的参数，初次调用时decay参数通常设置为0，以便直接复制当前模型的参数到EMA模型
    update_ema(ema, model, decay=0, sharded=False)

    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    # 初始化优化器HybridAdam
    # 使用filter过滤出模型中需要梯度更新的参数
    # adamw_mode设置为True，启用AdamW模式
    # 从配置字典cfg中获取学习率lr，如果未指定，则默认为1e-4
    # 从配置字典cfg中获取权重衰减weight_decay，如果未指定，则默认为0
    # 从配置字典cfg中获取Adam优化器中的eps值，如果未指定，则默认为1e-8
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    # 从配置字典cfg中获取warmup_steps值，如果没有指定，则默认为None
    warmup_steps = cfg.get("warmup_steps", None)

    # 根据warmup_steps的值决定是否使用学习率预热策略
    if warmup_steps is None:
        # 如果没有指定warmup_steps，则lr_scheduler设置为None
        lr_scheduler = None
    else:
        # 如果指定了warmup_steps，则创建LinearWarmupLR学习率调度器
        # 使用之前初始化的优化器optimizer
        # warmup_steps参数从配置字典cfg中获取
        # 优化器：对梯度下降方向和方式进行控制； 学习率调度和预热：对优化器的学习率进行控制，防止梯度消失和梯度爆炸
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype;
    # then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    model_sharding(ema)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    timers = {}
    timer_keys = [
        "move_data",
        "encode",
        "mask",
        "diffusion",
        "backward",
        "update_ema",
        "reduce_loss",
    ]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=coordinator)
        else:
            timers[key] = nullcontext()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                timer_list = []
                with timers["move_data"] as move_data_t:
                    x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                    y = batch.pop("text")
                if record_time:
                    timer_list.append(move_data_t)

                # == visual and text encoding ==
                with timers["encode"] as encode_t:
                    with torch.no_grad():
                        # Prepare visual inputs
                        if cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                        else:
                            x = vae.encode(x)  # [B, C, T, H/P, W/P]
                        # Prepare text inputs
                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            model_args = text_encoder.encode(y)
                if record_time:
                    timer_list.append(encode_t)

                # == mask ==
                with timers["mask"] as mask_t:
                    mask = None
                    if cfg.get("mask_ratios", None) is not None:
                        mask = mask_generator.get_masks(x)
                        model_args["x_mask"] = mask
                if record_time:
                    timer_list.append(mask_t)

                # == video meta info ==
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_args[k] = v.to(device, dtype)

                # == diffusion loss computation ==
                with timers["diffusion"] as loss_t:
                    loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)
                if record_time:
                    timer_list.append(loss_t)

                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    booster.backward(loss=loss, optimizer=optimizer)
                    optimizer.step()
                    optimizer.zero_grad()

                    # update learning rate
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                # == update EMA ==
                with timers["update_ema"] as ema_t:
                    update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    # wandb
                    if cfg.get("wandb", False):
                        wandb_dict = {
                            "iter": global_step,
                            "acc_step": acc_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        if record_time:
                            wandb_dict.update(
                                {
                                    "debug/move_data_time": move_data_t.elapsed_time,
                                    "debug/encode_time": encode_t.elapsed_time,
                                    "debug/mask_time": mask_t.elapsed_time,
                                    "debug/diffusion_time": loss_t.elapsed_time,
                                    "debug/backward_time": backward_t.elapsed_time,
                                    "debug/update_ema_time": ema_t.elapsed_time,
                                    "debug/reduce_loss_time": reduce_loss_t.elapsed_time,
                                }
                            )
                        wandb.log(wandb_dict, step=global_step)

                    running_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    if dist.get_rank() == 0:
                        model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )
                if record_time:
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    print(log_str)

        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()
