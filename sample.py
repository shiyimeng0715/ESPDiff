import sys

import torch
from utils.train_utils import set_random_seed

from utils import init_env
import os
import argparse
import time
from pathlib import Path

from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from torch.utils.data import DataLoader
from utils.trainer import Trainer
from model.train_val_forward import modification_train_val_forward

#set_random_seed(7)


def get_loader(cfg):
    ORSSD_test_dataset = instantiate_from_config(cfg.test_dataset.ORSSD)
    EORSSD_test_dataset = instantiate_from_config(cfg.test_dataset.EORSSD)
    ors4199_test_dataset = instantiate_from_config(cfg.test_dataset.ors4199)

    EORSSD_test_loader = DataLoader(
        EORSSD_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    ors4199_test_loader = DataLoader(
        ors4199_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    ORSSD_test_loader = DataLoader(
        ORSSD_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return ORSSD_test_loader, EORSSD_test_loader, ors4199_test_loader


def measure(unwrapped, test_loader):
    """返回 (n_params, flops_1step, fps, infer_elapsed)。"""
    device = next(unwrapped.parameters()).device
    unwrapped.eval()

    n_params = sum(p.numel() for p in unwrapped.parameters())

    # FLOPs：单步去噪网络（backbone + decoder 一次 forward）
    try:
        from thop import profile
        net = unwrapped.model  # net_mobilevit
        x = torch.randn(1, 1, 352, 352, device=device)
        cond = torch.randn(1, 3, 352, 352, device=device)
        t = torch.zeros(1, device=device)
        macs, _ = profile(net, inputs=(x, t, cond, False), verbose=False)
        flops_1step = float(macs)
    except Exception as e:
        flops_1step = float('nan')
        print(f"[WARN] thop profile failed: {e}")

    # FPS：T 步完整采样，纯推理（不保存）
    set_random_seed(0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    n_img = 0
    with torch.inference_mode():
        for data in test_loader:
            image = data['image'].to(device).squeeze(1)
            _ = modification_train_val_forward(unwrapped, image=image, mode=False, verbose=False)
            n_img += image.size(0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    fps = n_img / elapsed if elapsed > 0 else float('nan')

    return n_params, flops_1step, fps, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/model-best.pt')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_sample_steps', type=int, default=50)
    parser.add_argument('--target_dataset', nargs='+', type=str, default=['EORSSD'])#ORSSD EORSSD ors4199
    #parser.add_argument('--time_ensemble', type=str, default='True')
    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')
    parser.add_argument('--measure', action='store_true')

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    """
        Hack config here.
    """
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    ORSSD_test_loader, EORSSD_test_loader, ors4199_test_loader = get_loader(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)

    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    ORSSD_test_loader, EORSSD_test_loader, ors4199_test_loader = \
        trainer.accelerator.prepare(ORSSD_test_loader, EORSSD_test_loader, ors4199_test_loader)

    dataset_map = {
        'ORSSD': ORSSD_test_loader,
        'EORSSD': EORSSD_test_loader,
        'ors4199': ors4199_test_loader,
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        mask_path = Path(cfg.test_dataset.ORSSD.params.image_root).parent.parent
        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)
        if cfg.batch_ensemble:
            print("batch")
            mae, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            print("time")
            mae, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            print("val")
            mae, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')

        if trainer.accelerator.is_main_process:
            print(mask_path)
            from utils.eval import eval

            eval_score = eval(
                mask_path=mask_path,
                pred_path=cfg.results_folder,
                dataset_name=dataset_name)

            if cfg.measure:
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                n_params, flops_1step, fps, elapsed = measure(unwrapped, dataset)
                T = cfg.diffusion_model.params.num_sample_steps
                print(f"[MEASURE] dataset={dataset_name} params={n_params / 1e6:.4f}M "
                      f"flops_1step={flops_1step / 1e9:.4f}G flops_T={flops_1step / 1e9 * T:.4f}G "
                      f"fps={fps:.2f} infer_time={elapsed:.2f}s")
        trainer.accelerator.wait_for_everyone()
