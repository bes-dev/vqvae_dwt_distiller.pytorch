import argparse
import os
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from core.utils import load_cfg, load_weights
from core.distiller import Distiller

def build_logger(cfg):
    return getattr(pl_loggers, cfg.type)(
        **cfg.params
    )

def main(args):
    cfg = load_cfg(args.cfg)
    distiller = Distiller(cfg)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        load_weights(distiller, ckpt["state_dict"])
    logger = build_logger(cfg.logger)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.getcwd() if args.checkpoint_dir is None else args.checkpoint_dir,
        save_top_k=True,
        save_last=True,
        verbose=True,
        monitor=cfg.trainer.monitor,
        mode=cfg.trainer.monitor_mode
    )
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=args.grad_batches,
        distributed_backend=args.distributed_backend,
        val_check_interval=args.val_check_interval,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(distiller)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--gpus", type=int, default=0, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default="ddp", choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to checkpoint_dir")
    parser.add_argument("--val-check-interval", type=int, default=500, help="validation check interval")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--cfg", type=str, help="path to config file")
    args = parser.parse_args()
    main(args)
