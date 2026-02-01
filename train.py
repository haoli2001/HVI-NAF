import os
import json
import torch
import random
import logging
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.hvi_dual_naf import HVIDualNAF
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *


MODEL_CONFIG = {
    "width": 32,
    "enc_blk_nums": [2, 2, 4, 8],
    "middle_blk_num": 12,
    "dec_blk_nums": [2, 2, 2, 2],
}


def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(run_dir):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def init_run(opt):
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{opt.dataset}_crop{opt.cropSize}_{run_stamp}"
    run_dir = os.path.join("logs", run_name)
    os.makedirs(run_dir, exist_ok=False)

    opt.run_dir = run_dir
    opt.weights_dir = os.path.join(run_dir, "weights")
    opt.metrics_dir = os.path.join(run_dir, "metrics")
    opt.val_folder = os.path.join(run_dir, "val") + os.sep

    os.makedirs(opt.weights_dir, exist_ok=True)
    os.makedirs(opt.metrics_dir, exist_ok=True)
    os.makedirs(opt.val_folder, exist_ok=True)

    params_path = os.path.join(run_dir, "params.json")
    with open(params_path, "w") as f:
        payload = {"args": vars(opt), "model": MODEL_CONFIG}
        json.dump(payload, f, indent=2)

    return run_dir


def train_init(opt):
    seed_torch()
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if opt.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


def train_epoch(model, training_data_loader, optimizer, opt, logger, epoch, L1_loss, P_loss, E_loss, D_loss):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iters = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)

    for batch in tqdm(training_data_loader):
        im1, im2 = batch[0], batch[1]
        im1 = im1.cuda()
        im2 = im2.cuda()

        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)
        else:
            output_rgb = model(im1)

        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)

        loss_hvi = (
            L1_loss(output_hvi, gt_hvi)
            + D_loss(output_hvi, gt_hvi)
            + E_loss(output_hvi, gt_hvi)
            + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        )
        loss_rgb = (
            L1_loss(output_rgb, gt_rgb)
            + D_loss(output_rgb, gt_rgb)
            + E_loss(output_rgb, gt_rgb)
            + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        )
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iters += 1

        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_print += loss.item()
        loss_last_10 += loss.item()
        pic_cnt += 1
        pic_last_10 += 1

        if iters == train_len:
            msg = (
                f"===> Epoch[{epoch}]: Loss: {loss_last_10 / pic_last_10:.4f} "
                f"|| Learning rate: lr={optimizer.param_groups[0]['lr']}."
            )
            logger.info(msg)
            loss_last_10 = 0
            pic_last_10 = 0

            output_img = transforms.ToPILImage()(output_rgb[0].squeeze(0))
            gt_img = transforms.ToPILImage()(gt_rgb[0].squeeze(0))
            training_dir = os.path.join(opt.val_folder, "training")
            os.makedirs(training_dir, exist_ok=True)
            output_img.save(os.path.join(training_dir, "test.png"))
            gt_img.save(os.path.join(training_dir, "gt.png"))

    return loss_print, pic_cnt


def checkpoint(model, opt, epoch, logger):
    train_dir = os.path.join(opt.weights_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    model_out_path = os.path.join(train_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    logger.info(f"Checkpoint saved to {model_out_path}")
    return model_out_path


def load_datasets(opt):
    logger = logging.getLogger("train")
    logger.info(f"===> Loading datasets: {opt.dataset}")

    if opt.dataset == "lol_v1":
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_v1)
    elif opt.dataset == "lol_blur":
        train_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)
    elif opt.dataset == "lolv2_real":
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_real)
    elif opt.dataset == "lolv2_syn":
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    elif opt.dataset == "SID":
        train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
    elif opt.dataset == "SICE_mix":
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
    elif opt.dataset == "SICE_grad":
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
    elif opt.dataset == "fivek":
        train_set = get_fivek_training_set(opt.data_train_fivek, size=opt.cropSize)
        test_set = get_fivek_eval_set(opt.data_val_fivek)
    else:
        raise Exception("should choose a dataset")

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=opt.shuffle,
    )
    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=1,
        shuffle=False,
    )
    return training_data_loader, testing_data_loader


def build_model(opt):
    logger = logging.getLogger("train")
    logger.info("===> Building model")
    model = HVIDualNAF(**MODEL_CONFIG).cuda()
    if opt.start_epoch > 0:
        pth = os.path.join(opt.weights_dir, "train", f"epoch_{opt.start_epoch}.pth")
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model


def make_scheduler(opt, model):
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[(opt.nEpochs // 4) - opt.warmup_epochs, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001],
            )
            scheduler = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001],
            )
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-7,
            )
            scheduler = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-7,
            )
    else:
        raise Exception("should choose a scheduler")
    return optimizer, scheduler


def init_loss(opt):
    L1_weight = opt.L1_weight
    D_weight = opt.D_weight
    E_weight = opt.E_weight
    P_weight = 1.0

    L1_loss = L1Loss(loss_weight=L1_weight, reduction="mean").cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss(
        {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
        perceptual_weight=P_weight,
        criterion="mse",
    ).cuda()
    return L1_loss, P_loss, E_loss, D_loss


if __name__ == "__main__":
    opt = option().parse_args()
    run_dir = init_run(opt)
    logger = setup_logger(run_dir)

    logger.info("===> Starting HVI-Dual-NAF training")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Model config: {MODEL_CONFIG}")

    train_init(opt)
    training_data_loader, testing_data_loader = load_datasets(opt)
    model = build_model(opt)
    optimizer, scheduler = make_scheduler(opt, model)
    L1_loss, P_loss, E_loss, D_loss = init_loss(opt)

    psnr = []
    ssim = []
    lpips = []
    start_epoch = opt.start_epoch if opt.start_epoch > 0 else 0

    for epoch in range(start_epoch + 1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train_epoch(
            model, training_data_loader, optimizer, opt, logger, epoch, L1_loss, P_loss, E_loss, D_loss
        )
        scheduler.step()

        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(model, opt, epoch, logger)
            norm_size = True

            if opt.dataset == "lol_v1":
                output_folder = "LOLv1/"
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == "lolv2_real":
                output_folder = "LOLv2_real/"
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == "lolv2_syn":
                output_folder = "LOLv2_syn/"
                label_dir = opt.data_valgt_lolv2_syn
            if opt.dataset == "lol_blur":
                output_folder = "LOL_blur/"
                label_dir = opt.data_valgt_lol_blur
            if opt.dataset == "SID":
                output_folder = "SID/"
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == "SICE_mix":
                output_folder = "SICE_mix/"
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.dataset == "SICE_grad":
                output_folder = "SICE_grad/"
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
            if opt.dataset == "fivek":
                output_folder = "fivek/"
                label_dir = opt.data_valgt_fivek
                norm_size = False

            im_dir = opt.val_folder + output_folder + "*.png"
            is_lol_v1 = opt.dataset == "lol_v1"
            is_lolv2_real = opt.dataset == "lolv2_real"
            eval(
                model,
                testing_data_loader,
                model_out_path,
                opt.val_folder + output_folder,
                norm_size=norm_size,
                LOL=is_lol_v1,
                v2=is_lolv2_real,
                alpha=0.8,
            )

            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            logger.info(f"===> Avg.PSNR: {avg_psnr:.4f} dB")
            logger.info(f"===> Avg.SSIM: {avg_ssim:.4f}")
            logger.info(f"===> Avg.LPIPS: {avg_lpips:.4f}")
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)

        torch.cuda.empty_cache()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    metrics_path = os.path.join(opt.metrics_dir, f"metrics_{now}.md")
    with open(metrics_path, "w") as f:
        f.write("dataset: " + opt.dataset + "\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")
        for i in range(len(psnr)):
            f.write(
                f"| {opt.start_epoch + (i + 1) * opt.snapshots} | {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n"
            )

    logger.info(f"Training finished. Metrics saved to {metrics_path}")
