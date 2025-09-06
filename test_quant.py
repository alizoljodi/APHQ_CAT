import os
import sys
import torch
from torch import nn
import numpy as np
import argparse
import importlib
import timm
import copy
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from huggingface_hub import login
#login("")  # replace with your Hugging Face access token

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.mlp_recon import MLPReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import *
from datetime import datetime
import logging

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
        root_path = './checkpoint/quant_result/{}'.format(formatted_timestamp)
        os.makedirs(root_path)
        break
    except FileExistsError:
        time.sleep(10)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('{}/output.log'.format(root_path)),
                        logging.StreamHandler()
                    ])


import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="deit_tiny",
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
                        help="model")
    parser.add_argument('--config', type=str, default="./configs/4bit/best.py",
                        help="File path to import Config class from")
    parser.add_argument('--dataset', default="/home/alz07xz/imagenet",
                        help='path to dataset')
    parser.add_argument("--calib-size", default=argparse.SUPPRESS,
                        type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batch-size", default=500,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")

    parser.add_argument('--reconstruct-mlp', action='store_true',default=False, help='reconstruct mlp with ReLU function.')
    parser.add_argument('--load-reconstruct-checkpoint', type=str, default=False, help='Path to the reconstructed checkpoint.')
    parser.add_argument('--test-reconstruct-checkpoint', action='store_true', help='validate the reconstructed checkpoint.')
    
    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true',default=False, help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None, help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true',default=False, help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default="", help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')

    parser.add_argument("--print-freq", default=10,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=3407, type=int, help="seed")
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    parser.add_argument("--recon-metric", type=str, default=argparse.SUPPRESS, choices=['hessian_perturb', 'mse', 'mae'], 
                        help='mlp reconstruction metric')
    parser.add_argument("--calib-metric", type=str, default=argparse.SUPPRESS, choices=['mse', 'mae'], 
                        help='calibration metric')
    parser.add_argument("--optim-metric", type=str, default=argparse.SUPPRESS, choices=['hessian', 'hessian_perturb', 'mse', 'mae'], 
                        help='optimization metric')
    parser.add_argument('--optim-mode', type=str, default=argparse.SUPPRESS, choices=['qinp', 'rinp', 'qdrop'], 
                        help='`qinp`: use quanted input; `rinp`: use raw input; `qdrop`: use qdrop input.')
    parser.add_argument('--drop-prob', type=float, default=argparse.SUPPRESS, 
                        help='dropping rate in qdrop. set `drop-prob = 1.0` if do not use qdrop.')
    parser.add_argument('--pct', type=float, default=argparse.SUPPRESS, help='clamp percentile of mlp.fc2.')
    
    # Cluster affine correction parameters
    parser.add_argument('--alpha-list', type=float, nargs='+', default=[0.5], 
                        help='List of alpha values for blending (e.g., --alpha-list 0.3 0.5 0.7)')
    parser.add_argument('--num-clusters-list', type=int, nargs='+', default=[64], 
                        help='List of cluster numbers (e.g., --num-clusters-list 32 64 128)')
    parser.add_argument('--pca-dim-list', type=int, nargs='+', default=[50], 
                        help='List of PCA dimensions (e.g., --pca-dim-list 25 50 100)')
    
    return parser


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_calibsize_{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.calib_size, cfg.calib_metric)
    else:
        auto_name = '{}_w{}_a{}_optimsize_{}_{}_{}{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.optim_size, cfg.optim_metric, cfg.optim_mode, '_recon' if args.reconstruct_mlp else '')
    save_path = os.path.join(root_path, auto_name)

    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)


def load_model(model, args, device, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    ckpt = torch.load(ckpt_path)
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
                ckpt_name = name + '.' + attr + '.scale'
                getattr(module, attr).scale.data = ckpt[ckpt_name].clone()
 
    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model

    
def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    logging.info("Successfully imported Config class!")
        
    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.recon_metric = args.recon_metric if hasattr(args, 'recon_metric') else cfg.recon_metric
    cfg.calib_metric = args.calib_metric if hasattr(args, 'calib_metric') else cfg.calib_metric
    cfg.optim_metric = args.optim_metric if hasattr(args, 'optim_metric') else cfg.optim_metric
    cfg.optim_mode = args.optim_mode if hasattr(args, 'optim_mode') else cfg.optim_mode
    cfg.drop_prob = args.drop_prob if hasattr(args, 'drop_prob') else cfg.drop_prob
    cfg.reconstruct_mlp = args.reconstruct_mlp
    cfg.pct = args.pct if hasattr(args, 'pct') else cfg.pct
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")
        
    if args.device.startswith('cuda:'):
        gpu_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        args.device = 'cuda:0'
    device = torch.device(args.device)
    
    model_zoo = {
        'vit_tiny'  : 'vit_tiny_patch16_224',
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'vit_large' : 'vit_large_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
        'swin_base_384': 'swin_base_patch4_window12_384',
    }

    seed_all(args.seed)
    
    logging.info('Building model ...')
    try:
        model = timm.create_model(model_zoo[args.model], checkpoint_path='./checkpoint/vit_raw/{}.bin'.format(model_zoo[args.model]))
    except:
        model = timm.create_model(model_zoo[args.model], pretrained=True)
    full_model = copy.deepcopy(model)
    full_model.to(device)
    full_model.eval()
    
    model.to(device)
    model.eval()
    data_path = args.dataset
    g = mydatasets.ViTImageNetLoaderGenerator(data_path, args.val_batch_size, args.num_workers, kwargs={"model":model})
    
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    train_loader=g.train_loader()
    
    criterion = nn.CrossEntropyLoss().to(device)
    logging.info('Validating on test set on fp ...')
    val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    if args.reconstruct_mlp:
        for name, module in model.named_modules():
            if name.split('.')[-1] == 'mlp':
                module.act = nn.ReLU()
        if args.load_reconstruct_checkpoint is not None:
            logging.info(f"Restoring checkpoint from '{args.load_reconstruct_checkpoint}'")
            ckpt = torch.load(args.load_reconstruct_checkpoint)
            result = model.load_state_dict(ckpt, strict=False)
            logging.info(str(result))
            model.to(device)
            model.eval()
            if args.test_reconstruct_checkpoint:
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        elif args.load_calibrate_checkpoint is None:
            logging.info('Building calibrator ...')
            calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)

            logging.info('{} - Start reconstructing MLP blocks ...'.format(get_cur_time()))
            mlp_reconstructor = MLPReconstructor(model, full_model, calib_loader, metric=cfg.recon_metric, temp=cfg.temp)
            mlp_reconstructor.reconstruct_model(pct=cfg.pct)
            logging.info("{} - MLP reconstruction finished.".format(get_cur_time()))
            
            save_path = os.path.join(root_path, '{}_reconstructed.pth'.format(args.model))
            logging.info(f"Saving checkpoint to {save_path}")
            torch.save(model.state_dict(), save_path)
            logging.info('Validating after model reconstruction ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)

    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info('Wraping quantiztion modules (reparam: {}, recon: {}) ...'.format(reparam, args.reconstruct_mlp))
    model = wrap_modules_in_net(model, cfg, reparam=reparam, recon=args.reconstruct_mlp)
    model.to(device)
    model.eval()
    
    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, device, mode='calibrate')
            if args.test_calibrate_checkpoint:
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        else:
            logging.info("{} - start {} guided calibration".format(get_cur_time(), cfg.calib_metric))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
            quant_calibrator = QuantCalibrator(model, calib_loader)
            quant_calibrator.batching_quant_calib()
            model = wrap_reparamed_modules_in_net(model)
            model.to(device)
            logging.info("{} - {} guided calibration finished.".format(get_cur_time(), cfg.calib_metric))
            save_model(model, args, cfg, mode='calibrate')
            logging.info('Validating after calibration ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    
    if args.optimize:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        logging.info("{} - start {} guided block reconstruction".format(get_cur_time(), cfg.optim_metric))
        block_reconstructor = BlockReconstructor(model, full_model, calib_loader, metric=cfg.optim_metric, temp=cfg.temp, use_mean_hessian=cfg.use_mean_hessian)
        block_reconstructor.reconstruct_model(quant_act=True, mode=cfg.optim_mode, drop_prob=cfg.drop_prob, keep_gpu=cfg.keep_gpu)
        logging.info("{} - {} guided block reconstruction finished.".format(get_cur_time(), cfg.optim_metric))
        save_model(model, args, cfg, mode='optimize')
    if args.load_optimize_checkpoint:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        model = load_model(model, args, device, mode='optimize')
    if args.optimize or args.test_optimize_checkpoint:
        logging.info('Validating on calibration set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(calib_loader, model, criterion, print_freq=args.print_freq, device=device)
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    logging.info("{} - finished the process.".format(get_cur_time()))
    def extract_model_logits(q_model, fp_model, dataloader, device):
        """
        Extract logits from both quantized and full-precision models.
        Returns concatenated logits tensors.
        """
        q_model.eval()
        fp_model.eval()

        all_q, all_fp = [], []

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i>=10:
                    break
                images = images.to(device)
                q_logits = q_model(images)
                fp_logits = fp_model(images)
                all_q.append(q_logits.cpu())
                all_fp.append(fp_logits.cpu())

        all_q = torch.cat(all_q, dim=0)  # [N, C]
        all_fp = torch.cat(all_fp, dim=0)  # [N, C]
        
        return all_q, all_fp

    def build_cluster_affine(all_q, all_fp, num_clusters=64, pca_dim=None):
        """
        Build cluster affine correction model from pre-extracted logits.
        """
        # Optional PCA for clustering only
        pca = None
        if pca_dim is not None and pca_dim < all_q.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=42)
            q_features = pca.fit_transform(all_q.numpy())
        else:
            q_features = all_q.numpy()

        # Cluster quantized outputs
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_ids = cluster_model.fit_predict(q_features)

        # For each cluster: learn gamma, beta (per-class)
        gamma_dict = {}
        beta_dict = {}

        for cid in range(num_clusters):
            idxs = (cluster_ids == cid)
            if idxs.sum() == 0:
                # Empty cluster, default to identity
                gamma_dict[cid] = torch.ones(all_q.shape[1])
                beta_dict[cid] = torch.zeros(all_q.shape[1])
                continue

            q_c = all_q[idxs]  # [Nc, C]
            fp_c = all_fp[idxs]  # [Nc, C]

            # Closed-form least squares: fp ≈ gamma * q + beta
            mean_q = q_c.mean(dim=0)
            mean_fp = fp_c.mean(dim=0)

            # Compute variance, avoid div by zero
            var_q = q_c.var(dim=0, unbiased=False)
            var_q[var_q < 1e-8] = 1e-8

            gamma = ((q_c - mean_q) * (fp_c - mean_fp)).mean(dim=0) / var_q
            beta = mean_fp - gamma * mean_q

            gamma_dict[cid] = gamma
            beta_dict[cid] = beta

        return cluster_model, gamma_dict, beta_dict, pca

    def apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=None, alpha=0.4):
        """
        Apply per-cluster affine correction with optional PCA and alpha blending.
        """
        q_np = q_logits.cpu().numpy()

        # Apply same PCA as used during LUT building
        if pca is not None:
            q_np = pca.transform(q_np)

        cluster_ids = cluster_model.predict(q_np)

        corrected = []
        for i, q in enumerate(q_logits):
            cid = int(cluster_ids[i])
            gamma = gamma_dict[cid].to(q.device)
            beta = beta_dict[cid].to(q.device)
            affine_corrected = q * gamma + beta
            blended = q + alpha * (affine_corrected - q)
            corrected.append(blended)
        return torch.stack(corrected)
    
    def evaluate_cluster_affine_with_alpha(q_model, fp_model, cluster_model, gamma_dict, beta_dict, dataloader, device, pca=None, alpha=0.4):
        q_model.eval()
        fp_model.eval()
        total_top1, total_top5, total = 0, 0, 0
        
        # Store logits for plotting
        all_q_logits = []
        all_fp_logits = []
        all_corrected_logits = []
        all_cluster_ids = []

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                q_logits = q_model(images)
                fp_logits = fp_model(images)

                corrected_logits = apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=pca, alpha=alpha)

                # Store logits for plotting
                all_q_logits.append(q_logits.cpu())
                all_fp_logits.append(fp_logits.cpu())
                all_corrected_logits.append(corrected_logits.cpu())
                
                # Get cluster IDs for this batch
                q_np = q_logits.cpu().numpy()
                if pca is not None:
                    q_np = pca.transform(q_np)
                cluster_ids = cluster_model.predict(q_np)
                all_cluster_ids.append(cluster_ids)

                acc1, acc5 = accuracy(corrected_logits, targets, topk=(1, 5))
                total_top1 += acc1.item() * images.size(0)
                total_top5 += acc5.item() * images.size(0)
                total += images.size(0)

        print(f"[Alpha={alpha:.2f}] Top-1 Accuracy: {total_top1 / total:.2f}%")
        print(f"[Alpha={alpha:.2f}] Top-5 Accuracy: {total_top5 / total:.2f}%")
        
        # Plot randomly selected values from each cluster
        return total_top1 / total, total_top5 / total
        
    
    print("Extracting logits from quantized and full-precision models...")
    all_q, all_fp = extract_model_logits(model, full_model, train_loader, "cuda")
    
    # Save initial extracted logits for all models
    #initial_results_dir = f"initial_logits_{args.arch}_w{args.n_bits_w}bit_a{args.n_bits_a}bit_seed{args.seed}"
    #os.makedirs(initial_results_dir, exist_ok=True)
    
    # Convert to list format for the save function
    all_q_list = [all_q]
    all_fp_list = [all_fp]
    all_corrected_list = [all_q]  # Use quantized as placeholder for corrected
    
    
    # Use command-line arguments for parameter lists
    alpha_list = args.alpha_list
    num_clusters_list = args.num_clusters_list
    pca_dim_list = args.pca_dim_list
    
    print(f"Testing combinations:")
    print(f"  Alpha values: {alpha_list}")
    print(f"  Cluster numbers: {num_clusters_list}")
    print(f"  PCA dimensions: {pca_dim_list}")
    
    # Store results
    results = []
    
    # Loop through all parameter combinations
    for alpha in alpha_list:
        for num_clusters in num_clusters_list:
            for pca_dim in pca_dim_list:
                print(f"\n{'='*60}")
                print(f"Testing: alpha={alpha}, clusters={num_clusters}, pca_dim={pca_dim}")
                print(f"{'='*60}")
                
                # Build cluster affine model
                cluster_model, gamma_dict, beta_dict, pca = build_cluster_affine(
                    all_q, all_fp, num_clusters=num_clusters, pca_dim=pca_dim
                )
                
                # Evaluate with current parameters
                top1_acc, top5_acc = evaluate_cluster_affine_with_alpha(
                    model, full_model, cluster_model, gamma_dict, beta_dict, val_loader, "cuda", 
                    pca=pca, alpha=alpha
                )
                
                # Store results
                result = {
                    'alpha': alpha,
                    'num_clusters': num_clusters,
                    'pca_dim': pca_dim,
                    'top1_accuracy': top1_acc,
                    'top5_accuracy': top5_acc
                }
                results.append(result)
                
                print(f"Result: Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")
    
    # Print summary of all results
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*80}")
    print(f"{'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Top-1':<10} {'Top-5':<10}")
    print(f"{'-'*50}")
    
    for result in results:
        print(f"{result['alpha']:<8.2f} {result['num_clusters']:<10} {result['pca_dim']:<10} "
              f"{result['top1_accuracy']:<10.2f} {result['top5_accuracy']:<10.2f}")
    
    # Find best result
    best_result = max(results, key=lambda x: x['top1_accuracy'])
    print(f"\nBEST RESULT:")
    print(f"  Alpha: {best_result['alpha']}")
    print(f"  Clusters: {best_result['num_clusters']}")
    print(f"  PCA_dim: {best_result['pca_dim']}")
    print(f"  Top-1 Accuracy: {best_result['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {best_result['top5_accuracy']:.2f}%")
    
    # Create summary CSV of all saved logits files
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    