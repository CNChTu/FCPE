import argparse
import torch
from torch.optim import lr_scheduler
from transformer_f0.saver import utils
from transformer_f0.data_loaders import get_data_loaders
from transformer_f0.model import TransformerF0
from transformer_f0.model_with_bce import TransformerF0BCE
from transformer_f0.solver import train


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    # load model
    if args.model.type == 'TransformerF0':
        model = TransformerF0(
            input_channel=args.model.input_channel,
            out_dims=1,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
            use_siren=args.model.use_siren,
            loss_mse_scale=args.loss.loss_mse_scale,
            loss_l2_regularization=args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=args.loss.loss_grad1_mse_scale,
        )
    elif args.model.type == 'TransformerF0BCE':
        model = TransformerF0BCE(
            input_channel=args.model.input_channel,
            out_dims=args.model.out_dims,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
            use_siren=args.model.use_siren,
            use_full=args.model.use_full,
            loss_mse_scale=args.loss.loss_mse_scale,
            loss_l2_regularization=args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=args.loss.loss_grad1_mse_scale,
            f0_max=args.model.f0_max,
            f0_min=args.model.f0_min,
            confidence=args.model.confidence,
        )

    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")

    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step,
                                                                    0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma,
                                    last_epoch=initial_global_step - 2)

    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)

    # datas
    loader_train, loader_valid = get_data_loaders(args)

    # run
    train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_valid)
