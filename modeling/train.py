import argparse
import os
import numpy as np
import torch
import datetime
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import SP500SequenceDataset
from model import LinearSeqModel, TransformerEncoderModel

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train(args):
    # create sequential splits per-file to avoid leakage
    train_ds = SP500SequenceDataset(seq_len=args.seq_len, data_dir=args.data_dir, split="train", columns=args.train_cols)
    val_ds = SP500SequenceDataset(seq_len=args.seq_len, data_dir=args.data_dir, split="val", columns=args.train_cols)
    test_ds = SP500SequenceDataset(seq_len=args.seq_len, data_dir=args.data_dir, split="test", columns=args.train_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    input_seq_length = args.seq_len

    # # compute normalization stats from the training set (over all timesteps)
    # all_x = np.concatenate([s[0].reshape(-1, s[0].shape[-1]) for s in train_ds.samples], axis=0)
    # mean = all_x.mean(axis=0).astype(np.float32)
    # std = all_x.std(axis=0).astype(np.float32) + 1e-9

    # determine output dimension (number of target columns)
    output_dim = train_ds.output_dim

    # select model
    if args.model == "linear":
        model = LinearSeqModel(input_seq_length, latent_dim=args.latent_dim, output_dim=output_dim).to(device)
        out_name = "linear_seq_model.pth"
    elif args.model == "transformer":
        num_features = train_ds.samples[0][0].shape[-1]
        model = TransformerEncoderModel(
            seq_len=input_seq_length,
            num_features=num_features,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            output_dim=output_dim,
        ).to(device)
        out_name = "transformer_model.pth"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    mse_loss_fn = nn.MSELoss()
    cos_loss_fn = nn.CosineSimilarity(dim=1)

    best_val_loss = float("inf")
    patience = 0

    model_path = os.path.join(args.out_dir, out_name)
    ensure_dir(args.out_dir)
    with open(model_path.replace(".pth", "_config.txt"), "w") as f:
        f.write(f"Date and time of training: {datetime.datetime.now()}\n")
        f.write(f"Model type: {args.model}\n")
        f.write(f"Sequence length: {args.seq_len}\n")
        f.write(f"training hyperparameters: {args}\n")

        if args.model == "transformer":
            f.write(f"d_model: {args.d_model}\n")
            f.write(f"nhead: {args.nhead}\n")
            f.write(f"num_layers: {args.num_layers}\n")
            f.write(f"dim_feedforward: {args.dim_feedforward}\n")
            f.write(f"dropout: {args.dropout}\n")
        elif args.model == "linear":
            f.write(f"latent_dim: {args.latent_dim}\n")


    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse_loss = 0.0
        train_cos_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            mse_loss = mse_loss_fn(pred, yb)
            cos_loss = cos_loss_fn(pred, yb).mean()
            loss = mse_loss - cos_loss
            loss.backward()
            opt.step()
            if args.requires_scheduler:
                scheduler.step()
            train_mse_loss += mse_loss.item() * xb.size(0)
            train_cos_loss += cos_loss.item() * xb.size(0)
        train_mse_loss /= len(train_loader.dataset)
        train_cos_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = mse_loss_fn(pred, yb) - cos_loss_fn(pred, yb).mean()  # combine MSE with cosine similarity for better convergence
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs}  train_mse_loss={train_mse_loss:.6f}  train_cos_loss={train_cos_loss:.6f}  val_loss={val_loss:.6f}")


        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, model_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping due to no improvement in validation loss for {args.patience} epochs.")
                break
    
    # test set evaluation with best model
    model = torch.load(model_path, weights_only=False)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = mse_loss_fn(pred, yb) - cos_loss_fn(pred, yb).mean() 
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}")

    
    with open(model_path.replace(".pth", "_results_summary.txt"), "w") as f:
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Epochs trained: {epoch}\n")
    print(f"Saved model and config to {model_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/with_stats")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="modeling/artifacts")
    parser.add_argument("--model", type=str, choices=["linear", "transformer"], default="transformer", help="Which model to use from model.py")
    # linear model hyperparams
    parser.add_argument("--latent_dim", type=int, default=12)
    # transformer hyperparams
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10, help="Epochs to wait for improvement before early stopping")
    parser.add_argument("--scheduler_step", type=int, default=5, help="Epochs between learning rate decay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler_gamma", type=float, default=0.4, help="Learning rate decay factor for StepLR")
    parser.add_argument("--requires_scheduler", action="store_true", help="Whether to use a learning rate scheduler (StepLR) during training")
    parser.add_argument("--training_cols", type=str, default="open_change,high_change,low_change,close_change,exchange_portion", help="Optional list of columns to train on (defaults to all numeric columns)")
    args = parser.parse_args()
    args.train_cols = [col.strip() for col in args.training_cols.split(",")]
    print(f"Training with columns: {args.train_cols}")
    torch.manual_seed(args.seed)
    train(args)
