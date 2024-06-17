"""Module for training the Phonon DOS model with different embedders."""

import random
import time
from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from embedder_phDOS.DOSTransformer_phonon import DOSTransformer_phonon
from embedder_phDOS.graphnetwork_phonon import Graphnetwork_phonon, Graphnetwork2_phonon
from embedder_phDOS.mlp_phonon import mlp_phonon, mlp2_phonon
from embedder_phDOS.e3nn_phonon import e3nn_phonon, get_neighbors

from utils import (
    test_phonon1,
    test_phonon2,
    build_data,
    load_data,
    train_valid_test_split,
    r2,
)


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=str, default=".", metavar="", help="path to run directory"
    )
    parser.add_argument(
        "--r_max", type=float, default=4.0, metavar="", help="cutoff radius"
    )
    parser.add_argument(
        "--embedder", type=str, default="DOSTransformer", metavar="", help="embedder"
    )
    parser.add_argument("--device", type=int, default=0, metavar="", help="GPU to use")
    parser.add_argument(
        "--lr", type=float, default=0.0001, metavar="", help="learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, metavar="", help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, metavar="", help="batch size"
    )
    parser.add_argument(
        "--layers", type=int, default=3, metavar="", help="number of processor layers"
    )
    parser.add_argument(
        "--transformer",
        type=int,
        default=2,
        metavar="",
        help="number of transformer layers",
    )
    parser.add_argument(
        "--eval", type=int, default=5, metavar="", help="evaluation step"
    )
    parser.add_argument(
        "--es", type=int, default=200, metavar="", help="early stopping criteria"
    )
    parser.add_argument(
        "--hidden", type=int, default=256, metavar="", help="hidden dim"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        metavar="",
        help="random state for dataset split",
    )
    parser.add_argument(
        "--attn_drop",
        type=float,
        default=0.0,
        metavar="",
        help="attention dropout ratio",
    )
    parser.add_argument("--seed", type=int, default=0, metavar="", help="random seed")
    parser.add_argument(
        "--beta", type=float, default=1.0, metavar="", help="alpha for the spark loss2"
    )
    return parser.parse_args()


def setup_data(r_max):
    """Load and preprocess the dataset based on the given cutoff radius."""

    csv_file = Path(__file__).parent.joinpath("data/data.csv")
    df, species = load_data(f"{csv_file}")
    df["data"] = df.apply(lambda x: build_data(x, r_max), axis=1)  # type: ignore
    return df, species


def get_expt_info(args):
    """Get the experimental configuration."""

    train_config = {arg: getattr(args, arg) for arg in vars(args)}
    config = [
        "r_max",
        "embedder",
        "layers",
        "transformer",
        "attn_drop",
        "hidden",
        "beta",
        "lr",
        "epochs",
        "batch_size",
        "random_state",
        "seed",
    ]

    if args.embedder.lower() == "dostransformer":
        pass

    elif args.embedder.lower() in [
        "graphnetwork",
        "graphnetwork2",
        "mlp",
        "mlp2",
        "e3nn",
    ]:
        config.remove("transformer")
        config.remove("attn_drop")

    else:
        raise f"error occured : Inappropriate model name, `{args.embedder}`"  # type: ignore

    info = "Experimental Configuration\n--------------------------------\n" + "".join(
        f"{key}: {train_config[key]}\n" for key in config
    )

    info += "--------------------------------\n\n"
    return info


def main():  # sourcery skip: extract-duplicate-method  # sourcery skip: extract-duplicate-method
    """Train the model with the given embedder."""

    args = parse_args()

    torch.set_num_threads(2)  # limit CPU usage
    torch.set_default_dtype(torch.float64)  # Default data type float 64 for phonon DOS

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    expt_file = Path(args.run_dir).joinpath("expt.txt")

    # Backup the previous experimental configuration
    if expt_file.exists():
        backup_files = sorted(Path(args.run_dir).glob("bck_*_expt.txt"))
        idx = len(backup_files)
        backup_expt_file = Path(args.run_dir).joinpath(f"bck_{idx}_expt.txt")
        backup_expt_file.write_text(
            expt_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        expt_file.unlink()

    with open(f"{expt_file}", "w", encoding="utf-8") as fi:
        fi.write(get_expt_info(args))  # write experimental configuration

        # GPU setting
        device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        torch.cuda.set_device(device)
        fi.write(f"Device: {device}\n\n")

        df, species = setup_data(args.r_max)
        fi.write(f"Data pre-processed based on cutoff radius of {args.r_max} Å. \n\n")
        fi.flush()

        # Load dataset
        idx_train, idx_valid, idx_test = train_valid_test_split(
            df, species, valid_size=0.1, test_size=0.1, seed=args.random_state
        )

        fi.write(f"Total number of examples: {len(idx_train + idx_valid + idx_test)}\n")
        fi.write(f"train: {len(idx_train)}\n")
        fi.write(f"val: {len(idx_valid)}\n")
        fi.write(f"test: {len(idx_test)}\n\n")

        train_loader = DataLoader(
            df.iloc[idx_train]["data"].values,  # type: ignore
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            df.iloc[idx_valid]["data"].values, batch_size=args.batch_size  # type: ignore
        )
        test_loader = DataLoader(
            df.iloc[idx_test]["data"].values, batch_size=args.batch_size  # type: ignore
        )

        fi.write("Dataset Loaded!\n\n")
        fi.flush()

    embedder = args.embedder.lower()
    n_hidden = args.hidden
    attn_drop = args.attn_drop
    n_atom_feat = 118
    n_bond_feat = 4
    out_dim = len(df.iloc[0]["phfreq"])

    # Model selection
    if embedder == "dostransformer":
        model = DOSTransformer_phonon(
            args.layers,
            args.transformer,
            n_atom_feat,
            n_bond_feat,
            n_hidden,
            device,
            attn_drop,
        ).to(device)

    elif embedder == "graphnetwork":
        model = Graphnetwork_phonon(
            args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device
        ).to(device)

    elif embedder == "graphnetwork2":
        model = Graphnetwork2_phonon(
            args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device
        ).to(device)

    elif embedder == "mlp":
        model = mlp_phonon(
            args.layers, n_atom_feat, n_bond_feat, n_hidden, args.r_max, device
        ).to(device)

    elif embedder == "mlp2":
        model = mlp2_phonon(
            args.layers, n_atom_feat, n_bond_feat, n_hidden, args.r_max, device
        ).to(device)

    elif embedder == "e3nn":
        e3nn_n_bond_feat = 64
        model = e3nn_phonon(
            in_dim=n_atom_feat,  # dimension of one-hot encoding of atom type
            em_dim=e3nn_n_bond_feat,  # dimension of atom-type embedding
            irreps_in=str(e3nn_n_bond_feat)
            + "x0e",  # em_dim scalars (L=0 and even parity) on each atom to represent atom type
            irreps_out=str(out_dim)
            + "x0e",  # out_dim scalars (L=0 and even parity) to output
            irreps_node_attr=str(e3nn_n_bond_feat)
            + "x0e",  # em_dim scalars (L=0 and even parity) on each atom to represent atom type
            layers=args.layers,  # number of nonlinearities (number of convolutions = layers + 1)
            mul=32,  # multiplicity of irreducible representations
            lmax=1,  # maximum order of spherical harmonics
            max_radius=args.r_max,  # cutoff radius for convolution
            num_neighbors=get_neighbors(
                df, idx_train
            ).mean(),  # scaling factor based on the typical number of neighbors
            reduce_output=True,  # whether or not to aggregate features of all atoms at the end
        )
        # model.pool = True
        model.to(device)
    else:
        raise f"error occured : Inappropriate model name, `{embedder}`"  # type: ignore

    with open(f"{expt_file}", "a", encoding="utf-8") as fi:
        fi.write("Model Architecture:\n--------------------------------\n\n")
        fi.write(f"{model} \n--------------------------------\n\n")
        fi.write("Start training\n--------------------------------\n")
        fi.flush()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        criterion = nn.MSELoss()
        criterion_2 = nn.L1Loss()

        checkpoint_generator = iter(np.linspace(args.eval, args.epochs, 5, dtype=int))
        checkpoint = next(checkpoint_generator)

        start_time = time.time()

        if Path(args.run_dir).joinpath("checkpoints").exists():
            last_checkpoint = sorted(
                Path(args.run_dir).joinpath("checkpoints").iterdir()
            )[-1]

            results = torch.load(last_checkpoint)
            model.load_state_dict(results["state"])
            previous_epoch = results["epoch"]
            best_epoch = results["best_epoch"]
            test_rmse = results["test_rmse"]
            test_mse = results["test_mse"]
            test_mae = results["test_mae"]
            test_r2 = results["test_r2"]

            best_losses = results["best_losses"]
            best_rmse = test_rmse
            best_mae = test_mae

            fi.write(f"\nModel loaded from {last_checkpoint}\n")

        else:
            Path(args.run_dir).joinpath("checkpoints").mkdir(
                parents=True, exist_ok=False
            )
            previous_epoch = 0
            best_losses = []
            best_rmse = 1000
            best_mae = 1000

        if embedder == "dostransformer":

            for epoch in range(previous_epoch, args.epochs):
                model.train()

                for batch in train_loader:
                    batch.to(device)

                    preds_global, _, preds_system = model(  # pylint: disable=E1102
                        batch
                    )

                    mse_global = criterion(preds_global, batch.phdos).cpu()
                    rmse_global = torch.sqrt(mse_global).mean()

                    mse_system = criterion(preds_system, batch.phdos).cpu()
                    rmse_system = torch.sqrt(mse_system).mean()
                    loss = rmse_global + args.beta * rmse_system

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                wall = end_time - start_time

                if (epoch + 1) % args.eval == 0:

                    train_rmse, train_mse, train_mae, train_r2 = test_phonon1(
                        model, train_loader, criterion_2, r2, device
                    )

                    fi.write(
                        f"\n[ Epoch {epoch+1} ]: lr: {scheduler.get_last_lr()[0]:.2e} | "
                        + f"elapsed time: {time.strftime('%H:%M:%S', time.gmtime(wall))}"
                        + f"\n[ Epoch {epoch+1} ]: train_rmse: {train_rmse:.4f} | "
                        + f"train_mse: {train_mse:.4f} | train_mae: {train_mae:.4f} | "
                        + f"train_r2: {train_r2:.4f}"
                    )

                    valid_rmse, valid_mse, valid_mae, valid_r2 = test_phonon1(
                        model, valid_loader, criterion_2, r2, device
                    )
                    fi.write(
                        f"\n[ Epoch {epoch+1} ]: valid_rmse: {valid_rmse:.4f} | "
                        + f"valid_mse: {valid_mse:.4f} | valid_mae: {valid_mae:.4f} | "
                        + f"valid_r2: {valid_r2:.4f}"
                    )

                    if valid_rmse < best_rmse and valid_mae < best_mae:
                        best_rmse = valid_rmse
                        best_mae = valid_mae
                        best_epoch = epoch + 1

                        test_rmse, test_mse, test_mae, test_r2 = test_phonon1(
                            model, test_loader, criterion_2, r2, device
                        )
                        fi.write(
                            f"\n[ Epoch {epoch+1} ]: test_rmse: {test_rmse:.4f} | "
                            + f"test_mse: {test_mse:.4f} | test_mae: {test_mae:.4f} | "
                            + f"test_r2: {test_r2:.4f}"
                        )

                    if valid_rmse < best_rmse and valid_mae > best_mae:
                        best_rmse = valid_rmse
                        best_epoch = epoch + 1

                        test_rmse, test_mse, test_mae, test_r2 = test_phonon1(
                            model, test_loader, criterion_2, r2, device
                        )
                        fi.write(
                            f"\n[ Epoch {epoch+1} ]: test_rmse: {test_rmse:.4f} | "
                            + f"test_mse: {test_mse:.4f} | test_mae: {test_mae:.4f} | "
                            + f"test_r2: {test_r2:.4f}"
                        )

                    if valid_rmse > best_rmse and valid_mae < best_mae:
                        best_mae = valid_mae
                        best_epoch = epoch + 1

                        test_rmse, test_mse, test_mae, test_r2 = test_phonon1(
                            model, test_loader, criterion_2, r2, device
                        )
                        fi.write(
                            f"\n[ Epoch {epoch+1} ]: test_rmse: {test_rmse:.4f} | "
                            + f"test_mse: {test_mse:.4f} | test_mae: {test_mae:.4f} | "
                            + f"test_r2: {test_r2:.4f}"
                        )

                    best_losses.append(best_rmse)

                    fi.write(
                        f"\n**System [Best Epoch: {best_epoch}] Best RMSE: {test_rmse:.4f} | "
                        + f"Best MSE: {test_mse:.4f} | Best MAE: {test_mae:.4f}| "
                        + f"Best R2: {test_r2:.4f}**\n"
                    )

                    fi.flush()

                    if (
                        len(best_losses) > int(args.es / args.eval)
                        and best_losses[-1] == best_losses[-int(args.es / args.eval)]
                    ):
                        fi.write("\nEarly stop!!\n")
                        fi.flush()

                        with open(
                            f"checkpoints/epoch_{checkpoint:03d}.pt", "wb"
                        ) as f_pt:
                            results = {
                                "state": model.state_dict(),
                                "epoch": epoch + 1,
                                "best_epoch": best_epoch,
                                "test_rmse": test_rmse,
                                "test_mse": test_mse,
                                "test_mae": test_mae,
                                "test_r2": test_r2,
                                "best_losses": best_losses,
                            }
                            torch.save(results, f_pt)
                        break

                if epoch + 1 == checkpoint:
                    with open(f"checkpoints/epoch_{checkpoint:03d}.pt", "wb") as f_pt:
                        results = {
                            "state": model.state_dict(),
                            "epoch": epoch + 1,
                            "best_epoch": best_epoch,
                            "test_rmse": test_rmse,
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                            "test_r2": test_r2,
                            "best_losses": best_losses,
                        }
                        torch.save(results, f_pt)
                        if epoch + 1 < args.epochs:
                            checkpoint = next(checkpoint_generator)

                if scheduler is not None:
                    scheduler.step()

        else:
            for epoch in range(previous_epoch, args.epochs):
                model.train()

                for batch in train_loader:
                    batch.to(device)

                    dos = model(batch)  # pylint: disable=E1102
                    mse = criterion(dos, batch.phdos).cpu()
                    rmse = torch.sqrt(mse).mean()
                    loss = rmse

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                wall = end_time - start_time

                if (epoch + 1) % args.eval == 0:
                    train_rmse, train_mse, train_mae, train_r2 = test_phonon2(
                        model, train_loader, criterion, criterion_2, r2, device
                    )
                    fi.write(
                        f"\n[ Epoch {epoch+1} ]: lr: {scheduler.get_last_lr()[0]:.2e} | "
                        + f"elapsed time: {time.strftime('%H:%M:%S', time.gmtime(wall))}"
                        + f"\n[ Epoch {epoch+1} ]: train_rmse: {train_rmse:.4f} | "
                        + f"train_mse: {train_mse:.4f} | train_mae: {train_mae:.4f} | "
                        + f"train_r2: {train_r2:.4f}"
                    )

                    valid_rmse, valid_mse, valid_mae, valid_r2 = test_phonon2(
                        model, valid_loader, criterion, criterion_2, r2, device
                    )
                    fi.write(
                        f"\n[ Epoch {epoch+1} ]: valid_rmse: {valid_rmse:.4f} | "
                        + f"valid_mse: {valid_mse:.4f} | valid_mae: {valid_mae:.4f} | "
                        + f"valid_r2: {valid_r2:.4f}"
                    )

                    if valid_rmse < best_rmse and valid_mae < best_mae:
                        best_rmse = valid_rmse
                        best_mae = valid_mae
                        best_epoch = epoch + 1

                        test_rmse, test_mse, test_mae, test_r2 = test_phonon2(
                            model, test_loader, criterion, criterion_2, r2, device
                        )
                        fi.write(
                            f"\n[ Epoch {epoch+1} ]: test_rmse: {test_rmse:.4f} | "
                            + f"test_mse: {test_mse:.4f} | test_mae: {test_mae:.4f} | "
                            + f"test_r2: {test_r2:.4f}"
                        )

                    if valid_rmse < best_rmse and valid_mae > best_mae:
                        best_rmse = valid_rmse
                        best_epoch = epoch + 1

                        test_rmse, test_mse, test_mae, test_r2 = test_phonon2(
                            model, test_loader, criterion, criterion_2, r2, device
                        )
                        fi.write(
                            f"\n[ Epoch {epoch+1} ]: test_rmse: {test_rmse:.4f} | "
                            + f"test_mse: {test_mse:.4f} | test_mae: {test_mae:.4f} | "
                            + f"test_r2: {test_r2:.4f}"
                        )

                    if valid_rmse > best_rmse and valid_mae < best_mae:
                        best_mae = valid_mae
                        best_epoch = epoch + 1

                        test_rmse, test_mse, test_mae, test_r2 = test_phonon2(
                            model, test_loader, criterion, criterion_2, r2, device
                        )
                        fi.write(
                            f"\n[ Epoch {epoch+1} ]: test_rmse: {test_rmse:.4f} | "
                            + f"test_mse: {test_mse:.4f} | test_mae: {test_mae:.4f} | "
                            + f"test_r2: {test_r2:.4f}"
                        )

                    best_losses.append(best_rmse)

                    fi.write(
                        f"\n**System [Best Epoch: {best_epoch}] Best RMSE: {test_rmse:.4f} | "
                        + f"Best MSE: {test_mse:.4f} | Best MAE: {test_mae:.4f}| "
                        + f"Best R2: {test_r2:.4f}**\n"
                    )

                    fi.flush()

                    if (
                        len(best_losses) > int(args.es / args.eval)
                        and best_losses[-1] == best_losses[-int(args.es / args.eval)]
                    ):
                        fi.write("\nEarly stop!!\n")
                        fi.flush()

                        with open(
                            f"checkpoints/epoch_{checkpoint:03d}.pt", "wb"
                        ) as f_pt:
                            results = {
                                "state": model.state_dict(),
                                "epoch": epoch + 1,
                                "best_epoch": best_epoch,
                                "test_rmse": test_rmse,
                                "test_mse": test_mse,
                                "test_mae": test_mae,
                                "test_r2": test_r2,
                                "best_losses": best_losses,
                            }
                            torch.save(results, f_pt)
                        break

                if epoch + 1 == checkpoint:
                    with open(f"checkpoints/epoch_{checkpoint:03d}.pt", "wb") as f_pt:
                        results = {
                            "state": model.state_dict(),
                            "epoch": epoch + 1,
                            "best_epoch": best_epoch,
                            "test_rmse": test_rmse,
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                            "test_r2": test_r2,
                            "best_losses": best_losses,
                        }
                        torch.save(results, f_pt)
                        if epoch + 1 < args.epochs:
                            checkpoint = next(checkpoint_generator)

                if scheduler is not None:
                    scheduler.step()

        fi.write("\nTraining done!\n")
        fi.write(f"\nBest Epoch : {best_epoch} \n")
        fi.write(f"Best RMSE : {test_rmse:.4f} \n")
        fi.write(f"Best MSE : {test_mse:.4f} \n")
        fi.write(f"Best MAE : {test_mae:.4f} \n")
        fi.write(f"Best R2 : {test_r2:.4f} \n")
        end_time = time.time()
        total_time = end_time - start_time
        fi.write(
            f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n"
        )
        fi.flush()


if __name__ == "__main__":

    main()
