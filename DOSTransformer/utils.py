"""Module containing utility functions for the different model architectures."""

import math
import numpy as np
import pandas as pd
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
from torch_geometric.data import Data


def r2(x1, x2):
    """Compute the R^2 score between two tensors."""
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()
    return r2_score(x1.flatten(), x2.flatten(), multioutput="variance_weighted")


def test_phonon1(model, data_loader, criterion, r2, device):
    """Test function for DOS Transformer model architecture."""

    model.eval()

    with torch.no_grad():
        loss_rmse_sys, loss_mse_sys, loss_mae_sys, loss_r2_sys = 0, 0, 0, 0
        # sourcery skip: no-loop-in-tests
        for bc, batch in enumerate(data_loader):
            batch.to(device)

            preds_global, _, preds_system = model(batch)

            y = batch.phdos.reshape(preds_global.shape[0], -1)

            mse_sys = ((y - preds_system) ** 2).mean(dim=1)
            rmse_sys = torch.sqrt(mse_sys)

            loss_mse_sys += mse_sys.mean()
            loss_rmse_sys += rmse_sys.mean()

            mae_sys = criterion(preds_system, y).cpu()
            loss_mae_sys += mae_sys

            r2_score_sys = r2(y, preds_system)
            loss_r2_sys += r2_score_sys

    return (
        loss_rmse_sys / (bc + 1),
        loss_mse_sys / (bc + 1),
        loss_mae_sys / (bc + 1),
        loss_r2_sys / (bc + 1),
    )


def test_phonon2(model, data_loader, criterion, criterion2, r2, device):
    """Test function for model architectures is one of the following,
    `graphnetwork`, `graphnetwork2`, `mlp`, or `mlp2`."""

    model.eval()

    with torch.no_grad():
        loss_rmse, loss_mse, loss_mae, loss_r2 = 0, 0, 0, 0
        # sourcery skip: no-loop-in-tests
        for bc, batch in enumerate(data_loader):
            batch.to(device)

            dos = model(batch)
            mse = criterion(dos, batch.phdos).cpu()
            rmse = torch.sqrt(mse)

            loss_mse += mse.mean()
            loss_rmse += rmse.mean()

            mae = criterion2(dos, batch.phdos).cpu()
            loss_mae += mae

            r2_score = r2(batch.phdos, dos)
            loss_r2 += r2_score

    return (
        loss_rmse / (bc + 1),
        loss_mse / (bc + 1),
        loss_mae / (bc + 1),
        loss_r2 / (bc + 1),
    )


# default_dtype = torch.float64
# torch.set_default_dtype(default_dtype)


def load_data(filename):
    """Load data from a csv file and return a pandas dataframe and a
    list of unique species in the data structure."""

    df = pd.read_csv(filename)

    df["structure"] = df["structure"].apply(eval).map(Atoms.fromdict)
    df["formula"] = df["structure"].map(lambda x: x.get_chemical_formula())
    df["species"] = df["structure"].map(lambda x: list(set(x.get_chemical_symbols())))
    df["phfreq"] = df["phfreq"].apply(eval).apply(np.array)  # type: ignore
    df["phdos"] = df["phdos"].apply(eval).apply(np.array)  # type: ignore
    df["pdos"] = df["pdos"].apply(eval)

    species = sorted(list(set(df["species"].sum())))
    return df, species


def train_valid_test_split(df, species, valid_size, test_size, seed):
    """Perform an element-balanced train/valid/test split."""

    dev_size = valid_size + test_size
    stats = get_element_statistics(df, species)
    idx_train, idx_dev = split_data(stats, dev_size, seed)

    stats_dev = get_element_statistics(df.iloc[idx_dev], species)
    idx_valid, idx_test = split_data(stats_dev, test_size / dev_size, seed)
    idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

    assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0

    return idx_train, idx_valid, idx_test


def get_element_statistics(df, species):
    """Return a dataframe with element statistics."""

    species_dict = {k: [] for k in species}
    for entry in df.itertuples():
        for specie in entry.species:
            species_dict[specie].append(entry.Index)

    # create dataframe of element statistics
    stats = pd.DataFrame({"symbol": species})
    stats["data"] = stats["symbol"].astype("object")
    for specie in species:
        stats.at[stats.index[stats["symbol"] == specie].values[0], "data"] = (
            species_dict[specie]
        )
    stats["count"] = stats["data"].apply(len)

    return stats


def split_data(df, test_size, seed):
    """Split data."""

    # initialize output arrays
    idx_train, idx_test = [], []

    # remove empty examples
    df = df[df["data"].str.len() > 0]

    # sort df in order of fewest to most examples
    df = df.sort_values("count")

    for _, entry in df.iterrows():
        df_specie = entry.to_frame().T.explode("data")

        try:
            idx_train_s, idx_test_s = train_test_split(
                df_specie["data"].values, test_size=test_size, random_state=seed
            )
        except Exception as e:
            # too few examples to perform split - these examples will be assigned based on other constituent elements
            # (assuming not elemental examples)
            pass

        else:
            # add new examples that do not exist in previous lists
            idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
            idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]

    return idx_train, idx_test


def element_representation(x, idx):
    """Get fraction of samples containing given element in dataset."""

    return len([k for k in x if k in idx]) / len(x)


def build_data(entry, r_max=5.0):
    """Build a PyTorch Geometric Data object from a given entry."""

    # one-hot encoding atom type and mass
    type_encoding = {}
    specie_am = []
    for z in range(1, 119):
        specie = Atom(z)  # type: ignore
        type_encoding[specie.symbol] = z - 1
        specie_am.append(specie.mass)

    type_onehot = torch.eye(len(type_encoding))
    am_onehot = torch.diag(torch.tensor(specie_am))
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list(
        "ijS", a=entry.structure, cutoff=r_max, self_interaction=True
    )

    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[
        torch.from_numpy(edge_src)
    ]
    edge_vec = (
        positions[torch.from_numpy(edge_dst)]
        - positions[torch.from_numpy(edge_src)]
        + torch.einsum(
            "ni,nij->nj",
            torch.tensor(edge_shift, dtype=torch.float64),
            lattice[edge_batch],
        )
    )

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    if entry.crystal_system == "cubic":
        system = 0
    elif entry.crystal_system == "hexagonal":
        system = 1
    elif entry.crystal_system == "monoclinic":
        system = 5
    elif entry.crystal_system == "orthorhombic":
        system = 4
    elif entry.crystal_system == "tetragonal":
        system = 2
    elif entry.crystal_system == "trigonal":
        system = 3
    else:
        system = 6
    return Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        x=am_onehot[
            [type_encoding[specie] for specie in symbols]
        ],  # atomic mass (node feature)
        z=type_onehot[
            [type_encoding[specie] for specie in symbols]
        ],  # atom type (node attribute)
        edge_index=torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        ),
        edge_shift=torch.tensor(edge_shift, dtype=torch.float64),
        edge_vec=edge_vec,
        edge_len=edge_len,
        phdos=torch.from_numpy(entry.phdos).unsqueeze(0),
        system=torch.tensor(system),
        mp_id=entry.mp_id,
    )
