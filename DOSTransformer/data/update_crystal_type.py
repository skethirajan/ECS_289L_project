"""Module to update the crystal system of the materials in the dataset"""

import argparse
from pathlib import Path
import pandas as pd
from mp_api.client import MPRester, MPRestError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def main(filename: Path, api_key: str):
    """Update the crystal system of the materials in the dataset"""

    df = pd.read_csv(filename)
    crystal_types = []

    remove_entries = []

    with MPRester(api_key) as mpr:
        for i, mp_id in enumerate(df["mp_id"]):
            try:
                print(i)
                structure = mpr.get_structure_by_material_id(mp_id)
                sga = SpacegroupAnalyzer(structure)
                crystal_system = sga.get_crystal_system()
                crystal_types.append(crystal_system)
            except MPRestError as e:
                crystal_types.append(None)
                print(f"MPIRest Error fetching data for {mp_id}: {e}")
            except AttributeError as e:
                crystal_types.append(None)
                remove_entries.append(i)
                print(f"Attribute Error fetching data for {mp_id}: {e}")

    df["crystal_system"] = crystal_types
    df.to_csv(filename.parent.joinpath("data.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api_key",
        type=str,
        metavar="",
        required=True,
        help="API key for Materials Project",
    )

    args = parser.parse_args()

    file = Path(__file__).parent.joinpath("data_old.csv").resolve()
    main(file, api_key=args.api_key)
