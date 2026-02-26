from dataclasses import asdict, dataclass, field
from typing import Optional
from core import as_function_node
import numpy as np
import pandas as pd


@dataclass
class EmbeddingsALL:
    """
    Global embedding configuration for ACE-like interatomic potentials.

    This dataclass defines the embedding functional form and its
    associated hyperparameters shared across all chemical elements.
    It is typically used as part of a hierarchical potential
    configuration for linear or nonlinear ACE models.

    Attributes:
        npot (str):
            Name of the embedding potential type (e.g. Finnis–Sinclair).
        fs_parameters (list[int]):
            Parameters controlling the embedding functional form.
        ndensity (int):
            Number of density channels used in the embedding.
    """
    npot: str = "FinnisSinclairShiftedScaled"
    fs_parameters: list[int] = field(default_factory=lambda: [1, 1])
    ndensity: int = 1


@dataclass
class Embeddings:
    """
    Container for embedding configurations.

    This wrapper dataclass groups embedding settings and allows
    extension to element-specific or interaction-specific embeddings
    in future potential configurations.

    Attributes:
        ALL (EmbeddingsALL):
            Embedding parameters applied to all elements.
    """
    ALL: EmbeddingsALL = field(default_factory=EmbeddingsALL)


@dataclass
class BondsALL:
    """
    Global bond (radial basis) configuration for ACE potentials.

    This dataclass specifies how pairwise distances are expanded
    into radial basis functions, including cutoff behavior and
    smoothing parameters.

    Attributes:
        radbase (str):
            Type of radial basis function.
        radparameters (list[float]):
            Parameters defining the radial basis.
        rcut (float | int):
            Radial cutoff distance in angstrom.
        dcut (float):
            Width of the cutoff smoothing region.
    """
    radbase: str = "SBessel"
    radparameters: list[float] = field(default_factory=lambda: [5.25])
    rcut: float | int = 7.0
    dcut: float = 0.01


@dataclass
class Bonds:
    """
    Container for bond interaction configurations.

    This wrapper allows bond settings to be shared or extended
    across different interaction types or chemical species.

    Attributes:
        ALL (BondsALL):
            Bond parameters applied to all elements.
    """
    ALL: BondsALL = field(default_factory=BondsALL)


@dataclass
class FunctionsALL:
    """
    Global basis function configuration for ACE potentials.

    This dataclass controls the angular and radial resolution
    of the basis expansion at different correlation orders,
    directly impacting accuracy and computational cost.

    Attributes:
        nradmax_by_orders (list[int]):
            Maximum number of radial basis functions per body order.
        lmax_by_orders (list[int]):
            Maximum angular momentum quantum number per body order.
    """
    nradmax_by_orders: list[int] = field(default_factory=lambda: [15, 3, 2, 1])
    lmax_by_orders: list[int] = field(default_factory=lambda: [0, 3, 2, 1])


@dataclass
class Functions:
    """
    Configuration for basis function allocation in ACE models.

    This dataclass specifies global limits on the number of basis
    functions and provides access to per-order angular and radial
    settings.

    Attributes:
        number_of_functions_per_element (int, optional):
            Maximum number of basis functions per chemical element.
        ALL (FunctionsALL):
            Basis function settings applied to all elements.
    """
    number_of_functions_per_element: Optional[int] = None
    ALL: FunctionsALL = field(default_factory=FunctionsALL)


@dataclass
class PotentialConfig:
    """
    Full configuration object for an ACE interatomic potential.

    This dataclass aggregates embedding, bond, and basis-function
    settings into a single hierarchical configuration that can be
    serialized and passed to fitting and evaluation routines.

    It is designed to be compatible with pyACE / LinearACE workflows.

    Attributes:
        deltaSplineBins (float):
            Resolution parameter for spline discretization.
        elements (list[str] or None):
            Chemical elements included in the potential.
        embeddings (Embeddings):
            Embedding configuration.
        bonds (Bonds):
            Bond interaction configuration.
        functions (Functions):
            Basis function configuration.
    """
    deltaSplineBins: float = 0.001
    elements: list[str] | None = None

    embeddings: Embeddings = field(default_factory=Embeddings)
    bonds: Bonds = field(default_factory=Bonds)
    functions: Functions = field(default_factory=Functions)

    def __post_init__(self):
        if not isinstance(self.embeddings, Embeddings):
            self.embeddings = Embeddings()
        if not isinstance(self.bonds, Bonds):
            self.bonds = Bonds()
        if not isinstance(self.functions, Functions):
            self.functions = Functions()

    def to_dict(self):
        """
        Convert the potential configuration to a dictionary.

        The resulting dictionary is recursively cleaned of ``None``
        values and is suitable for serialization or direct use in
        pyACE configuration constructors.

        Returns:
            dict:
                Nested dictionary representation of the potential
                configuration.
        """
        def remove_none(d):
            """Recursively remove None values from dictionaries."""
            if isinstance(d, dict):
                return {k: remove_none(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [remove_none(v) for v in d if v is not None]
            else:
                return d

        return remove_none(asdict(self))
    

@as_function_node
def ParameterizePotentialConfig(
    nrad_max: list | tuple  = (15, 6, 4, 1),
    l_max: list | tuple = (0, 6, 5, 1),
    number_of_functions_per_element: int = 10,
    rcut: float = 7.0,
):
    """
    Construct a parameterized ACE potential configuration.

    This node initializes a default ``PotentialConfig`` object and
    updates key hyperparameters controlling radial/angular resolution,
    basis size, and cutoff radius. It is typically used as a preparatory
    step before fitting a linear ACE model.

    Args:
        nrad_max (list or tuple):
            Maximum number of radial basis functions per body order.
        l_max (list or tuple):
            Maximum angular momentum per body order.
        number_of_functions_per_element (int):
            Maximum number of basis functions per chemical element.
        rcut (float):
            Radial cutoff distance in angstrom.

    Returns:
        PotentialConfig:
            Fully parameterized potential configuration object.
    """
    potential_config = PotentialConfig()

    potential_config.bonds.ALL.rcut = rcut
    potential_config.functions.ALL.nradmax_by_orders = list(nrad_max)
    potential_config.functions.ALL.lmax_by_orders = list(l_max)
    potential_config.functions.number_of_functions_per_element = (
        number_of_functions_per_element
    )

    return potential_config


@as_function_node
def RunLinearFit(
    potential_config,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    verbose: bool = False,
    filename: str = "",
    store: bool = False,
) -> str:
    """
    Fit a linear ACE interatomic potential to training data.

    This node constructs ACE basis functions from a potential
    configuration, fits a linear model to reference energies
    and forces, evaluates training and test errors, and writes
    the resulting potential to disk.

    It is intended for rapid prototyping and baseline model
    generation in atomistic machine-learning workflows.

    Args:
        potential_config:
            ACE potential configuration object.
        df_train (pandas.DataFrame):
            Training dataset containing ASE atoms, energies, and forces.
        df_test (pandas.DataFrame):
            Optional test dataset for validation.
        verbose (bool):
            If ``True``, print detailed information during matrix construction.
        filename (str):
            Base filename for saving the fitted potential.
        store (bool):
            Whether to store outputs in a workflow backend.

    Returns:
        str:
            Path to the saved potential YAML file.
    """
    from pyace import create_multispecies_basis_config
    from pyace.linearacefit import LinearACEDataset, LinearACEFit
    from pyiron_snippets.logger import logger
    import os

    logger.setLevel(30)

    elements_set = set()
    for at in df_train["ase_atoms"]:
        elements_set.update(at.get_chemical_symbols())
    for at in df_test["ase_atoms"]:
        elements_set.update(at.get_chemical_symbols())

    elements = sorted(elements_set)
    potential_config.elements = elements
    potential_config_dict = potential_config.to_dict()

    bconf = create_multispecies_basis_config(potential_config_dict)

    train_ds = LinearACEDataset(bconf, df_train)
    train_ds.construct_design_matrix(verbose=verbose)
    if df_test.empty is False:
        test_ds = LinearACEDataset(bconf, df_test)
        test_ds.construct_design_matrix(verbose=verbose)
    else:
        test_ds = None

    linear_fit = LinearACEFit(train_dataset=train_ds)
    linear_fit.fit()

    training_dict = linear_fit.compute_errors(train_ds)
    training_e_rmse = round(training_dict["epa_rmse"] * 1000, 2)
    training_f_rmse = round(training_dict["f_comp_rmse"] * 1000, 2)
    print("====================== TRAINING INFO ======================")
    print(f"Training E RMSE: {training_e_rmse:.2f} meV/atom")
    print(f"Training F RMSE: {training_f_rmse:.2f} meV/A")

    if test_ds is not None:
        testing_dict = linear_fit.compute_errors(test_ds)
        testing_e_rmse = round(testing_dict["epa_rmse"] * 1000, 2)
        testing_f_rmse = round(testing_dict["f_comp_rmse"] * 1000, 2)
        print("======================= TESTING INFO =======================")
        print(f"Testing E RMSE: {testing_e_rmse:.2f} meV/atom")
        print(f"Testing F RMSE: {testing_f_rmse:.2f} meV/A")
    else:
        testing_e_rmse = None
        testing_f_rmse = None

    basis = linear_fit.get_bbasis()
    
    if filename == "":
        filename = f"{'_'.join(basis.elements_name)}_linear_potential"
        folder_name = "Linear_ace_potentials"
    else:
        folder_name = os.path.dirname(filename)
        if folder_name =="":
            folder_name = "Linear_ace_potentials"
        filename = os.path.basename(filename)

    os.makedirs(folder_name, exist_ok=True)
    folder_path = os.path.join(os.getcwd(), folder_name)

    print(
        f'Potentials "{filename}.yaml" and "{filename}.yace" are saved in "{folder_path}".'
    )

    yace_file_path = f"{folder_path}/{filename}.yace"
    yaml_file_path = f"{folder_path}/{filename}.yaml"
    basis.save(yaml_file_path)
    basis.to_ACECTildeBasisSet().save_yaml(yace_file_path)
    potential_file_path = yaml_file_path
    
    return potential_file_path


@as_function_node
def PredictEnergiesAndForces(
    potential_file_path: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    store: bool = False
):
    """
    Predict energies and forces using a fitted ACE potential.

    This node evaluates a trained ACE model on training and test
    structures and returns reference and predicted quantities
    in a structured dictionary format suitable for error analysis
    and visualization.

    Args:
        potential_file_path (str):
            Path to the saved ACE potential file.
        df_train (pandas.DataFrame):
            Training dataset containing ASE atoms and reference data.
        df_test (pandas.DataFrame):
            Test dataset containing ASE atoms and reference data.
        store (bool):
            Whether to store outputs in a workflow backend.

    Returns:
        dict:
            Dictionary containing reference and predicted energies
            (per atom) and forces (per atom) for training and test data.
    """
    from pyace import PyACECalculator

    data_dict = {}

    ace = PyACECalculator(potential_file_path)

    training_structures = df_train.ase_atoms

    training_number_of_atoms = df_train.NUMBER_OF_ATOMS.to_numpy()
    training_energies = df_train.energy_corrected.to_numpy()

    training_epa = training_energies / training_number_of_atoms
    training_fpa = np.concatenate(df_train.forces.to_numpy()).flatten()
    data_dict["reference_training_epa"] = training_epa
    data_dict["reference_training_fpa"] = training_fpa

    training_predict = _get_predicted_energies_forces(
        ace=ace, structures=training_structures
    )
    data_dict["predicted_training_epa"] = (
        np.array(training_predict[0]) / training_number_of_atoms
    )
    data_dict["predicted_training_fpa"] = np.concatenate(training_predict[1]).flatten()

    if df_test.empty is False:
        testing_structures = df_test.ase_atoms

        testing_number_of_atoms = df_test.NUMBER_OF_ATOMS.to_numpy()
        testing_energies = df_test.energy_corrected.to_numpy()

        testing_epa = testing_energies / testing_number_of_atoms
        testing_fpa = np.concatenate(df_test.forces.to_numpy()).flatten()
        data_dict["reference_testing_epa"] = testing_epa
        data_dict["reference_testing_fpa"] = testing_fpa

        testing_predict = _get_predicted_energies_forces(
            ace=ace, structures=testing_structures, data_type='testing')
        data_dict["predicted_testing_epa"] = (
            np.array(testing_predict[0]) / testing_number_of_atoms
        )
        data_dict["predicted_testing_fpa"] = np.concatenate(
            testing_predict[1]
        ).flatten()

    return data_dict


def _get_predicted_energies_forces(ace, structures, data_type: str = 'training'):
    """
    Compute predicted energies and forces for a list of structures.

    This internal helper function evaluates an ACE calculator on a
    sequence of ASE ``Atoms`` objects and collects total energies
    and atomic forces.

    Args:
        ace:
            Initialized ACE calculator.
        structures:
            Iterable of ASE ``Atoms`` objects.
        data_type (str):
            Label used for progress reporting (e.g. training or testing).

    Returns:
        tuple[list[float], list[np.ndarray]]:
            Predicted total energies and forces for each structure.
    """
    from tqdm.auto import tqdm
    forces = []
    energies = []

    for s in tqdm(structures, desc = f"Predicting {data_type.capitalize()} Data"):
        s.calc = ace
        energies.append(s.get_potential_energy())
        forces.append(s.get_forces())
        s.calc = None
    return energies, forces