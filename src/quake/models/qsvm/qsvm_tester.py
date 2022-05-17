from pathlib import Path
import pickle
import numpy as np
from qiskit import QuantumCircuit
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from quake.models.svm.utils import extract_feats, rearrange_scale
from quake.models.attention.train import (
    load_and_compile_network as load_attention_network,
)
from quake.models.attention.attention_dataloading import (
    read_data as read_data_attention,
)
from quake.models.cnn.cnn_dataloading import read_data as read_data_cnn
from quake.models.cnn.train import load_and_compile_network as load_cnn_network
from quake.dataset.generate_utils import Geometry
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import qutip
from qiskit_machine_learning.kernels import QuantumKernel
import time
from qiskit.circuit.parametervector import ParameterVector

from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from typing import Tuple


class QKTCallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]

def get_features(
    data_folder: Path,
    train_folder: Path,
    extractor_type: str,
    setup: dict,
    seed: int = 42,
    run_tf_eagerly: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Getting track features extracted.

    Parameters
    ----------
    data_folder: Path
        The input data folder path.
    train_folder: Path
        The output model folder path.
    extractor_type: str
        Wether to extract with cnn or with attention.
    setup: dict
        Settings dictionary.
    seed: int
        Rng seed for reproducibility
    run_tf_eagerly: bool
        Wether to enable eager execution.

    Raises
    ------
    NotImplementedError
        If extractor type not one of `svm` or `attention`.

    Returns
    -------
    dataset: list[np.ndarray]
        The featurespace for svm.
    labels: list[np.ndarray]
        The truth labels.
    """
    setup.update({"seed": seed, "run_tf_eagerly": run_tf_eagerly})

    load_map_folder = train_folder.parent / extractor_type

    if extractor_type == "cnn":
        read_data_fn = read_data_cnn
        load_net_fn = load_cnn_network
    elif extractor_type == "attention":
        read_data_fn = read_data_attention
        load_net_fn = load_attention_network
    else:
        raise NotImplementedError(
            f"exctractor model not implemented, found: {extractor_type}"
        )

    train_generator, val_generator, test_generator = read_data_fn(
        data_folder, load_map_folder, setup, split_from_maps=True
    )

    geo = Geometry(setup["detector"])

    # extractor setup
    esetup = setup["model"][extractor_type]
    esetup.update({"ckpt": load_map_folder / f"{extractor_type}.h5"})
    network = load_net_fn(esetup, setup["run_tf_eagerly"], geo=geo)
    should_add_extra_feats = setup["model"]["svm"]["should_add_extra_feats"]
    train_features, train_labels = extract_feats(
        train_generator, network, should_add_extra_feats
    )
    val_features, val_labels = extract_feats(
        val_generator, network, should_add_extra_feats
    )
    test_features, test_labels = extract_feats(
        test_generator, network, should_add_extra_feats
    )

    # training and saving the SVMs
    dataset = rearrange_scale(
        train_features,
        val_features,
        test_features,
        setup["model"]["svm"]["should_do_scaling"],
    )
    labels = [train_labels, val_labels, test_labels]

    return dataset, labels


def get_spherical_coordinates(statevector: Statevector, qubit: int) -> list[np.float64]:
    """Getting qubit's spherical coordinates from quantum the quantum statevector.

    Parameters
    ----------
    statevector: Statevector
        The multi-qubit final state.
    qbit: int
        The number of qubits of the circuit.

    Returns
    -------
    [theta, phi]: list[np.float64]
        Qubit's state in the Bloch sphere representation (azimuthal and polar angle).
    """
    state_qubit_base = list(statevector.to_dict().keys())
    state_qubit_amplitudes = list(statevector.to_dict().values())
    s0 = 0
    s1 = 0
    for i, state_qb in enumerate(state_qubit_base):
        if state_qb[qubit] == "0":
            s0 += state_qubit_amplitudes[i]
        else:
            s1 += state_qubit_amplitudes[i]
    r0 = np.abs(s0)
    phi0 = np.angle(s0)
    r1 = np.abs(s1)
    phi1 = np.angle(s1)

    r = np.sqrt(r0**2 + r1**2)
    theta = 2 * np.arccos(r0 / r)
    phi = phi1 - phi0
    return [theta, phi]

def get_subsample(dataset: np.ndarray, labels: np.ndarray, size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Getting a smaller subsample of a given dataset.
    
    Parameters
    ----------
    dataset: np.ndarray
        The features distribution.
    labels: np.ndarray
        The truth labels.
    size: int
        Size of the subsample.
    seed: int
        Seed initialization for reproducibility.

    Returns
    -------
    subs_dataset: np.ndarray
        Subsampling of the feature distribution.
    subs_labels: np.ndarray
        Subsample truth labels. 
    """
    subs_dataset, subs_labels = train_test_split(dataset, labels, train_size = size, random_state = seed)[::2]
    return subs_dataset, subs_labels

def align_kernel(kernel: QuantumKernel, dataset: np.ndarray, labels: np.ndarray, c: float):
    """Performing kernel alignment, forcing a statevector simulator.
    
    Parameters
    ----------
    kernel: QuantumKernel
        The parametric circuit for encoding.
    dataset: np.ndarray
        The features distribution.
    labels: np.ndarray
        The truth labels.
    c: float
        The cost SVM hyperparameter.
    
    Returns
    -------
    aligned_kernel: QuantumKernel
        The aligned parametric circuit for encoding
    """
    if labels.shape[0] > 100:
        dataset, labels = get_subsample(dataset, labels, 100, 42)
    loss_func = SVCLoss(C=c)
    initial_point = [1]
    original_instance = kernel.quantum_instance
    if not kernel.quantum_instance.is_statevector:
        kernel.quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    cb_qkt = QKTCallback()
    spsa_opt = SPSA(maxiter=10, callback=cb_qkt.callback, learning_rate=0.05, perturbation=0.05)
    qkt = QuantumKernelTrainer(quantum_kernel=kernel, loss=loss_func, optimizer=spsa_opt, initial_point=initial_point)
    try:
        qka_results = qkt.fit(dataset, labels)
        aligned_kernel = qka_results.quantum_kernel
    except:
        print("Couldn't perform kernel alignment. Returning a non-optimized kernel")
        kernel.assign_user_parameters(initial_point)
        aligned_kernel = kernel
    if not original_instance.is_statevector:
        aligned_kernel.quantum_instance = original_instance
    return aligned_kernel

# def align_kernel_2(kernel, dataset, labels, c):
#     loss_func = SVCLoss(C=c)
#     nevals = 20
#     thetas = np.linspace(0, 2*np.pi, num = nevals)
#     losses = np.zeros(nevals)
#     for i, th in enumerate(thetas):
#         losses[i] = loss_func.evaluate(parameter_values = [th], quantum_kernel = kernel, data = dataset, labels = labels)
#     best_parameters = [thetas[np.argmin(losses)]]
#     kernel.assign_user_parameters(best_parameters)
#     return kernel


def plot_bloch(x: ParameterVector, quantum_kernel:QuantumKernel, train_set: np.ndarray, train_labels: np.ndarray, title: str)->plt.figure:
    """Plotting the encoded dataset on Bloch spheres.

    Parameters
    ----------
    x: ParameterVector
        The free parameters of the circuit.
    quantum_kernel: QuantumKernel
        The encoding parametric circuit.
    train_set: np.ndarray
        Points to encode in the quantum state.
    train_labels: np.ndarray
        Truth labels.
    title: str
        Name of the encoding as figure title.        

    Returns
    -------
    fig: plt.figure
        Plot of the encoded distribution.
    """
    featuremap = quantum_kernel.feature_map
    nqubits = featuremap.num_qubits
    if nqubits > 2:
        cols = 3
        rows = nqubits // cols + 1
    else:
        cols = nqubits
        rows = 1

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(5)
    fig.set_figwidth(5)

    if quantum_kernel.user_parameters:
        featuremap = quantum_kernel.feature_map
    for qb in range(nqubits):
        ax = fig.add_subplot(rows, cols, qb + 1, projection="3d")
        b = qutip.Bloch(fig=fig, axes=ax)
        b.point_size = [1]
        b.point_marker = ["o"]
        pnts = np.zeros((len(train_labels), 3))

        for i, val in enumerate(train_set):
            bound_circuits = featuremap.assign_parameters({x: val})
            state = Statevector.from_instruction(bound_circuits)
            spherical = get_spherical_coordinates(state, qb)
            xs = np.sin(spherical[0]) * np.cos(spherical[1])
            ys = np.sin(spherical[0]) * np.sin(spherical[1])
            zs = np.cos(spherical[0])
            pnts[i] = [xs, ys, zs]
        b.add_points(pnts[train_labels == 1].T)
        b.add_points(pnts[train_labels == 0].T)
        b.render()
        ax.set_title(f"Feature {qb}", y=1.1, fontsize=10)
    fig.suptitle(f'{title}{" kernel"}', y = 0.9)
    return fig

accuracy = lambda label, y: np.sum(label == y)/label.shape[0]

def make_kernels(maps: list[QuantumCircuit], qt_instance: QuantumInstance)->list[QuantumKernel]:
    """Returning quantum kernels from quantum featuremaps.

    Parameters
    ----------
    maps: list[QuantumCircuit]
        Quantum featuremaps.
    qt_instance:
        Backend for circuit execution

    Returns
    -------
    kernels: list[QuantumKernel]
        Quantum kernels associated to the featuremaps.
    """
    kernels = []
    for fmap in maps:
        if len(fmap.parameters) > len(fmap.qubits):
            theta = fmap.parameters[len(fmap.qubits):]
            kernels.append(QuantumKernel(feature_map=fmap, user_parameters=theta, quantum_instance=qt_instance))
        else:
            kernels.append(QuantumKernel(feature_map=fmap, quantum_instance=qt_instance))
    return kernels


def save_object(directory: Path, var: dict, name: str):
    """Saving pickle or png images in a folder.

    Parameters
    ----------
    directory: Path
        Folder path.
    var: dict
        Object to save.
    name: str
        Filename.
    """
    if name.endswith(".pkl"):
        with open(directory / Path(name), "wb") as f:
            pickle.dump(var, f)
    elif name.endswith(".png"):
        var.savefig(directory / Path(name), bbox_inches="tight", dpi=500)
    else:
        print(f'{"Could not save "} {name} {", can only save .pkl and .png"}')

def plot_data_2d(dataset: list[np.ndarray], labels: list[np.ndarray])->plt.figure:
    """Creating a picture that displays feature distribution of a 2D dataset.
    
    Parameters
    -----------
    dataset: list[np.ndarray]
        The featurespace for SVM.
    labels: list[np.ndarray]
        The truth labels.
    
    Returns
    -------
    fig: plt.figure
        Figure showing 1D and 2D feature distributions.
    """

    fig = plt.figure(figsize=(50, 100))
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    x1_min, x1_max = np.min(dataset[0][:,0]), np.max(dataset[0][:,0])
    x2_min, x2_max = np.min(dataset[0][:,1]), np.max(dataset[0][:,1])
    correlation_coefficients = np.zeros(2)

    for i in range(2):
        axs[0, i].hist(
            dataset[0][labels[0] == 0, i], range=[x1_min, x1_max], bins=50, alpha=0.6
        )
        axs[0, i].hist(
            dataset[0][labels[0] == 1, i], range=[x2_min, x2_max], bins=50, alpha=0.6
        )
        axs[0, i].legend(["Single beta", "Double beta"])
        axs[0, i].set_title("Feature " + str(i))

        axs[1, i].scatter(
            dataset[i][labels[i] == 0, 0],
            dataset[i][labels[i] == 0, 1],
            s=1,
            alpha=0.3,
        )
        axs[1, i].scatter(
            dataset[i][labels[i] == 1, 0],
            dataset[i][labels[i] == 1, 1],
            s=1,
            alpha=0.3,
        )
        axs[1, i].set_xlim([x1_min, x1_max])
        axs[1, i].set_ylim([x2_min, x2_max])
        axs[1, i].set_xlabel("Feature 0")
        axs[1, i].set_ylabel("Feature 1")
        if i == 0:
            axs[1, i].set_title("Training sample distribution")
        elif i == 1:
            axs[1, i].set_title("Validation sample distribution")

        correlation_coefficients[i] = np.corrcoef(
            dataset[i][:, 0], dataset[i][:, 1]
        )[0, 1]

        axs[1, i].text(
            0.35,
            0.40,
            f"L. correlation = {correlation_coefficients[i]:.2}",
            bbox=props,
        )

        axs[1, i].legend(["Single beta", "Double beta"])

    return fig

def plot_data_nd(dataset: list[np.ndarray], labels: list[np.ndarray], nfeatures: int)->plt.figure:
    """Creating a picture that displays feature distribution of a n-dimensional dataset.
    
    Parameters
    -----------
    dataset: list[np.ndarray]
        The featurespace for SVM.
    labels: list[np.ndarray]
        The truth labels.
    nfeatures: int
        The number of features in the dataset.
    
    Returns
    -------
    fig: plt.figure
        Figure showing the feature distributions.
    fig: plt.figure
        Figure showing the correlation matrix.
    """
    fig = plt.figure(figsize=(50, 100))
    rows = (nfeatures - 1) // 2 + 1
    if nfeatures == 1:
        cols = 1
    else:
        cols = 2
    fig, axs = plt.subplots(rows, cols)
    fig.set_figheight(5 * rows)
    fig.set_figwidth(5 * cols)

    for i in range(nfeatures):
        x_min, x_max = np.min(dataset[0][:,0]), np.max(dataset[0][:,0])
        col = i % 2
        row = i // 2
        axs[row, col].hist(
            dataset[0][labels[0] == 0, i], range=[x_min, x_max], bins=50, alpha=0.6
        )
        axs[row, col].hist(
            dataset[0][labels[0] == 1, i], range=[x_min, x_max], bins=50, alpha=0.6
        )
        axs[row, col].legend(["Single beta", "Double beta"])
        axs[row, col].set_title(f' Feature {i}')

    fig_correlation_matrix = plt.imshow(
                        np.abs(np.asmatrix(np.corrcoef(dataset[0].T))),
                        interpolation="nearest",
                        origin="upper",
                        cmap="Blues",
                    )                 
    plt.xticks(np.arange(nfeatures), [f'feature{i}' for i in range(nfeatures)], fontsize = 8)   
    plt.yticks(np.arange(nfeatures), [f'feature{i}' for i in range(nfeatures)], fontsize = 8)   

    plt.colorbar(orientation="vertical")
    plt.title("Correlation matrix (abs values)")
    return fig, fig_correlation_matrix


class SvmsComparison:
    """Class that trains and compares SVMs and QSVMs."""

    def __init__(
        self,
        x: ParameterVector,
        quantum_featuremaps: list[QuantumCircuit],
        quantum_instance: QuantumInstance,
        quantum_kernels: list[QuantumKernel],
        cs: list[float],
        backend,
        folder_name: Path = None,
        training_size: list[int] = [10, 20, 50, 100, 200],
        val_size: int = 200,
        test_size: int = 200,
        folds: int = 20,
        kernel_names: list[str] = None,
    ):
        """
        Parameters:
        x: ParameterVector
            The free parameters of the circuit.
        quantum_featuremaps: list[QuantumCircuit]
            The quantum featuremaps.
        quantum_instance: QuantumInstance
            Backend for circuit execution.
        quantum_kernels: list[QuantumKernel]
            The quantum kernels.
        cs: list[float]
            Penalty hyperparameters for the SVM algorithm.
        training_size: [int]
            Sizes of the training dataset.
        val_size: int
            Size of the validation dataset.
        test_size: int
            Size of the testing dataset.
        folds: int
            Number of trainings to execute for each element of training_size.
        folder_name: Path
            Folder name in which to save results.
        Kernel_Names list[str]:
            Names of the quantum encodings.
        """
        self.x = x
        self.quantum_featuremaps = quantum_featuremaps
        self.quantum_instance = quantum_instance
        self.quantum_kernels = quantum_kernels
        self.cs = cs
        self.training_size = training_size
        self.val_size = val_size
        self.test_size = test_size
        self.folds = folds
        if folder_name: self.path = Path("../../") / folder_name
        self.titles = ["Linear", "Polynomial", "Rbf"] + kernel_names
        self.backend = backend
        self.alignment = False
        for kernels in self.quantum_kernels:
            if kernels.user_parameters:
                self.alignment = True
                break

    def make_folder(self):
        """Creating the folder in the path stored in self.path. If self.path does not exists, the folder name is the datetime."""

        if not hasattr(self, 'path'):
            t = time.strftime("%Y_%m_%d-%Hh%Mm")
            self.path = Path('../../'+str(t))
        self.path.mkdir(exist_ok = True)

    def plot_data(self, dataset: list[np.ndarray], labels: list[np.ndarray]):
        """Plotting the feature distributions.

        Parameters:
        ----------
        dataset: list[np.ndarray]
            The featurespace for SVM.
        labels: list[np.ndarray]
            The truth labels.
        """
        print(f"Plotting the dataset")
        nfeatures = dataset[0].shape[1]
        if nfeatures == 2:
            self.feature_distributions = plot_data_2d(dataset, labels)
        elif nfeatures > 2:
            self.feature_distributions, self.correlation_matrix = plot_data_nd(dataset, labels, nfeatures)
  
    def train_svms(self, dataset: list[np.ndarray], labels: list[np.ndarray]):
        """Training classical and quantum SVMs and getting the predictions for different training set size and subsamples.

        Parameters:
        -----------
        dataset: list[np.ndarray]
            The featurespae for SVM.
        labels: list[np.ndarray]
            The truth labels.
        """
        print(f"Starting the session")
        linear = {"kernel": "linear", "C": 1, "gamma": 10}
        poly = {"kernel": "poly", "C": 0.1, "degree": 3}
        rbf = {"kernel": "rbf", "C": 1, "gamma": 10}
        opts = [linear, poly, rbf]

        svms_batch = [] # [trsize, folds, number of kernels]
        train_batch = [] # [trsize, folds, 0]; [trsize, folds, 1]
        validation_batch = []
        test_batch = []
        kernels_batch = [] # [trsize, folds, number of quantum kernels]
        training_preds_batch = []
        validation_preds_batch = []
        test_preds_batch = []

        seed = np.arange(1, 1 + self.folds)

        self.do_kernel_alignment(dataset[1], labels[1])
        for j, trs in enumerate(self.training_size):
            svms = []
            train = []
            validation = []
            test = []
            kernels = []
            train_preds = []
            validation_preds = []
            test_preds = []
            for i in range(0, self.folds):
                print(f"Training subset {i} with {trs} samples.")
                subset_train_data, subset_train_labels = get_subsample(
                    dataset[0], labels[0], trs, seed[i]
                )
                subset_val_data, subset_val_labels = get_subsample(
                    dataset[1],
                    labels[1],
                    self.val_size,
                    seed[i],
                )

                subset_test_data, subset_test_labels = get_subsample(
                    dataset[2],
                    labels[2],
                    self.test_size,
                    seed[i],
                )

                svcs = []
                kers = []
                tr_preds = []
                val_preds = []
                te_preds = []
                for cl in range(3):
                    svcs.append(SVC(**opts[cl]).fit(subset_train_data, subset_train_labels))
                    tr_preds.append(svcs[-1].predict(subset_train_data))
                    val_preds.append(svcs[-1].predict(subset_val_data))
                    te_preds.append(svcs[-1].predict(subset_test_data))

                for q, encoding in enumerate(self.quantum_kernels):
                    ker = encoding
                    # if encoding.user_parameters:
                    #     ker = align_kernel(encoding, subset_val_data, subset_val_labels, self.cs[q])
                    # else:
                    #     ker = encoding
                    ker_matrix_train = ker.evaluate(x_vec=subset_train_data)
                    ker_matrix_val = ker.evaluate(x_vec = subset_val_data, y_vec=subset_train_data)
                    ker_matrix_test = ker.evaluate(x_vec = subset_test_data, y_vec=subset_train_data)
                    kers.append(np.real(ker_matrix_train))
                    clf = SVC(kernel = 'precomputed', C = self.cs[q]).fit(ker_matrix_train, subset_train_labels)
                    svcs.append(clf)
                    tr_preds.append(clf.predict(ker_matrix_train))
                    val_preds.append(clf.predict(ker_matrix_val))
                    te_preds.append(clf.predict(ker_matrix_test))
                    

                svms.append(svcs)
                train.append([subset_train_data, subset_train_labels])
                validation.append([subset_val_data, subset_val_labels])
                test.append([subset_test_data, subset_test_labels])
                kernels.append(kers)
                train_preds.append(tr_preds)
                validation_preds.append(val_preds)
                test_preds.append(te_preds) 

            svms_batch.append(svms)
            train_batch.append(train)
            validation_batch.append(validation)
            test_batch.append(test)
            if trs < 201:
                kernels_batch.append(kernels)
            training_preds_batch.append(train_preds)
            validation_preds_batch.append(validation_preds)
            test_preds_batch.append(test_preds)

        self.svms = svms_batch
        self.train = train_batch
        self.validation = validation_batch
        self.test = test_batch
        if kernels_batch:
            self.kernels = kernels_batch
        self.train_preds = training_preds_batch
        self.validation_preds = validation_preds_batch
        self.test_preds = test_preds_batch

    def save(self, setting: dict):
        """Saving results and specifications into a folder, if they were previously produced.

        Parameters:
        -----------
        setting: dict
            Training session specifications.
        """
        print(f"Saving data in {self.path}")
        self.make_folder()
        save_object(self.path, setting, "setup.pkl")
        if hasattr(self, "svms"):
            save_object(self.path, self.svms, "SVMS.pkl")
        if hasattr(self, "train"):
            save_object(self.path, self.train, "TRAIN.pkl")
        if hasattr(self, "validation"):
            save_object(self.path, self.validation, "VALIDATION.pkl")
        if hasattr(self, "test"):
            save_object(self.path, self.test, "TEST.pkl")
        if hasattr(self, "kernels"):
            save_object(self.path, self.kernels, "KERNELS.pkl")
        if hasattr(self, "train_preds"):
            save_object(self.path, self.train_preds, "TRAIN_PREDICTIONS.pkl")
        if hasattr(self, "validation_preds"):
            save_object(self.path, self.validation_preds, "VALIDATION_PREDICTIONS.pkl")
        if hasattr(self, "test_preds"):
            save_object(self.path, self.test_preds, "TEST_PREDICTIONS.pkl")
        if hasattr(self, "feature_distributions"):
            save_object(self.path, self.feature_distributions, "feature_distributions.png")
        if hasattr(self, "correlation_matrix"):
            save_object(self.path, self.correlation_matrix, "correlation_matrix.png")
        if hasattr(self, "learning_curve"):
            save_object(self.path, self.learning_curve, "learning_curve.png")
        if hasattr(self, "learning_curve_cv"):
            save_object(self.path, self.learning_curve_cv, "learning_curve_cv.png")
        if hasattr(self, "kernel_plot"):
            save_object(self.path, self.kernel_plot, "kernel_plot.png")

        if hasattr(self, "featuremaps_plot"):
            subfolder = self.path / Path("Featuremaps") 
            subfolder.mkdir(exist_ok = True)
            for i in range(len(self.featuremaps_plot)):
                save_object(subfolder, self.featuremaps_plot[i], self.titles[3+i] +".png")

        if hasattr(self, "bloch_sphere_list"):
            subfolder = self.path / Path("Bloch Spheres")
            subfolder.mkdir(exist_ok = True)
            for i in range(len(self.bloch_sphere_list)):
                save_object(
                    subfolder,
                    self.bloch_sphere_list[i],
                    "Bloch_Spheres" + self.titles[3 + i] + ".png",
                )

    def load_files(self, path: Path):
        """Loading results contained in a folder into class fields.

        Parameters:
        path: Path
            Output folder name.
        """
        print(f"Loading data from {self.path}")
        with open(path / Path("SVMS.pkl"), "rb") as f:
            self.svms = pickle.load(f)
        with open(path / Path("TRAIN.pkl"), "rb") as f:
            self.train = pickle.load(f)
        with open(path / Path("VALIDATION.pkl"), "rb") as f:
            self.validation = pickle.load(f)
        with open(path / Path("TEST.pkl"), "rb") as f:
            self.test = pickle.load(f)
        with open(path / Path("KERNELS.pkl"), "rb") as f:
            self.kernels = pickle.load(f)
        with open(path / Path("TRAIN_PREDICTIONS.pkl"), "rb") as f:
            self.train_preds = pickle.load(f)
        with open(path / Path("VALIDATION_PREDICTIONS.pkl"), "rb") as f:
            self.validation_preds = pickle.load(f)
        with open(path / Path("TEST_PREDICTIONS.pkl"), "rb") as f:
            self.test_preds = pickle.load(f)

    def learning_curves(self):
        """Plotting learning curves for the different kernels."""
        print("Plotting the learning curves")
        fig = plt.figure(figsize=(50, 100))
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        acc_train = np.zeros((len(self.titles), len(self.training_size), self.folds))
        acc_validation = np.zeros((len(self.titles), len(self.training_size), self.folds))
        acc_test = np.zeros((len(self.titles), len(self.training_size), self.folds))
        for i, (tr_pred1, val_pred1, te_pred1) in enumerate(zip(self.train_preds, self.validation_preds, self.test_preds)):
            for j, (tr_pred2, val_pred2, te_pred2) in enumerate(zip(tr_pred1, val_pred1, te_pred1)):
                for k, (tr_pred3, val_pred3, te_pred3) in enumerate(zip(tr_pred2, val_pred2, te_pred2)):
                    acc_train[k, i, j] = accuracy(np.array(self.train[i][j][1]), tr_pred3)
                    acc_validation[k, i, j] = accuracy(np.array(self.validation[i][j][1]), val_pred3)
                    acc_test[k, i, j] = accuracy(np.array(self.test[i][j][1]), te_pred3)
        n = np.sqrt(acc_train[0][0].shape[0])
        y = [
            np.mean(acc_train, axis=2),
            np.mean(acc_validation, axis=2),
            np.mean(acc_test, axis=2),
        ]
        yerr = [
            np.std(acc_train, axis=2)/ n,
            np.std(acc_validation, axis=2)/ n,
            np.std(acc_test, axis=2)/ n,
        ]

        ylims = [[0, 1], [0, 1], [0, 1]]
        subtitle = [
            "Accuracy on training set",
            "Accuracy on validation set",
            "Accuracy on testing set",
        ]
        for j in range(3):
            col = j % 2
            row = j // 2
            for i in range(len(self.titles)):
                if yerr[j][i].sum() == 0:
                    axs[row, col].scatter(x=self.training_size, y=y[j][i])
                else:
                    axs[row, col].errorbar(x=self.training_size, y=y[j][i], yerr=yerr[j][i])
            axs[row, col].legend(self.titles)
            axs[row, col].set_title(subtitle[j])
            axs[row, col].set_xlabel("Training set size")
            axs[row, col].set_ylim(ylims[j])
            # axs[row,col].set_xscale('log')
        self.learning_curve = fig

    def plot_kernels(self):
        """Plotting the kernel matrices and returning the sum of all the elements as an indicator of data sparsity."""
        if not hasattr(self, "kernels"):
            return
        print("Plotting kernel matrices")
        nkernels = len(self.quantum_kernels)
        fig = plt.figure(constrained_layout=True)
        rows = (nkernels - 1) // 2 + 1

        if nkernels != 1:
            for j in range(nkernels):

                try:
                    ker = self.kernels[-1][0][j]
                except:
                    try:
                        ker = self.kernels[-1][j]
                    except:
                        ker = self.kernels[j]

                if nkernels == 2:
                    ax = fig.add_subplot(1, 2, j + 1)
                    kern_img = ax.imshow(
                         np.asmatrix(ker),
                         interpolation="nearest",
                         origin="upper",
                         cmap="Blues",
                    )
                    ax.set_title(
                        f'{self.titles[j + 3]} {" sum: "} {np.sum(ker):.0f}'
                    )
                    fig.colorbar(kern_img, ax = ax, fraction=0.046, pad=0.04)
                else:
                    ax = fig.add_subplot(rows, 2, j+1)
                    kern_img = ax.imshow(
                        np.asmatrix(ker),
                        interpolation="nearest",
                        origin="upper",
                        cmap="Blues",
                    )
                    ax.set_title(
                        f'{self.titles[j + 3]} {" sum: "} {np.sum(ker):.0f}'
                    )
                    fig.colorbar(kern_img, ax = ax, fraction=0.046, pad=0.04)
        else:
            ker = self.kernels[-1][0][0]
            plt.imshow(
                np.asmatrix(ker), interpolation="nearest", origin="upper", cmap="Blues"
            )
            plt.title(f'{self.titles[3]} {" sum: "}  {np.sum(ker):.0f}')
            plt.colorbar(fraction=0.046, pad=0.04)
            
        
        self.kernel_plot = fig

    def plot_decision_boundaries(self, cheap_version: bool = True):
        """Plotting and saving several decision boundaries for different training size and subsamples.
        
        Parameters
        ----------
        cheap_version: bool
            If true, creates only the datapoints colored according to the svms predictions.
        """
        print("Plotting decision boundaries")
        self.make_folder()
        subfolder = self.path / Path("Decision Boundaries")
        subfolder.mkdir(exist_ok = True)
        self.contourlist = []
        x_min, x_max = np.min(self.train[-1][0][0][:, 0]), np.max(
            self.train[-1][0][0][:, 0]
        )
        y_min, y_max = np.min(self.train[-1][0][0][:, 1]), np.max(
            self.train[-1][0][0][:, 1]
        )

        for i, trs in enumerate(self.training_size):
            for k, kern_titles in enumerate(self.titles):
                contour = plt.figure(constrained_layout=True)
                contour.set_figheight(8)
                contour.set_figwidth(12)
                if self.folds <= 3:
                    cols = self.folds
                    rows = 1
                    contour.set_figheight(4)
                    contour.set_figwidth(4*self.folds)
                    titlesize = 20
                else:
                    rows = 2
                    cols = 3
                    contour.set_figheight(8)
                    contour.set_figwidth(12)
                    titlesize = 30
                if k < 3:
                    h = 0.01
                else:
                    h = 0.2
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
                )
                for fld in range(np.min([self.folds, 6])):
                    ax = contour.add_subplot(rows, cols, fld + 1)
                    clf = self.svms[i][fld][k]
                    if not cheap_version and self.train[0][0][0].shape[1] == 2:
                        if k < 3:
                            z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                        else:
                            grid_kernel = self.quantum_kernels[k - 3].evaluate(
                                x_vec=np.c_[xx.ravel(), yy.ravel()],
                                y_vec=self.train[i][fld][0],
                            )
                            z = clf.predict(grid_kernel)
                        z = z.reshape(xx.shape)
                        ax.contourf(xx, yy, z, cmap=plt.cm.seismic, alpha=0.5)
                        ax.scatter(
                            self.train[i][fld][0][:, 0],
                            self.train[i][fld][0][:, 1],
                            c=self.train[i][fld][1],
                            cmap=plt.cm.seismic,
                            s=1,
                        )
                    else:
                        z = self.train_preds[i][fld][k]
                        ax.scatter(
                            self.train[i][fld][0][:, 0],
                            self.train[i][fld][0][:, 1],
                            c=z,
                            cmap=plt.cm.bwr,
                            s=1,
                        )


                    xmin, ymin = np.arange(x_min, x_max, 0.2).min(), np.arange(y_min, y_max, 0.2).min()
                    xmax, ymax = np.arange(x_min, x_max, 0.2).max(), np.arange(y_min, y_max, 0.2).max()
                    ax.set_xlabel("Feature 1")
                    ax.set_ylabel("Feature 2")
                    ax.set_xlim([xmin, xmax])
                    ax.set_ylim([ymin, ymax])
                    ax.set_xticks(())
                    ax.set_yticks(())
                contour.suptitle(f'{kern_titles} with {trs} samples', y = 1.1, fontsize=titlesize)
                save_object(self.path /  Path("Decision Boundaries"), contour, "Contours" + str(trs) + kern_titles + ".png")
                plt.close(contour)
                #plt.clf()
                del contour

    def plot_bloch_spheres(self, dataset: list[np.ndarray], labels: list[np.ndarray]):
        """Plotting the Bloch spheres for every circuit.
        
        Parameters:
        -----------
        dataset: list[np.ndarray]
            The dataset containing feature distribution to encode
        labels: list[np.ndarray]
            The truth labels
        """
        print("Plotting Bloch spheres")
        train_dataset, train_labels = get_subsample(dataset[0], labels[0], 100, 42)
        self.do_kernel_alignment(dataset[1], labels[1])
        self.bloch_sphere_list = []
        for i, qker in enumerate(self.quantum_kernels):
            self.bloch_sphere_list.append(plot_bloch(self.x, qker, train_dataset, train_labels, self.titles[3+i]))
    
    def train_svms_cv(self, dataset: list[np.ndarray], labels: list[np.ndarray]):
        """Training classical and quantum svms on a fixed subsample for every training size with cross-validation.
        
        Parameters
        ----------
        dataset: list[np.ndarray]
            The dataset containing the training subsets.
        labels: list[np.ndarray]
            The truth labels.
        """
        print("Doing cross validated trainings")
        linear = {"kernel": "linear", "C": 1, "gamma": 10}
        poly = {"kernel": "poly", "C": 0.1, "degree": 3}
        rbf = {"kernel": "rbf", "C": 1, "gamma": 10}
        opts = [linear, poly, rbf]

        train_dataset, train_labels = dataset[0], labels[0]
        self.do_kernel_alignment(dataset[1], labels[1])
        fig = plt.figure(constrained_layout=True)
        fig.set_figheight(5)
        fig.set_figwidth(5)

        folds = 5
        for cl in range(3):
            scores = np.zeros((len(self.training_size), folds))
            for i, trs in enumerate(self.training_size):
                subset, labels = get_subsample(train_dataset, train_labels, trs, 42)
                clf = SVC(**opts[cl])
                scores[i] = cross_val_score(clf, subset, labels, cv=folds)            
            avg_scores = np.mean(scores, axis = 1)
            std_scores = np.std(scores, axis = 1)/np.sqrt(folds)
            plt.errorbar(x=self.training_size, y=avg_scores, yerr=std_scores)    


        for q, encoding in enumerate(self.quantum_kernels):
            scores = np.zeros((len(self.training_size), folds))
            for i, trs in enumerate(self.training_size):
                subset, labels = get_subsample(train_dataset, train_labels, trs, 42)
                ker = encoding
                ker_matrix = ker.evaluate(x_vec=subset)
                clf = SVC(kernel="precomputed", C=self.cs[q])
                scores[i] = cross_val_score(clf, ker_matrix, labels, cv=folds)            
            avg_scores = np.mean(scores, axis = 1)
            std_scores = np.std(scores, axis = 1)/np.sqrt(folds)
            plt.errorbar(x=self.training_size, y=avg_scores, yerr=std_scores) 

        plt.ylim([0, 1])
        plt.title("Cross validated learning curve")
        plt.xlabel("Sample size")
        plt.ylabel("Accuracy")
        plt.legend(self.titles)

        self.learning_curve_cv = fig
    
    def plot_featuremaps(self):
        """Plotting the quantum featuremaps in the circuit representation."""
        print("Drawing the featuremaps")
        self.featuremaps_plot = []
        large_font = {
            "fontsize": 10,
            "subfontsize": 8,  
        }
        for i, fm in enumerate(self.quantum_featuremaps):
            fig = plt.figure(constrained_layout=True)
            fig.set_figheight(5)
            fig.set_figwidth(15)
            fig = fm.draw(output = 'mpl', plot_barriers = False, initial_state = True, style = large_font, fold = 20)
            plt.title(f'{self.titles[3+i]} featuremap', y = 0.9, fontsize=15)
            self.featuremaps_plot.append(fig)

    def do_kernel_alignment(self, dataset: list[np.ndarray], labels: list[np.ndarray]):
        """Checking if the input kernels need alignment. If so, perform kernel alignment.
        
        Parameters
        ----------
        dataset: list[np.ndarray]
            Dataset containing the alignment sample.
        labels: list[np.ndarray]
            The truth labels.
        """
        if self.alignment:
            kernels = self.quantum_kernels
            for i, kers in enumerate(kernels):
                if kers.user_parameters:
                    print(f'Doing kernel alignment of {self.titles[3+i]}  kernel')
                    self.quantum_kernels[i] = align_kernel(kers, dataset, labels, self.cs[i])
            self.alignment = False