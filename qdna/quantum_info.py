# Copyright 2023 qdna-lib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import networkx as nx
from qiskit.quantum_info import Statevector, partial_trace
import numpy as np

def von_neumann_entropy(rho):
    '''
    Compute the Von Neumann entropy (entanglement measure).

    To calculate the entropies, it is convenient to calculate the
    eigendecomposition of `rho` and use the eigenvalues `lambda_i` to determine
    the entropy:
        `S(rho) = -sum_i( lambda_i * ln(lambda_i)  )`
    '''
    evals = np.real(np.linalg.eigvals(rho.data))
    return -np.sum([e * np.log2(e) for e in evals if 0 < e < 1])

def concurrence(rho):
    '''
    Concurrence specifically quantifies the quantum entanglement between the
    two qubits. It does not consider classical correlations.

    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.2245
    '''
    # Compute the spin-flipped state
    sigma_y = np.array([[0, -1j], [1j, 0]])
    rho_star = np.conj(rho)
    rho_tilde = np.kron(sigma_y, sigma_y) @ rho_star @ np.kron(sigma_y, sigma_y)

    # Calculate the eigenvalues of the product matrix
    eigenvalues = np.linalg.eigvals(rho @ rho_tilde)
    # Sort in decreasing order
    eigenvalues = np.sort(np.sqrt(np.abs(eigenvalues)))[::-1]

    # Compute the concurrence
    return max(0, eigenvalues[0] - sum(eigenvalues[1:]))

def mutual_information(rho_a, rho_b, rho_ab):
    '''
    Mutual information quantifies the total amount of correlation between two
    qubits. It includes both classical and quantum correlations.
    '''

    # Compute the Von Neumann entropy for each density matrix
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    s_ab = von_neumann_entropy(rho_ab)

    # Calculate the mutual information
    return s_a + s_b - s_ab

def correlation(state_vector, set_a, set_b, correlation_measure=mutual_information):
    '''
    Compute the correlation between subsystems A and B.
    '''

    if (len(set_a) > 1 or len(set_b) > 1) and correlation_measure is concurrence:
        raise ValueError(
            "The value of `correlation_measure` cannot be `concurrence` when "
            "`len(set_a) > 1` or `len(set_b) > 1`. Choose, for example, "
            "`mutual_information` instead."
        )

    psi = Statevector(state_vector)

    # Compute the reduced density matrix for the union of the two sets.
    set_ab = set_a.union(set_b)
    rho_ab = partial_trace(psi, list(set(range(psi.num_qubits)).difference(set_ab)))

    # Maintains the relative position between the qubits of the two subsystems.
    new_set_a = [sum(i < item for i in set_ab) for item in set_a]
    new_set_b = [sum(i < item for i in set_ab) for item in set_b]

    # Calculate the reduced density matrice for each set.
    rho_a = partial_trace(rho_ab, new_set_b)
    rho_b = partial_trace(rho_ab, new_set_a)

    if correlation_measure is mutual_information:
        return correlation_measure(rho_a, rho_b, rho_ab)

    return correlation_measure(rho_ab)

def correlation_graph(state_vector,
                      n_qubits,
                      min_set_size=1,
                      max_set_size=1,
                      correlation_measure=mutual_information
):
    '''
    Initialize a graph where nodes represent qubits and the weights represent
    the entanglement between pairs of qubits in a register of `n` qubits for a
    pure state.

    O(n^2) x O(2^n)
    '''
    if n_qubits <= max_set_size <= 0:
        raise ValueError(
            f"The value of `max_set_size` [{max_set_size}] must be greater "
            f"than zero and less than `n_qubits` [{n_qubits}]."
        )

    if max_set_size < min_set_size <= 0:
        raise ValueError(
            f"The value of `min_set_size` [{min_set_size}] must be greater "
            f"than zero and less or equal than `max_set_size` "
            f"[{max_set_size}]."
        )

    # Create a graph.
    graph = nx.Graph()

    # Add nodes for each set of qubits up to the size `max_set_size`.
    for set_size in range(min_set_size, max_set_size + 1):
        for qubit_set in itertools.combinations(range(n_qubits), set_size):
            graph.add_node(qubit_set)

    # Add edges with weights representing entanglement.
    for node_a in graph.nodes():
        set_a = set(node_a)
        for node_b in graph.nodes():
            set_b = set(node_b)
            # Ensure non-overlapping sets.
            if not set_a.intersection(set_b):
                # Compute the correlation betweem subsystems.
                weight = correlation(
                    state_vector,
                    set_a,
                    set_b,
                    correlation_measure=correlation_measure
                )

                # Add an edge with the shared info as weight.
                graph.add_edge(node_a, node_b, weight=weight)

    return graph
