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
import random
import networkx as nx
from qiskit.quantum_info import partial_trace, Statevector
import numpy as np
from scipy.linalg import logm

def concurrence(rho_ab):
    '''
    Concurrence specifically quantifies the quantum entanglement between the
    two qubits. It does not consider classical correlations.

    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.2245
    '''
    # Compute the spin-flipped state
    sigma_y = np.array([[0, -1j], [1j, 0]])
    rho_star = np.conj(rho_ab)
    rho_tilde = np.kron(sigma_y, sigma_y) @ rho_star @ np.kron(sigma_y, sigma_y)

    # Calculate the eigenvalues of the product matrix
    eigenvalues = np.linalg.eigvals(rho_ab @ rho_tilde)
    # Sort in decreasing order
    eigenvalues = np.sort(np.sqrt(np.abs(eigenvalues)))[::-1]

    # Compute the concurrence
    return max(0, eigenvalues[0] - sum(eigenvalues[1:]))

def mutual_information(rho_ab):
    '''
    Mutual information quantifies the total amount of correlation between two
    qubits. It includes both classical and quantum correlations.
    '''
    # Compute the reduced density matrices for each qubit
    rho_a = partial_trace(rho_ab, [1])
    rho_b = partial_trace(rho_ab, [0])

    # Compute the Von Neumann entropy for each density matrix
    #     To calculate entropies, it is convenient to calculate the
    #     eigendecomposition of \rho.
    #         S(\rho) = -sum_i( \lambda_i * ln(\lambda_i)  )
    #     But as the matrices are small, I'll use the definition to
    #     make it easier to read.
    s_a = -np.trace(rho_a @ logm(rho_a)).real
    s_b = -np.trace(rho_b @ logm(rho_b)).real
    s_ab = -np.trace(rho_ab @ logm(rho_ab)).real

    # Calculate the mutual information
    return s_a + s_b - s_ab

def initialize_entanglement_graph(state_vector, n_qubits, quantify_shared_info=concurrence):
    '''
    Initialize a graph where nodes represent qubits and the weights represent
    the entanglement between pairs of qubits in a register of `n` qubits for a
    pure state.
    '''
    # Create a graph
    graph = nx.Graph()

    # Add nodes for each qubit
    for i in range(n_qubits):
        graph.add_node(i)

    # Add edges with weights representing entanglement
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # Compute the reduced density matrix for qubits i and j
            psi = Statevector(state_vector)
            rho_ij = partial_trace(psi, list(set(range(n_qubits)).difference([i,j])))

            # Compute the Von Neumann entropy (entanglement measure)
            shared_info = quantify_shared_info(rho_ij)

            # Add an edge with this entanglement measure as weight
            graph.add_edge(i, j, weight=shared_info)

    return graph

def min_cut_fixed_size_optimal(graph, size_a, size_b):
    '''
    Optimal solution to the minimum cut problem with fixed sizes for the sets.
    '''
    nodes = list(graph.nodes())
    min_cut_weight = float('inf')
    min_cut_partition = (set(), set())

    # Iterate over all combinations for set A
    for nodes_a in itertools.combinations(nodes, size_a):
        set_a = set(nodes_a)
        set_b = set(nodes) - set_a

        # Ensure the size of set B is as required
        if len(set_b) != size_b:
            continue

        # Calculate the sum of weights of edges between the two sets
        cut_weight = sum(
            graph[u][v]['weight'] for u in set_a for v in set_b if graph.has_edge(u, v)
        )

        # Update min cut if a lower weight is found
        if cut_weight < min_cut_weight:
            min_cut_weight = cut_weight
            min_cut_partition = (set_a, set_b)

    return min_cut_partition, min_cut_weight

def min_cut_fixed_size_heuristic(graph, size_a, size_b):
    '''
    Heuristic approach for the Min-Cut problem with a fixed number of nodes in
    each partition.

    O(k * n_a * n_b * m):
        k: number of iterations (vary significantly based on the graph's
        structure and the initial partitioning).
        n_a: subsystem A number of qubits.
        n_b: subsystem B number of qubits.
        m: number of edges between a node and subsystem B (typically equal to n_b).

        Example (n_a=n_b=n/2):
        O(k * n^3)
    '''
    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Shuffle nodes to randomize initial selection

    # Initialize sets A and B with random nodes
    set_a = set(nodes[:size_a])
    set_b = set(nodes[size_a:size_a + size_b])

    def calculate_cut_weight(node, set_b):
        return sum(graph[node][neighbor]['weight'] for neighbor in graph[node] if neighbor in set_b)

    # Iteratively try to improve the cut by swapping nodes between A and B
    improved = True
    while improved:
        improved = False
        for node in set_a:
            for other_node in set_b:
                set_c = set_b.copy()
                set_c.remove(other_node)
                set_c.add(node)
                if calculate_cut_weight(node, set_b) > calculate_cut_weight(other_node, set_c):
                    # Swap nodes
                    set_a.remove(node)
                    set_b.add(node)
                    set_b.remove(other_node)
                    set_a.add(other_node)
                    improved = True
                    break
            if improved:
                break

    # Sorts the sets
    if sorted(set_a)[0] > sorted(set_b)[0]:
        set_a, set_b = set_b, set_a

    # Calculate the sum of weights of edges between the two sets
    cut_weight = sum(
        graph[u][v]['weight'] for u in set_a for v in set_b if graph.has_edge(u, v)
    )

    return (set_a, set_b), cut_weight
