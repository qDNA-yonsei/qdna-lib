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
from dwave.samplers import SimulatedAnnealingSampler

def min_cut_fixed_size_optimal(graph, size_a, size_b):
    '''
    Optimal solution to the minimum cut problem with fixed sizes for the sets.

    O(n! / k!(n-k)!) x O(m), where m is the number of edges in the graph.
    The total number of edges m in a complete graph with n nodes is given by
    m = n(n-1) / 2
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

    This approach aims to find a partition of the graph where the total edge
    weight crossing the cut is as low as possible, given the size constraints.
    The algorithm iteratively attempts to reduce the total weight of the cut by
    swapping nodes between sets A and B, provided the swap decreases the total
    weight of the cut.
    This is a heuristic approach and may not always find the globally optimal
    solution, especially for complex or large graphs.

    O(k * n_a * n_b * m):
        k: number of iterations (vary significantly based on the graph's
        structure and the initial partitioning).
        n_a: subsystem A number of qubits.
        n_b: subsystem B number of qubits.
        m: number of edges between a node and subsystem B (typically equal to n_b).

        Example worst-case scenario (n_a=n_b=n/2):
        O(k * n^3)
    '''
    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Shuffle nodes to randomize initial selection

    # Initialize sets A and B with random nodes
    set_a = set(nodes[:size_a])
    set_b = set(nodes[size_a:size_a + size_b])

    def swapping_weight(node, other_node, set_node, set_other_node):
        # Entanglement (without - with) swapping.
        # A positive result indicates that a swap should be made
        # to reduce entanglement. That is, the entanglement with
        # a swap is smaller than without.
        return \
        sum(graph[node][neighbor]['weight']
                for neighbor in set_other_node if neighbor is not other_node) - \
        sum(graph[node][neighbor]['weight']
                for neighbor in set_node if neighbor is not node)

    # Iteratively try to improve the cut by swapping nodes between A and B
    improved = True
    while improved:
        improved = False
        for node_a in set_a:
            for node_b in set_b:
                weight_a = swapping_weight(node_a, node_b, set_a, set_b)
                weight_b = swapping_weight(node_b, node_a, set_b, set_a)
                total_weight = weight_a + weight_b
                # Here, the entanglements of both nodes are
                # considered simultaneously.
                if total_weight > 0:
                    # Swap the nodes if the total entanglement with the swap is
                    # smaller. In other words, the entanglement without
                    # swapping is greater, which means that `total_weight` is
                    # positive.
                    set_a.remove(node_a)
                    set_b.remove(node_b)
                    set_a.add(node_b)
                    set_b.add(node_a)
                    improved = True
                    break

            if improved:
                break

    # Sorts the sets when the subsystem sizes are equal
    if size_a == size_b and sorted(set_a)[0] > sorted(set_b)[0]:
        set_a, set_b = set_b, set_a

    # Calculate the sum of weights of edges between the two sets
    cut_weight = sum(
        graph[u][v]['weight'] for u in set_a for v in set_b if graph.has_edge(u, v)
    )

    return (set_a, set_b), cut_weight


# D-Wave functions

def graph_to_qubo(graph):
    '''
    Map the graph to a QUBO model.
    '''
    qubo = {}

    max_weight = max(weight for _, _, weight in graph.edges(data='weight'))

    for i, j, weight in graph.edges(data='weight'):
        # Set qubo_{ij} to be `max_weight - weight` for min-cut.
        weight = max_weight - weight
        qubo[(i, i)] = qubo.get((i, i), 0) - weight
        qubo[(j, j)] = qubo.get((j, j), 0) - weight
        qubo[(i, j)] = qubo.get((i, j), 0) + 2 * weight

    return qubo

def graph_to_ising(graph):
    '''
    Map the graph to an Ising model.
    '''
    ising = {}
    local_field = {}

    max_weight = max(weight for _, _, weight in graph.edges(data='weight'))

    for i, j, weight in graph.edges(data='weight'):
        # Set qubo_{ij} to be `max_weight - weight` for min-cut.
        weight = max_weight - weight
        ising[(i, j)] = weight

    # Add interaction strengths and local fields.
    for i in graph.nodes():
        local_field[i] = 0  # local fields

    return ising, local_field

def min_cut_dwave(graph, size_a=None, sample_method='ising', sampler=SimulatedAnnealingSampler(), num_reads=100):

    if sample_method == 'qubo':
        # Map the graph to a QUBO model.
        qubo = graph_to_qubo(graph)
        response = sampler.sample_qubo(qubo, num_reads=num_reads)
    else:
        # Map the graph to an Ising model.
        ising, local_field = graph_to_ising(graph)
        response = sampler.sample_ising(local_field, ising, num_reads=num_reads)
    print(response)
    # Find the sample with the lowest energy (best result), respecting the size of the block.
    min_cut_weight = float('inf')
    set_a = None
    if size_a is not None:
        for sample, energy in response.data(['sample', 'energy']):
            if energy < min_cut_weight and sum(v for v in sample.values() if v==1) == size_a:
                min_cut_weight = energy
                set_a = sample
    else:
        set_a = response.first.sample
        min_cut_weight = response.first.energy

    # Subsystem B is the complement of subsystem A.
    nodes = set(graph.nodes())
    set_a = {k for k, v in set_a.items() if v == 1}
    size_a = len(set_a)
    set_b = nodes - set_a
    size_b = len(set_b)

    # Sorts the sets when the subsystem sizes are equal.
    if size_a == size_b and sorted(set_a)[0] > sorted(set_b)[0]:
        set_a, set_b = set_b, set_a

    # Calculate the sum of weights of edges between the two sets.
    cut_weight = sum(
        graph[u][v]['weight'] for u in set_a for v in set_b if graph.has_edge(u, v)
    )

    return (set_a, set_b), cut_weight
