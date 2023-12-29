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

"""
Tests for the entanglement.py module.
"""

from unittest import TestCase
from math import isclose
import numpy as np
from qdna.entanglement import min_cut_fixed_size_optimal, \
                              initialize_entanglement_graph, \
                              min_cut_fixed_size_heuristic


# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestEntanglement(TestCase):

    def test_min_cut_optimal_product_state(self):
        n_qubits = 3

        state_vector1 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector1 = state_vector1 / np.linalg.norm(state_vector1)

        state_vector2 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector2 = state_vector2 / np.linalg.norm(state_vector2)

        state_vector = np.kron(state_vector1, state_vector2)

        graph = initialize_entanglement_graph(state_vector, 6)
        (set_a, set_b), cut_weight = min_cut_fixed_size_optimal(graph, 3, 3)

        self.assertTrue(isclose(cut_weight, 0.0))
        self.assertTrue(np.allclose(sorted(set_a), [0, 1, 2]))
        self.assertTrue(np.allclose(sorted(set_b), [3, 4, 5]))

    def test_min_cut_heuristic_product_state(self):
        n_qubits = 3

        state_vector1 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector1 = state_vector1 / np.linalg.norm(state_vector1)

        state_vector2 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector2 = state_vector2 / np.linalg.norm(state_vector2)

        state_vector = np.kron(state_vector1, state_vector2)

        graph = initialize_entanglement_graph(state_vector, 6)
        (set_a, set_b), cut_weight = min_cut_fixed_size_heuristic(graph, 3, 3)

        self.assertTrue(isclose(cut_weight, 0.0))
        self.assertTrue(np.allclose(sorted(set_a), [0, 1, 2]))
        self.assertTrue(np.allclose(sorted(set_b), [3, 4, 5]))
