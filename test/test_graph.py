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
from qdna.quantum_info import correlation_graph, concurrence
from qdna.graph import min_cut_fixed_size_optimal, \
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

        graph = correlation_graph(state_vector, 6, correlation_measure=concurrence)
        (set_a, set_b), cut_weight = min_cut_fixed_size_optimal(graph, 3, 3)

        self.assertTrue(isclose(cut_weight, 0.0))
        self.assertTrue(np.allclose(sorted(sum(set_a, ())), [0, 1, 2]))
        self.assertTrue(np.allclose(sorted(sum(set_b, ())), [3, 4, 5]))

    def test_min_cut_heuristic_product_state(self):
        n_qubits = 3

        state_vector1 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector1 = state_vector1 / np.linalg.norm(state_vector1)

        state_vector2 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector2 = state_vector2 / np.linalg.norm(state_vector2)

        state_vector = np.kron(state_vector1, state_vector2)

        graph = correlation_graph(state_vector, 6, correlation_measure=concurrence)
        (set_a, set_b), cut_weight = min_cut_fixed_size_heuristic(graph, 3, 3)

        self.assertTrue(isclose(cut_weight, 0.0))
        self.assertTrue(np.allclose(sorted(sum(set_a, ())), [0, 1, 2]))
        self.assertTrue(np.allclose(sorted(sum(set_b, ())), [3, 4, 5]))

    def test_min_cut_optimal_fixed(self):
        state_vector = [0.00000000e+00, 6.94991284e-04, 6.34061054e-02, 2.13286776e-01,
                        2.05658826e-01, 8.15141431e-02, 8.40762648e-03, 0.00000000e+00,
                        0.00000000e+00, 9.96282923e-03, 1.59787371e-01, 2.46020212e-01,
                        2.40154124e-01, 1.88551405e-01, 1.90979862e-02, 0.00000000e+00,
                        9.31053205e-04, 4.54430541e-02, 2.03018262e-01, 1.86283916e-01,
                        1.52698965e-01, 1.86716850e-01, 3.98788754e-02, 0.00000000e+00,
                        9.31053205e-04, 7.52123527e-02, 2.05815792e-01, 1.50703564e-01,
                        1.29600833e-01, 1.44311684e-01, 7.06191583e-02, 0.00000000e+00,
                        0.00000000e+00, 7.69075128e-02, 1.73599511e-01, 1.17960532e-01,
                        1.25934109e-01, 1.33062442e-01, 8.39199837e-02, 0.00000000e+00,
                        0.00000000e+00, 3.84563398e-02, 1.76000915e-01, 1.11207757e-01,
                        1.36568022e-01, 1.59108898e-01, 6.19303017e-02, 0.00000000e+00,
                        0.00000000e+00, 9.01723392e-03, 1.70193955e-01, 1.96598638e-01,
                        2.21354420e-01, 1.95836976e-01, 4.14572396e-02, 6.39589243e-03,
                        0.00000000e+00, 2.11959665e-04, 6.15966561e-02, 2.17282223e-01,
                        2.49816212e-01, 1.27759885e-01, 2.73632167e-02, 1.20487999e-02]

        graph = correlation_graph(state_vector, 6, correlation_measure=concurrence)
        (set_a, set_b), cut_weight = min_cut_fixed_size_optimal(graph, 3, 3)

        self.assertTrue(isclose(cut_weight, 0.0))
        self.assertTrue(np.allclose(sorted(sum(set_a, ())), [0, 1, 2]))
        self.assertTrue(np.allclose(sorted(sum(set_b, ())), [3, 4, 5]))

    def test_min_cut_heuristic_fixed(self):
        state_vector = [0.00000000e+00, 6.94991284e-04, 6.34061054e-02, 2.13286776e-01,
                        2.05658826e-01, 8.15141431e-02, 8.40762648e-03, 0.00000000e+00,
                        0.00000000e+00, 9.96282923e-03, 1.59787371e-01, 2.46020212e-01,
                        2.40154124e-01, 1.88551405e-01, 1.90979862e-02, 0.00000000e+00,
                        9.31053205e-04, 4.54430541e-02, 2.03018262e-01, 1.86283916e-01,
                        1.52698965e-01, 1.86716850e-01, 3.98788754e-02, 0.00000000e+00,
                        9.31053205e-04, 7.52123527e-02, 2.05815792e-01, 1.50703564e-01,
                        1.29600833e-01, 1.44311684e-01, 7.06191583e-02, 0.00000000e+00,
                        0.00000000e+00, 7.69075128e-02, 1.73599511e-01, 1.17960532e-01,
                        1.25934109e-01, 1.33062442e-01, 8.39199837e-02, 0.00000000e+00,
                        0.00000000e+00, 3.84563398e-02, 1.76000915e-01, 1.11207757e-01,
                        1.36568022e-01, 1.59108898e-01, 6.19303017e-02, 0.00000000e+00,
                        0.00000000e+00, 9.01723392e-03, 1.70193955e-01, 1.96598638e-01,
                        2.21354420e-01, 1.95836976e-01, 4.14572396e-02, 6.39589243e-03,
                        0.00000000e+00, 2.11959665e-04, 6.15966561e-02, 2.17282223e-01,
                        2.49816212e-01, 1.27759885e-01, 2.73632167e-02, 1.20487999e-02]

        graph = correlation_graph(state_vector, 6, correlation_measure=concurrence)
        (set_a, set_b), cut_weight = min_cut_fixed_size_heuristic(graph, 3, 3)

        self.assertTrue(isclose(cut_weight, 0.0))
        self.assertTrue(np.allclose(sorted(sum(set_a, ())), [0, 1, 2]))
        self.assertTrue(np.allclose(sorted(sum(set_b, ())), [3, 4, 5]))
