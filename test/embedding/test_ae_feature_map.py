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
Tests for the ae_feature_map.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qdna.embedding import AeFeatureMap

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestAeFeatureMap(TestCase):

    def test_rnd_state(self):
        n_qubits = 6
        state_vector = (np.random.rand(2**n_qubits)-0.5)*2
        state_vector = state_vector / np.linalg.norm(state_vector)

        feature_map = AeFeatureMap(n_qubits, normalize=False, set_global_phase=True)

        circuit = QuantumCircuit(n_qubits)
        circuit.compose(feature_map, list(range(n_qubits)), inplace=True)
        circuit.assign_parameters(state_vector, inplace=True)

        state = Statevector(circuit)

        self.assertTrue(np.allclose(state_vector, state, rtol=1e-03, atol=1e-05))

    def test_non_normalized_state(self):
        # This test is important because during NQE training
        # the parameter vector is not normalized.
        n_qubits = 4
        state_vector = [
            -3.1416, 0.0000, -0.7667, 0.5835, 0.3468, 0.5320, 0.6706, 0.6756,
            0.5270, 0.7115, 0.5165, 0.7194, -0.5392, 0.3878, 0.5628, 0.7304
        ]

        feature_map = AeFeatureMap(n_qubits, normalize=True, set_global_phase=True)

        circuit = QuantumCircuit(n_qubits)
        circuit.compose(feature_map, list(range(n_qubits)), inplace=True)
        circuit.assign_parameters(state_vector, inplace=True)

        state = Statevector(circuit)

        state_vector = state_vector / np.linalg.norm(state_vector)
        self.assertTrue(np.allclose(state_vector, state, rtol=1e-03, atol=1e-05))

    def test_sparse_state(self):
        # Ensure that embedding is preventing division by zero.
        n_qubits = 4
        state_vector = [
            0.0000, 0.0000, 0.0000, 0.5835, 0.0000, 0.5320, 0.6706, 0.6756,
            0.5270, 0.0000, 0.0000, 0.7194, 0.0000, 0.0000, 0.0000, 0.7304
        ]

        feature_map = AeFeatureMap(n_qubits, normalize=True, set_global_phase=False)

        circuit = QuantumCircuit(n_qubits)
        circuit.compose(feature_map, list(range(n_qubits)), inplace=True)
        circuit.assign_parameters(state_vector, inplace=True)

        state = Statevector(circuit)

        state_vector = state_vector / np.linalg.norm(state_vector)
        self.assertTrue(np.allclose(state_vector, state, rtol=1e-03, atol=1e-05))

    def test_fixed_state(self):
        n_qubits = 6
        state_vector = [-0.00750879, -0.19509541,  0.13329143, -0.05016399,
                         0.15207381,  0.07559872,  0.06584141,  0.02580133,
                         0.07838591,  0.1797185,  -0.08671011,  0.12130839,
                        -0.01520751, -0.16350927, -0.14340019,  0.06187,
                        -0.15696974, -0.07004782, -0.10200592, -0.13275576,
                         0.03975269, -0.06866376, -0.19666519, -0.19759358,
                        -0.06338929,  0.01957956, -0.04101603, -0.17654797,
                         0.13379771,  0.13126602, -0.06316672, -0.16629322,
                         0.19740418, -0.01161936,  0.00660316,  0.18850766,
                         0.05001657,  0.05713368, -0.10440029, -0.19293127,
                        -0.21320381, -0.15065956,  0.20290586, -0.03929146,
                         0.0254895,   0.03343766, -0.17821551, -0.1903655,
                         0.13099242, -0.11116633,  0.15833651, -0.02708352,
                        -0.03610143,  0.20473973,  0.03146476, -0.03039286,
                        -0.14796486,  0.00971746, -0.15238775,  0.12996466,
                        -0.19630254, -0.08859373, -0.12830557,  0.12446281]

        feature_map = AeFeatureMap(n_qubits, normalize=False, set_global_phase=True)

        circuit = QuantumCircuit(n_qubits)
        circuit.compose(feature_map, list(range(n_qubits)), inplace=True)
        circuit.assign_parameters(state_vector, inplace=True)

        state = Statevector(circuit)

        self.assertTrue(np.allclose(state_vector, state, rtol=1e-03, atol=1e-05))
