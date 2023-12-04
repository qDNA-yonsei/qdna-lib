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

        self.assertTrue(np.allclose(state_vector, state))

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
        self.assertTrue(np.allclose(state_vector, state))

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
        self.assertTrue(np.allclose(state_vector, state))
