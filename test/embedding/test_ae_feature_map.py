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
from qdna_lib.embedding import AeFeatureMap
from qiskit.quantum_info import Statevector

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestAeFeatureMap(TestCase):

    def test_state(self):
        n_qubits = 6
        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        feature_map = AeFeatureMap(n_qubits)

        circuit = QuantumCircuit(n_qubits)
        circuit.compose(feature_map, list(range(n_qubits)), inplace=True)
        circuit.assign_parameters(state_vector, inplace=True)

        state = Statevector(circuit)

        self.assertTrue(np.allclose(state_vector, state))
