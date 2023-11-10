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
Tests for the schmidt.py module.
"""

from unittest import TestCase
import numpy as np
import random
from qiskit import QuantumCircuit
from qdna_lib.compression import SchmidtCompressor
from qiskit.quantum_info import Statevector

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestSchmidt(TestCase):

    def test_exact(self):
        n_qubits = 6
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        
        partition = random.sample(range(n_qubits), n_qubits//2)
        compressor = SchmidtCompressor(state_vector, opt_params={'partition': partition})
        decompressor = compressor.inverse()

        circuit = QuantumCircuit(n_qubits)
        circuit.initialize(state_vector)

        circuit.append(compressor.definition, list(range(n_qubits)))
        circuit.reset(compressor.trash_qubits)
        circuit.append(decompressor.definition, list(range(n_qubits)))

        state = Statevector(circuit)
        
        self.assertTrue(np.allclose(state_vector, state))

    def test_trash(self):
        n_qubits = 6
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        partition = random.sample(range(n_qubits), n_qubits//2)
        compressor = SchmidtCompressor(state_vector, opt_params={'partition': partition})

        circuit = QuantumCircuit(n_qubits)
        circuit.initialize(state_vector)

        circuit.append(compressor.definition, list(range(n_qubits)))

        state = Statevector(circuit).probabilities(list(compressor.trash_qubits))
        reference_state = [1] + [0] * (2**len(compressor.trash_qubits)-1)

        self.assertTrue(np.allclose(reference_state, state))
