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
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RZGate
from qdna.embedding.util.ucr import ucr

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestUcr(TestCase):

    def test_comparison(self):
        n_ctrl_qubits = 2

        target_qubit = n_ctrl_qubits
        control_qubits = list(range(n_ctrl_qubits))

        angles = np.random.rand(2**n_ctrl_qubits) * np.pi

        # Uniformly controlled rotation
        ucr_circuit = QuantumCircuit(n_ctrl_qubits+1)
        ucry = ucr(RZGate, angles)
        ucr_circuit.compose(ucry, [target_qubit, *control_qubits], inplace=True)

        # Sequence of multicontrolled rotations
        mcr_circuit = QuantumCircuit(n_ctrl_qubits+1)
        mcr_00 = RZGate(angles[0]).control(num_ctrl_qubits=n_ctrl_qubits, ctrl_state='00')
        mcr_01 = RZGate(angles[1]).control(num_ctrl_qubits=n_ctrl_qubits, ctrl_state='01')
        mcr_10 = RZGate(angles[2]).control(num_ctrl_qubits=n_ctrl_qubits, ctrl_state='10')
        mcr_11 = RZGate(angles[3]).control(num_ctrl_qubits=n_ctrl_qubits, ctrl_state='11')
        mcr_circuit.compose(mcr_00, [*control_qubits, target_qubit], inplace=True)
        mcr_circuit.compose(mcr_01, [*control_qubits, target_qubit], inplace=True)
        mcr_circuit.compose(mcr_10, [*control_qubits, target_qubit], inplace=True)
        mcr_circuit.compose(mcr_11, [*control_qubits, target_qubit], inplace=True)

        # Compare
        ucr_op = Operator(ucr_circuit).data
        mcr_op = Operator(mcr_circuit).data

        self.assertTrue(np.allclose(ucr_op, mcr_op))
