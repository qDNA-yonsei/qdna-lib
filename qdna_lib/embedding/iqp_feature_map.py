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

"""The Pauli expansion circuit module."""

from typing import Callable, List, Union
from math import comb
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library.standard_gates import HGate

from qiskit.circuit.library import NLocal


class IqpFeatureMap(NLocal):

    def __init__(
        self,
        num_qubits: int,
        reps: int = 2,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "full",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "IqpFeatureMap",
    ) -> None:

        super().__init__(
            num_qubits=num_qubits,
            reps=reps,
            rotation_blocks=HGate(),
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            skip_final_rotation_layer=True,
            name=name,
        )

        self._paulis = ["Z", "ZZ"]
        self._alpha = 2.0

    @property
    def num_parameters_settable(self):
        """The number of distinct parameters."""
        return self.num_qubits + comb(self.num_qubits, 2)


    @property
    def entanglement_blocks(self):
        return [self.pauli_block(pauli) for pauli in self._paulis]

    @entanglement_blocks.setter
    def entanglement_blocks(self, entanglement_blocks):
        self._entanglement_blocks = entanglement_blocks

    @property
    def feature_dimension(self) -> int:
        """Returns the feature dimension.

        Returns:
            The feature dimension of this feature map.
        """
        return self.num_qubits

    @feature_dimension.setter
    def feature_dimension(self, feature_dimension: int) -> None:
        """Set the feature dimension.

        Args:
            feature_dimension: The new feature dimension.
        """
        self.num_qubits = feature_dimension

    def pauli_block(self, pauli_string):
        """Get the Pauli block for the feature map circuit."""
        params = ParameterVector("_", length=1)
        time = np.asarray(params)[0]
        return self.pauli_evolution(pauli_string, time)


    def pauli_evolution(self, pauli_string, time):
        """Get the evolution block for the given pauli string."""

        indices = list(range(len(pauli_string)))

        evo = QuantumCircuit(len(pauli_string))

        def cx_chain(circuit, inverse=False):
            num_cx = len(indices) - 1
            for i in reversed(range(num_cx)) if inverse else range(num_cx):
                circuit.cx(indices[i], indices[i + 1])

        cx_chain(evo)
        evo.p(self._alpha * time, indices[-1])
        cx_chain(evo, inverse=True)

        return evo
