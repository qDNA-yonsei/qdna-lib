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
"""

from __future__ import annotations

from typing import Sequence, Mapping

import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression

from qiskit.circuit.library import BlueprintCircuit, RYGate, RZGate

from qdna_lib.embedding.util.state_tree_preparation import state_decomposition
from qdna_lib.embedding.util.angle_tree_preparation import create_angles_tree
from qdna_lib.embedding.util.ucr import ucr

#from qclib.gates.ucr import ucr
from qclib.state_preparation.util.tree_utils import children

class AeFeatureMap(BlueprintCircuit):
    """The AeFeatureMap circuit class.

        TODO: documentation
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        reps: int = 1,
        insert_barriers: bool = False,
        parameter_prefix: str = "x",
        name: str | None = "AeFeatureMap"
    ) -> None:
        
        super().__init__(name=name)

        self._num_qubits: int | None = None
        self._insert_barriers = insert_barriers
        self._reps = reps
        self._parameter_prefix=parameter_prefix
        self._initial_state: QuantumCircuit | None = None
        self._initial_state_circuit: QuantumCircuit | None = None
        self._bounds: list[tuple[float | None, float | None]] | None = None

        if int(reps) != reps:
            raise TypeError("The value of reps should be int.")

        if reps < 1:
            raise ValueError("The value of reps should be larger than or equal to 1.")

        if num_qubits is not None:
            self.num_qubits = num_qubits


    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return self._num_qubits if self._num_qubits is not None else 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the qcnn circuit.

        Args:
            The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = num_qubits
            self.qregs = [QuantumRegister(num_qubits, name="q")]

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the circuit.

        Returns:
            The number of layers in the circuit.
        """

        # A pool and a convolutional layer per iteration.
        # Multiply all by the number of repetitions.
        return self.reps * self.num_qubits

    @property
    def insert_barriers(self) -> bool:
        """If barriers are inserted in between the layers or not.

        Returns:
            ``True``, if barriers are inserted in between the layers, ``False`` if not.
        """
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        """Specify whether barriers should be inserted in between the layers or not.

        Args:
            insert_barriers: If True, barriers are inserted, if False not.
        """
        # if insert_barriers changes, we have to invalidate the circuit definition,
        # if it is the same as before we can leave the instance as it is
        if insert_barriers is not self._insert_barriers:
            self._invalidate()
            self._insert_barriers = insert_barriers

    @property
    def reps(self) -> int:
        """The number of times the circuit is repeated.

        Returns:
            The number of repetitions.
        """
        return self._reps

    @reps.setter
    def reps(self, repetitions: int) -> None:
        """Set the repetitions.

        Args:
            repetitions: The new repetitions.

        Raises:
            ValueError: If reps setter has parameter repetitions < 1.
        """
        if repetitions < 1:
            raise ValueError("The repetitions should be larger than or equal to 1")
        if repetitions != self._reps:
            self._invalidate()
            self._reps = repetitions

    def print_settings(self) -> str:
        """Returns information about the setting.

        Returns:
            The class name and the attributes/parameters of the instance as ``str``.
        """
        ret = f"Qcnn: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the configuration of the class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the number of repetitions is not set.
            ValueError: If the number of qubits is not set.
        """
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")

        if self.reps is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of repetitions specified.")

        return valid

    @property
    def initial_state(self) -> QuantumCircuit:
        """Return the initial state that is added in front of the circuit.

        Returns:
            The initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        """Set the initial state.

        Args:
            initial_state: The new initial state.

        Raises:
            ValueError: If the number of qubits has been set before and the initial state
                does not match the number of qubits.
        """
        self._initial_state = initial_state
        self._invalidate()

    def assign_parameters(
        self,
        parameters: Mapping[Parameter, ParameterExpression | float]
        | Sequence[ParameterExpression | float],
        inplace: bool = False,
    ) -> QuantumCircuit | None:
        """Assign parameters to the circuit.

        This method also supports passing a list instead of a dictionary. If a list
        is passed, the list must have the same length as the number of unbound parameters in
        the circuit.

        Returns:
            A copy of the circuit with the specified parameters.

        Raises:
            AttributeError: If the parameters are given as list and do not match the number
                of parameters.
        """
        if parameters is None or len(parameters) == 0:
            return self

        if not self._is_built:
            self._build()

        return super().assign_parameters(parameters, inplace=inplace)

    @property
    def num_parameters_settable(self) -> int:
        """The number of total parameters that can be set to distinct values.

        This does not change when the parameters are bound or exchanged for same parameters,
        and therefore is different from ``num_parameters`` which counts the number of unique
        :class:`~qiskit.circuit.Parameter` objects currently in the circuit.

        Returns:
            The number of parameters originally available in the circuit.

        Note:
            This quantity does not require the circuit to be built yet.
        """

        return 2**self.num_qubits
    

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        if self.num_qubits == 0:
            return

        circuit = QuantumCircuit(*self.qregs, name=self.name)
        params = ParameterVector(self._parameter_prefix, length=self.num_parameters_settable)

        state_tree = state_decomposition(self.num_qubits, params)
        angle_tree = create_angles_tree(state_tree)

        nodes = [angle_tree]
        control_qubits = []
        target_qubit = self.num_qubits - 1

        while len(nodes) > 0:
            angles = [node.angle_y for node in nodes]
            ucry = ucr(RYGate, angles)
            circuit.append(ucry, [target_qubit] + control_qubits[::-1])
            control_qubits.append(target_qubit)
            nodes = children(nodes)
            target_qubit -= 1

            if self._insert_barriers and target_qubit >= 0:
                circuit.barrier()

        for _ in range(self._reps):
            self.append(circuit, self.qubits)
