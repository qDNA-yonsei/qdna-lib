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
Implements the quantum state compressor defined at ... .
"""

from qiskit import QuantumCircuit
from qclib.unitary import unitary as decompose_unitary
from qclib.isometry import decompose as decompose_isometry
from qclib.entanglement import schmidt_decomposition, _to_qubits
from qdna.compression.compressor import Compressor

# pylint: disable=maybe-no-member


class SchmidtCompressor(Compressor):
    """
    
    """

    def __init__(self, params, label=None, opt_params=None):
        """
        Parameters
        ----------
        params: list of complex
            A unit vector representing a quantum state.
            Values are amplitudes.

        opt_params: {'lr': low_rank,
                     'iso_scheme': isometry_scheme,
                     'unitary_scheme': unitary_scheme,
                     'partition': partition}
            low_rank: int
                ``state`` low-rank approximation (1 <= ``low_rank`` < 2**(n_qubits//2)).
                If ``low_rank`` is not in the valid range, it will be ignored.
                This parameter limits the rank of the Schmidt decomposition. If the Schmidt rank
                of the state decomposition is greater than ``low_rank``, a low-rank
                approximation is applied.

            iso_scheme: string
                Scheme used to decompose isometries.
                Possible values are ``'knill'`` and ``'ccd'`` (column-by-column decomposition).
                Default is ``isometry_scheme='ccd'``.

            unitary_scheme: string
                Scheme used to decompose unitaries.
                Possible values are ``'csd'`` (cosine-sine decomposition) and ``'qsd'`` (quantum
                Shannon decomposition).
                Default is ``unitary_scheme='qsd'``.

            partition: list of int
                Set of qubit indices that represent a part of the bipartition.
                The other partition will be the relative complement of the full set of qubits
                with respect to the set ``partition``.
                The valid range for indexes is ``0 <= index < n_qubits``. The number of indexes
                in the partition must be greater than or equal to ``1`` and less than or equal
                to ``n_qubits//2`` (``n_qubits//2+1`` if ``n_qubits`` is odd).
                Default is ``partition=list(range(n_qubits//2 + odd))``.

            svd: string
                Function to compute the SVD, acceptable values are 'auto' (default), 'regular',
                and 'randomized'. 'auto' sets `svd='randomized'` for `n_qubits>=14 and rank==1`.
        """
        self._name = "compressor"
        self._get_num_qubits(params)

        if opt_params is None:
            self.isometry_scheme = "ccd"
            self.unitary_scheme = "qsd"
            self.low_rank = 0
            self.partition = self._default_partition(self.num_qubits)
            self.svd = "auto"
        else:
            self.low_rank = 0 if opt_params.get("lr") is None else \
                opt_params.get("lr")
            self.partition = self._default_partition(self.num_qubits) if \
                opt_params.get("partition") is None else sorted(opt_params.get("partition"))
            self.isometry_scheme = "ccd" if opt_params.get("iso_scheme") is None else \
                opt_params.get("iso_scheme")
            self.unitary_scheme = "qsd" if opt_params.get("unitary_scheme") is None else \
                opt_params.get("unitary_scheme")
            self.svd = "auto" if opt_params.get("svd") is None else \
                opt_params.get("svd")

        # The trash and latent qubits must take into account that the qiskit qubits are reversed.
        complement = sorted(
            set(range(self.num_qubits)).difference(set(self.partition))
        )[::-1]
        self.latent_qubits = [
            self.num_qubits-i-1 for i in complement
        ]
        self.trash_qubits = sorted(
            set(range(self.num_qubits)).difference(set(self.latent_qubits))
        )

        if label is None:
            label = "COMPRESSOR"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):

        if self.num_qubits < 2:
            circuit = QuantumCircuit(1)
            circuit.initialize(self.params)
            return  circuit.inverse()

        circuit, reg_a, reg_b = self._create_quantum_circuit()

        # Schmidt decomposition
        rank, svd_u, _, svd_v = schmidt_decomposition(
            self.params, reg_a, rank=self.low_rank, svd=self.svd
        )

        # Schmidt measure of entanglement
        e_bits = _to_qubits(rank)

        # Phase 3 and 4 encode gates U and V.T
        self._encode(svd_u, circuit, reg_b)
        self._encode(svd_v.T, circuit, reg_a)

        # Phase 2. Entangles only the necessary qubits, according to rank.
        for j in range(e_bits):
            circuit.cx(reg_b[j], reg_a[j])

        return circuit.reverse_bits()

    def _encode(self, data, circuit, reg):
        """
        Encodes data using the most appropriate method.
        """
        if data.shape[1] == 1:
            # state preparation
            gate_u = SchmidtCompressor(data[:, 0], opt_params={
                "iso_scheme": self.isometry_scheme,
                "unitary_scheme": self.unitary_scheme,
                "svd": self.svd
            })

        elif data.shape[0] // 2 == data.shape[1]:
            # isometry 2^(n-1) to 2^n.
            gate_u = decompose_isometry(data, scheme="csd")

        elif data.shape[0] > data.shape[1]:
            gate_u = decompose_isometry(data, scheme=self.isometry_scheme)

        else:
            gate_u = decompose_unitary(data, decomposition=self.unitary_scheme)

        # Apply gate U to the register reg
        circuit.compose(gate_u.inverse(), reg, inplace=True)

    @staticmethod
    def _default_partition(n_qubits):
        odd = n_qubits % 2
        return list(range(n_qubits // 2 + odd))

    def _create_quantum_circuit(self):

        if self.partition is None:
            self.partition = self._default_partition(self.num_qubits)

        complement = sorted(set(range(self.num_qubits)).difference(set(self.partition)))

        circuit = QuantumCircuit(self.num_qubits, name=self.label)

        return circuit, self.partition[::-1], complement[::-1]
