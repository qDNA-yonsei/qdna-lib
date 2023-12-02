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
https://arxiv.org/abs/2108.10182
"""

from dataclasses import dataclass
from qiskit.circuit import ParameterExpression
from qiskit.utils import optionals as _optionals
from sympy import sqrt as _sqrt
import symengine

@dataclass
class Node:
    """
    Binary tree node used in state_decomposition function
    """

    index: int
    level: int
    left: "Node"
    right: "Node"
    norm: ParameterExpression

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.norm}"
        )

def _sqrt(self):
    """Square root of a ParameterExpression"""
    if _optionals.HAS_SYMENGINE:
        return self._call(symengine.sqrt)
    else:
        return self._call(_sqrt)

def state_decomposition(nqubits, data):
    """
    :param nqubits: number of qubits required to generate a
                    state with the same length as the data vector (2^nqubits)
    :param data: list with exactly 2^nqubits pairs (index, amplitude)
    :return: root of the state tree
    """
    new_nodes = []

    # leafs
    for i, k in enumerate(data):
        new_nodes.append(
            Node(
                i,
                nqubits,
                None,
                None,
                k
            )
        )

    # build state tree
    while nqubits > 0:
        nodes = new_nodes
        new_nodes = []
        nqubits = nqubits - 1
        k = 0
        n_nodes = len(nodes)
        if nqubits == 0:
            # Optimization.
            # The value of the first node is always 1, so there is no need to
            # calculate it.
            new_nodes.append(
                Node(nodes[k].index // 2, nqubits, nodes[k], nodes[k + 1], 1.0)
            )
        else:
            while k < n_nodes:
                norm = _sqrt(
                    nodes[k].norm*nodes[k].norm + nodes[k + 1].norm*nodes[k + 1].norm
                )

                new_nodes.append(
                    Node(nodes[k].index // 2, nqubits, nodes[k], nodes[k + 1], norm)
                )
                k = k + 2

    tree_root = new_nodes[0]
    return tree_root
