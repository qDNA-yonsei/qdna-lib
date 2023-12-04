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
from sympy import sqrt as sp_sqrt #, Abs as sp_abs
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
    sign: ParameterExpression

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.sign*self.norm}"
        )

# def _abs(self):

#     # SymPy's `Abs` function doesn't work with hybrid optimization backwards.

#     """Square root of a ParameterExpression"""
#     if _optionals.HAS_SYMENGINE:
#         return self._call(symengine.Abs)

#     return self._call(sp_abs)

def _sqrt(self):
    """Square root of a ParameterExpression"""
    if _optionals.HAS_SYMENGINE:
        return self._call(symengine.sqrt)

    return self._call(sp_sqrt)

def _sign(self):
    """Sign of a ParameterExpression"""
    # if _optionals.HAS_SYMENGINE:
    #     return self._call(symengine.sign)

    # return self._call(sp_sign)

    # SymPy's `sign` function doesn't work with hybrid optimization backwards.
    # If self is exactly `-10^-6`, the algorithm is interrupted.
    # This is because the `sign` function will return 0, zeroing the norms.
    return (self + 1e-6) / _sqrt((self + 1e-6)*(self + 1e-6))

def state_decomposition(nqubits, data, normalize=False):
    """
    :param nqubits: number of qubits required to generate a
                    state with the same length as the data vector (2^nqubits)
    :param data: list with exactly 2^nqubits pairs (index, amplitude)
    :return: root of the state tree
    """

    new_nodes = []

    # leafs
    for i, k in enumerate(data):
        sign = _sign(k)
        # If one of the coordinates of the state vector
        # is exactly `-10^-64`, the algorithm breaks.
        # This is because one of the norms will be zero,
        # causing division by zero.
        new_nodes.append(
            Node(
                i,
                nqubits,
                None,
                None,
                k + 1e-64, # prevents division by zero.
                sign
            )
        )

    # normalization
    if normalize:
        norm = 0.0
        for i, k in enumerate(data):
            norm += k*k
        norm = _sqrt(norm)
        for node in new_nodes:
            node.norm = node.norm/norm

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
                Node(
                    nodes[k].index // 2,
                    nqubits,
                    nodes[k],
                    nodes[k + 1],
                    nodes[k].sign,
                    nodes[k].sign
                )
            )
        else:
            while k < n_nodes:
                norm = nodes[k].sign * _sqrt(
                    nodes[k].norm*nodes[k].norm + nodes[k + 1].norm*nodes[k + 1].norm
                )

                new_nodes.append(
                    Node(
                        nodes[k].index // 2,
                        nqubits,
                        nodes[k],
                        nodes[k + 1],
                        norm,
                        nodes[k].sign
                    )
                )
                k = k + 2

    tree_root = new_nodes[0]
    return tree_root
