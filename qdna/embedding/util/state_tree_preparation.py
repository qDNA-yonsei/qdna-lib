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
from sympy import sqrt as sp_sqrt
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
#     """Absolute value of a ParameterExpression"""
#     if _optionals.HAS_SYMENGINE:
#         import symengine
#         return ParameterExpression(self._parameter_symbols, symengine.Abs(self._symbol_expr))
#     else:
#         from sympy import Abs as sp_abs
#         return ParameterExpression(self._parameter_symbols, sp_abs(self._symbol_expr))

def _sqrt(self):
    """Square root of a ParameterExpression"""
    if _optionals.HAS_SYMENGINE:
        return self._call(symengine.sqrt)

    return self._call(sp_sqrt)

def _sign(self):
    # SymPy's `sign` function doesn't work with hybrid optimization backwards.
    """Sign of a ParameterExpression"""
    # if _optionals.HAS_SYMENGINE:
    #     import symengine
    #     return ParameterExpression(
    #         self._parameter_symbols, symengine.sign(self._symbol_expr + 1e-4)
    #     )
    # else:
    #     from sympy import sign
    #     return ParameterExpression(
    #         self._parameter_symbols, sign(self._symbol_expr + 1e-4)
    #     )

    # If self is exactly `-10^-4`, the algorithm is interrupted.
    # This is because the `sign` function will return 0, zeroing the norms.
    return self / _sqrt(self*self)

def state_decomposition(nqubits, data, normalize=False):
    """
    :param nqubits: number of qubits required to generate a
                    state with the same length as the data vector (2^nqubits)
    :param data: list with exactly 2^nqubits pairs (index, amplitude)
    :return: root of the state tree
    """

    new_nodes = []

    # leafs
    r = 10**-6
    for i, k in enumerate(data):
        # If one of the coordinates of the state vector
        # is exactly `-r*10^-6`, the algorithm breaks.
        # This is because one of the norms will be zero,
        # causing division by zero.
        k = k + r # prevents division by zero.
        sign = _sign(k)
        new_nodes.append(
            Node(
                i,
                nqubits,
                None,
                None,
                k,
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
