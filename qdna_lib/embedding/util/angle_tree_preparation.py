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
from qclib.state_preparation.util.tree_utils import is_leaf
from qiskit.circuit import ParameterExpression

@dataclass
class NodeAngleTree:
    """
    Binary tree node used in function create_angles_tree
    """

    index: int
    level: int
    angle_y: ParameterExpression
    left: "NodeAngleTree"
    right: "NodeAngleTree"

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.angle_y}"
        )


def create_angles_tree(state_tree):
    """
    :param state_tree: state_tree is an output of state_decomposition function
    :param tree: used in the recursive calls
    :return: tree with angles that will be used to perform the state preparation
    """
    
    amplitude = state_tree.right.norm / state_tree.norm
    angle_y = 2 * amplitude.arcsin()

    node = NodeAngleTree(
        state_tree.index, state_tree.level, angle_y, None, None
    )

    if not is_leaf(state_tree.left):
        node.right = create_angles_tree(state_tree.right)
        node.left = create_angles_tree(state_tree.left)

    return node
