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

from typing import Callable, List, Union, Optional
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qdna_lib.embedding.nqe_base import NqeBase


class NqeZZFeatureMap(ZZFeatureMap, NqeBase):

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "full",
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "NqeZZFeatureMap",
        nn = None,
    ) -> None:

        super().__init__(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name
        )

        NqeBase.__init__(self, nn=nn)

        self._feature_map = ZZFeatureMap
        self._init_parameters = {
            'feature_dimension':feature_dimension,
            'reps':reps,
            'entanglement':entanglement,
            'data_map_func':data_map_func,
            'parameter_prefix':parameter_prefix,
            'insert_barriers':insert_barriers,
            'name':name
        }
