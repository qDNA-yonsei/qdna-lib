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

from qdna.embedding.nqe_base import NqeBase
from qdna.embedding.ae_feature_map import AeFeatureMap


class NqeAeFeatureMap(AeFeatureMap, NqeBase):

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "NqeAeFeatureMap",
        nn = None,
    ) -> None:

        super().__init__(
            num_qubits=num_qubits,
            reps=reps,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name
        )

        NqeBase.__init__(self, nn=nn)

        self._feature_map = AeFeatureMap
        self._init_parameters = {
            'num_qubits':num_qubits,
            'reps':reps,
            'parameter_prefix':parameter_prefix,
            'insert_barriers':insert_barriers,
            'name':name
        }
