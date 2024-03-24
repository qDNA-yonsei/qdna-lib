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

import numpy as np

from qiskit import QuantumCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

import torch
from torch import optim
from torch.nn import (
    Module,
    Linear,
    MSELoss,
    Sequential,
    ReLU
)

# pylint: disable=maybe-no-member

class _Transform(Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x):
        x = self.nn(x)
        return x.detach().numpy()

class _NetFidelity(Module):
    def __init__(self, qnn, nn):
        super().__init__()
        self.nn = nn
        self.qnn = TorchConnector(qnn)

    def forward(self, x1, x2):
        x1 = self.nn(x1)
        x2 = self.nn(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qnn(x)

        return x[:,0]


class NqeBase():

    def __init__(self, nn) -> None:
        self.nn = nn
        self.model = None
        self._transform = None


    def fit(
        self,
        x, y,
        batch_size=25,
        iters=100,
        optimizer=None,
        loss_func=None,
        distance='fidelity',
        sampler=None,
        verbose=1
    ):
        if self.nn is None:
            self.nn = Sequential(
                Linear(
                    self.num_parameters_settable,
                    self.num_parameters_settable*3
                ),
                ReLU(),
                Linear(
                    self.num_parameters_settable*3,
                    self.num_parameters_settable*3
                ),
                ReLU(),
                Linear(
                    self.num_parameters_settable*3,
                    self.num_parameters_settable
                )
            )

        qnn = self.create_qnn(distance, sampler)
        if distance == 'fidelity': # only 'fidelity' for now.
            model = _NetFidelity(qnn, self.nn)

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
        if loss_func is None:
            loss_func = MSELoss()

        # Start training
        loss_list = []  # Store loss history
        model.train()   # Set model to training mode

        for i in range(iters):
            # Random sampling of data.
            x1_batch, x2_batch, y_batch = self.new_data(batch_size, x, y)

            try:
                optimizer.zero_grad(set_to_none=True) # Initialize gradient
                output = model(x1_batch, x2_batch)    # Forward pass
                loss = loss_func(output, y_batch)     # Calculate loss
                loss.backward()                       # Backward pass
                optimizer.step()                      # Optimize weights

                loss_list.append(loss.item())         # Store loss
            except QiskitMachineLearningError:
                loss_list.append(loss_list[-1])

            if verbose:
                print(
                    f"Training [{100.0 * (i + 1) / iters:.0f}%]\t"
                    f"Loss: {loss_list[-1]:.4f}\t"
                    f"Avg. Loss: {sum(loss_list) / len(loss_list):.4f}"
                )

        self.model = model

        state_dict = self.model.state_dict()
        if "qnn.weight" in state_dict:
            state_dict.pop("qnn.weight")
        if "qnn._weights" in state_dict:
            state_dict.pop("qnn._weights")
        self.transform.load_state_dict(state_dict)


    @property
    def transform(self):
        if self._transform is None:
            self._transform = _Transform(self.nn)
        return self._transform


    # Define and create QNN
    def create_qnn(self, distance, sampler):
        oririnal_prefix = self._init_parameters['parameter_prefix']

        self._init_parameters['parameter_prefix'] = 'x'
        feature_map = self._feature_map(**self._init_parameters)
        self._init_parameters['parameter_prefix'] = 'y'
        feature_map_inv = self._feature_map(**self._init_parameters).inverse()

        self._init_parameters['parameter_prefix'] = oririnal_prefix

        if distance == 'hs':
            qc = QuantumCircuit(self.num_qubits + 1)
            qc.h(0)
            qc.compose(
                feature_map.control(1, label='map'),
                range(self.num_qubits+1),
                inplace=True
            )
            qc.compose(
                feature_map_inv.control(1, label='map_inv'),
                range(self.num_qubits+1),
                inplace=True
            )
            qc.h(0)
        else:
            qc = QuantumCircuit(self.num_qubits)
            qc.compose(
                feature_map,
                inplace=True
            )
            qc.compose(
                feature_map_inv,
                inplace=True
            )

        qnn = SamplerQNN(
            sampler=sampler,
            circuit=qc,
            input_params=qc.parameters,
            input_gradients=True,
        )

        return qnn


    @staticmethod
    def new_data(batch_size, x, y):
        x1_new, x2_new, y_new = [], [], []
        for _ in range(batch_size):
            n, m = np.random.randint(len(x)), np.random.randint(len(x))
            x1_new.append(x[n])
            x2_new.append(x[m])
            if y[n] == y[m]:
                y_new.append(1)
            else:
                y_new.append(0)

        return (
            torch.tensor(np.array(x1_new), dtype=torch.float32),
            torch.tensor(np.array(x2_new), dtype=torch.float32),
            torch.tensor(np.array(y_new), dtype=torch.float32)
        )
