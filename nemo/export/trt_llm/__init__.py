# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from mpi4py import MPI

try:
    import tensorrt_llm
except Exception as e:
    print(
        "tensorrt_llm package is not installed. Please build or install tensorrt_llm package"
        " properly before calling the llm deployment API."
    )
    raise (e)

from nemo.export.trt_llm.model_config_trt import *  # noqa
from nemo.export.trt_llm.nemo_utils import *  # noqa
from nemo.export.trt_llm.quantization_utils import *  # noqa
from nemo.export.trt_llm.tensorrt_llm_run import *  # noqa
