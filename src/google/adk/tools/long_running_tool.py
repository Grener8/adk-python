# Copyright 2026 Google LLC
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

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

from google.genai import types
from typing_extensions import override

from .function_tool import FunctionTool


class LongRunningFunctionTool(FunctionTool):
  """A function tool that returns the result asynchronously.

  This tool is used for long-running operations that may take a significant
  amount of time to complete. The framework will call the function. Once the
  function returns, the response will be returned asynchronously to the
  framework which is identified by the function_call_id.

  Example:
  ```python
  # Always pause after execution (default behavior):
  tool = LongRunningFunctionTool(a_long_running_function)

  # Conditionally pause based on the tool result:
  def needs_confirmation(result: dict) -> bool:
    return result.get('status') == 'pending_confirmation'

  tool = LongRunningFunctionTool(
      func=a_long_running_function,
      is_long_running=needs_confirmation,
  )
  ```

  Attributes:
    is_long_running: Controls whether execution pauses after the tool runs.
      If ``True`` (the default), the framework always pauses and waits for an
      external signal to resume.  If a callable is provided, it is invoked
      with the tool's return value; the framework pauses only when the
      callable returns ``True``, allowing the tool to skip the pause for
      validation errors or other conditions where the LLM should continue
      immediately.
  """

  def __init__(
      self,
      func: Callable,
      is_long_running: Union[bool, Callable[[Any], bool]] = True,
  ):
    super().__init__(func)
    self.is_long_running = is_long_running

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    declaration = super()._get_declaration()
    if declaration:
      instruction = (
          "\n\nNOTE: This is a long-running operation. Do not call this tool"
          " again if it has already returned some intermediate or pending"
          " status."
      )
      if declaration.description:
        declaration.description += instruction
      else:
        declaration.description = instruction.lstrip()
    return declaration
