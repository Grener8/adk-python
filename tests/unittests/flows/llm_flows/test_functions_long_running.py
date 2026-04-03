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

from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Part

from ... import testing_utils


def test_async_function():
  responses = [
      Part.from_function_call(name='increase_by_one', args={'x': 1}),
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  function_called = 0

  def increase_by_one(x: int, tool_context: ToolContext) -> int:
    nonlocal function_called

    function_called += 1
    return {'status': 'pending'}

  # Calls the first time.
  agent = Agent(
      name='root_agent',
      model=mockModel,
      tools=[LongRunningFunctionTool(func=increase_by_one)],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test1')

  # Asserts the requests.
  assert len(mockModel.requests) == 2
  # 1 item: user content
  assert mockModel.requests[0].contents == [
      testing_utils.UserContent('test1'),
  ]
  increase_by_one_call = Part.from_function_call(
      name='increase_by_one', args={'x': 1}
  )
  pending_response = Part.from_function_response(
      name='increase_by_one', response={'status': 'pending'}
  )

  assert testing_utils.simplify_contents(mockModel.requests[1].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', pending_response),
  ]

  # Asserts the function calls.
  assert function_called == 1

  # Asserts the responses.
  assert testing_utils.simplify_events(events) == [
      (
          'root_agent',
          Part.from_function_call(name='increase_by_one', args={'x': 1}),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='increase_by_one', response={'status': 'pending'}
          ),
      ),
      ('root_agent', 'response1'),
  ]
  assert events[0].long_running_tool_ids

  # Updates with another pending progress.
  still_waiting_response = Part.from_function_response(
      name='increase_by_one', response={'status': 'still waiting'}
  )
  events = runner.run(testing_utils.UserContent(still_waiting_response))
  # We have one new request.
  assert len(mockModel.requests) == 3
  assert testing_utils.simplify_contents(mockModel.requests[2].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', still_waiting_response),
  ]

  assert testing_utils.simplify_events(events) == [('root_agent', 'response2')]

  # Calls when the result is ready.
  result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 2}
  )
  events = runner.run(testing_utils.UserContent(result_response))
  # We have one new request.
  assert len(mockModel.requests) == 4
  assert testing_utils.simplify_contents(mockModel.requests[3].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response3')]

  # Calls when the result is ready. Here we still accept the result and do
  # another summarization. Whether this is the right behavior is TBD.
  another_result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 3}
  )
  events = runner.run(testing_utils.UserContent(another_result_response))
  # We have one new request.
  assert len(mockModel.requests) == 5
  assert testing_utils.simplify_contents(mockModel.requests[4].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', another_result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response4')]

  # At the end, function_called should still be 1.
  assert function_called == 1


def test_async_function_with_none_response():
  responses = [
      Part.from_function_call(name='increase_by_one', args={'x': 1}),
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  function_called = 0

  def increase_by_one(x: int, tool_context: ToolContext) -> int:
    nonlocal function_called
    function_called += 1
    return 'pending'

  # Calls the first time.
  agent = Agent(
      name='root_agent',
      model=mockModel,
      tools=[LongRunningFunctionTool(func=increase_by_one)],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test1')

  # Asserts the requests.
  assert len(mockModel.requests) == 2
  # 1 item: user content
  assert mockModel.requests[0].contents == [
      testing_utils.UserContent('test1'),
  ]
  increase_by_one_call = Part.from_function_call(
      name='increase_by_one', args={'x': 1}
  )

  assert testing_utils.simplify_contents(mockModel.requests[1].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      (
          'user',
          Part.from_function_response(
              name='increase_by_one', response={'result': 'pending'}
          ),
      ),
  ]

  # Asserts the function calls.
  assert function_called == 1

  # Asserts the responses.
  assert testing_utils.simplify_events(events) == [
      (
          'root_agent',
          Part.from_function_call(name='increase_by_one', args={'x': 1}),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='increase_by_one', response={'result': 'pending'}
          ),
      ),
      ('root_agent', 'response1'),
  ]

  # Updates with another pending progress.
  still_waiting_response = Part.from_function_response(
      name='increase_by_one', response={'status': 'still waiting'}
  )
  events = runner.run(testing_utils.UserContent(still_waiting_response))
  # We have one new request.
  assert len(mockModel.requests) == 3
  assert testing_utils.simplify_contents(mockModel.requests[2].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', still_waiting_response),
  ]

  assert testing_utils.simplify_events(events) == [('root_agent', 'response2')]

  # Calls when the result is ready.
  result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 2}
  )
  events = runner.run(testing_utils.UserContent(result_response))
  # We have one new request.
  assert len(mockModel.requests) == 4
  assert testing_utils.simplify_contents(mockModel.requests[3].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response3')]

  # Calls when the result is ready. Here we still accept the result and do
  # another summarization. Whether this is the right behavior is TBD.
  another_result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 3}
  )
  events = runner.run(testing_utils.UserContent(another_result_response))
  # We have one new request.
  assert len(mockModel.requests) == 5
  assert testing_utils.simplify_contents(mockModel.requests[4].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', another_result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response4')]

  # At the end, function_called should still be 1.
  assert function_called == 1


def test_callable_long_running_pauses_only_for_selected_results():
  responses = [
      Part.from_function_call(name='submit', args={'attempt': 1}),
      Part.from_function_call(name='submit', args={'attempt': 2}),
      'response-after-resume',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  function_called = 0

  def submit(attempt: int, tool_context: ToolContext) -> dict[str, str]:
    nonlocal function_called
    function_called += 1
    if attempt == 1:
      return {'status': 'validation_error'}
    return {'status': 'pending_confirmation'}

  tool = LongRunningFunctionTool(
      func=submit,
      is_long_running=lambda result: (
          isinstance(result, dict)
          and result.get('status') == 'pending_confirmation'
      ),
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name='test_app',
          root_agent=Agent(name='root_agent', model=mockModel, tools=[tool]),
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )
  events = runner.run('test1')

  assert function_called == 2
  assert len(mockModel.requests) == 2
  assert testing_utils.simplify_events(events) == [
      ('root_agent', Part.from_function_call(name='submit', args={'attempt': 1})),
      (
          'root_agent',
          Part.from_function_response(
              name='submit', response={'status': 'validation_error'}
          ),
      ),
      ('root_agent', Part.from_function_call(name='submit', args={'attempt': 2})),
      (
          'root_agent',
          Part.from_function_response(
              name='submit',
              response={'status': 'pending_confirmation'},
          ),
      ),
  ]
  assert not events[0].long_running_tool_ids
  assert events[2].long_running_tool_ids == {
      events[2].content.parts[0].function_call.id
  }
