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


def _make_resumable_runner(agent: Agent) -> testing_utils.InMemoryRunner:
  """Returns a runner that has resumability enabled."""
  app = App(
      name='test_app',
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  return testing_utils.InMemoryRunner(app=app)


def test_callable_is_long_running_pauses_when_true():
  """Callable is_long_running=True → execution pauses; model_response_event has long_running_tool_ids."""
  responses = [
      Part.from_function_call(name='submit_order', args={'item': 'book'}),
      'confirmed',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def submit_order(item: str) -> dict:
    return {'status': 'pending_confirmation', 'item': item}

  def needs_pause(result: dict) -> bool:
    return result.get('status') == 'pending_confirmation'

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[
          LongRunningFunctionTool(func=submit_order, is_long_running=needs_pause)
      ],
  )
  runner = _make_resumable_runner(agent)
  events = runner.run('order book')

  # Only the first LLM call should have been made; the invocation paused
  # before making a second call because the callable returned True.
  assert len(mock_model.requests) == 1

  # The model-response event (function call) must carry long_running_tool_ids.
  fc_event = events[0]
  assert fc_event.long_running_tool_ids, (
      'model_response_event must have long_running_tool_ids set when callable'
      ' is_long_running returns True'
  )

  # The events consist of the function-call event and the function-response
  # event (tool executed and returned a result).
  assert testing_utils.simplify_events(events) == [
      (
          'root_agent',
          Part.from_function_call(name='submit_order', args={'item': 'book'}),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='submit_order',
              response={'status': 'pending_confirmation', 'item': 'book'},
          ),
      ),
  ]


def test_callable_is_long_running_no_pause_when_false():
  """Callable is_long_running returning False → no pause; LLM receives error."""
  responses = [
      Part.from_function_call(name='validate_input', args={'value': -1}),
      'Please provide a positive number.',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def validate_input(value: int) -> dict:
    if value < 0:
      return {'error': 'value must be non-negative'}
    return {'status': 'ok', 'value': value}

  def needs_pause(result: dict) -> bool:
    # Only pause on success; propagate errors back to the LLM immediately.
    return result.get('status') == 'ok'

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[
          LongRunningFunctionTool(
              func=validate_input, is_long_running=needs_pause
          )
      ],
  )
  runner = _make_resumable_runner(agent)
  events = runner.run('check -1')

  # Both LLM calls must happen: one for the function call and one after the
  # LLM sees the error response (no pause occurred).
  assert len(mock_model.requests) == 2

  # The model-response event must NOT have long_running_tool_ids because the
  # callable returned False for this result.
  fc_event = events[0]
  assert not fc_event.long_running_tool_ids, (
      'model_response_event must NOT have long_running_tool_ids when callable'
      ' is_long_running returns False'
  )

  # LLM received the error and produced a text response.
  assert testing_utils.simplify_events(events) == [
      (
          'root_agent',
          Part.from_function_call(
              name='validate_input', args={'value': -1}
          ),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='validate_input',
              response={'error': 'value must be non-negative'},
          ),
      ),
      ('root_agent', 'Please provide a positive number.'),
  ]


def test_callable_is_long_running_backward_compat():
  """Boolean is_long_running=True still works the same as before."""
  responses = [
      Part.from_function_call(name='increase_by_one', args={'x': 1}),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def increase_by_one(x: int, tool_context: ToolContext) -> int:
    return {'status': 'pending'}

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=increase_by_one)],
  )
  runner = _make_resumable_runner(agent)
  events = runner.run('test1')

  # With boolean True and resumability enabled, the invocation pauses after
  # the first LLM call (no second LLM call in this turn).
  assert len(mock_model.requests) == 1
  assert events[0].long_running_tool_ids
