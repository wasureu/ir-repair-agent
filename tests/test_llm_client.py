import unittest
from types import SimpleNamespace
from unittest import mock

from ir_repair_agent.llm.client import LLMClient


def response(content, reasoning=None):
    usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(message=message)],
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)

    def create(self, **_kwargs):
        return next(self.responses)


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


class LLMClientTests(unittest.TestCase):
    def test_content_and_usage(self):
        usages = []
        client = LLMClient(
            client=FakeClient([response("hello")]),
            model="toy",
            usage_callback=usages.append,
        )
        self.assertEqual(client.complete("prompt"), "hello")
        self.assertEqual(usages[0].total_tokens, 5)

    def test_reasoning_fallback(self):
        client = LLMClient(
            client=FakeClient([response("", "```python\nimport gurobipy\n```")]),
            model="toy",
        )
        self.assertIn("import gurobipy", client.complete("prompt"))

    @mock.patch("ir_repair_agent.llm.client.time.sleep")
    def test_empty_content_retries(self, _sleep):
        client = LLMClient(
            client=FakeClient([response(""), response("retry")]),
            model="toy",
        )
        self.assertEqual(client.complete("prompt"), "retry")


if __name__ == "__main__":
    unittest.main()

