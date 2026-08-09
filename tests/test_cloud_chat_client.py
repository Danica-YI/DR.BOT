import base64
import unittest

from robot.cloud_chat_client import CloudChatClient


class FakeResponse:
    def __init__(self, ok=True, status_code=200, body=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.response


class CloudChatClientTests(unittest.TestCase):
    def test_ask_text_posts_openai_compatible_payload(self):
        session = FakeSession(
            FakeResponse(
                body={
                    "choices": [
                        {"message": {"content": "hello from cloud"}}
                    ]
                }
            )
        )
        client = CloudChatClient(
            enabled=True,
            api_key="test-key",
            base_url="https://example.com/openai",
            model="gemini-test",
            session=session,
        )

        result = client.ask_text("hello", system_prompt="be concise")

        self.assertTrue(result.success)
        self.assertEqual(result.text, "hello from cloud")
        self.assertEqual(session.calls[0]["url"], "https://example.com/openai/chat/completions")
        self.assertEqual(session.calls[0]["json"]["model"], "gemini-test")
        self.assertEqual(session.calls[0]["json"]["messages"][0]["role"], "system")
        self.assertEqual(session.calls[0]["json"]["messages"][1]["content"], "hello")

    def test_ask_audio_encodes_wav_as_input_audio(self):
        session = FakeSession(
            FakeResponse(
                body={
                    "choices": [
                        {"message": {"content": "audio answer"}}
                    ]
                }
            )
        )
        client = CloudChatClient(enabled=True, api_key="test-key", session=session)

        result = client.ask_audio(b"wav-bytes", instruction="answer this")

        self.assertTrue(result.success)
        message = session.calls[0]["json"]["messages"][0]
        audio_part = message["content"][1]
        self.assertEqual(audio_part["type"], "input_audio")
        self.assertEqual(
            audio_part["input_audio"]["data"],
            base64.b64encode(b"wav-bytes").decode("ascii"),
        )
        self.assertEqual(audio_part["input_audio"]["format"], "wav")

    def test_http_error_is_returned(self):
        session = FakeSession(FakeResponse(ok=False, status_code=429, text="quota exceeded"))
        client = CloudChatClient(enabled=True, api_key="test-key", session=session)

        result = client.ask_text("hello")

        self.assertFalse(result.success)
        self.assertIn("HTTP 429", result.error)


if __name__ == "__main__":
    unittest.main()
