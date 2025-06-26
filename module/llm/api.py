"""
For the Raspberry Pi desktop assistant's large language model API.
"""

import types
from functools import wraps
from openai import OpenAI


class LLMAPI:
    def __init__(self, api_key=None, base_url=None, max_turns=10, system_prompt=None):
        """
        Initalize the LLM API client.
        :param api_key: API key for authentication
        :param base_url: Base URL for the API (e.g., "https://api.deepseek.com")
        :param max_turns: Maximum number of turns to keep in conversation history
        :param system_prompt: Initial system prompt for the assistant
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_turns = max_turns
        self.system_prompt = {
            "role": "system",
            "content": system_prompt or "You are a helpful assistant.",
        }
        self.messages = []
        self.add_message(self.system_prompt)

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def add_message(self, message):
        """
        Add a message to the conversation history, while controlling the length of multi-turn conversations.
        :param message: Message to add, should be a dict with "role" and "content" keys
        """
        self.messages.append(message)

        # 保持历史记录长度(保留最新的消息)
        while len(self.messages) > self.max_turns:
            # 保留第一条系统消息和其他最新消息
            if self.messages[1]["role"] != "system":
                del self.messages[1]
            else:
                # 如果第二条还是系统消息，删除第三条
                if len(self.messages) > 2:
                    del self.messages[2]

    def get_messages(self):
        """Obtain the current conversation history."""
        return self.messages

    def clear_messages(self, keep_system=True):
        """
        Clear the conversation history.
        :param keep_system: Whether to retain the system prompt in the history
        """
        if keep_system:
            self.messages = []
            self.add_message(self.system_prompt)
        else:
            self.messages = []

    def with_message_history(func):
        """ 
        Decorator to handle message history and non-streaming/streaming responses.
        """

        @wraps(func)
        def wrapper(self, model, user_input, *args, **kwargs):
            # 1. add user input to message history
            self.add_message({"role": "user", "content": user_input})

            # 2. call the wrapped function to get the response
            result = func(self, model, user_input, *args, **kwargs)

            # 3. process the result
            if isinstance(result, types.GeneratorType):
                # streaming response
                def stream_wrapper():
                    full_response = ""
                    try:
                        for content in result:
                            full_response += content
                            yield content
                    finally:
                        # 4. add the full response to message history
                        if full_response:
                            self.add_message(
                                {"role": "assistant", "content": full_response}
                            )

                return stream_wrapper()
            else:
                # non-streaming response
                # non-streaming functions should return (content, message)
                if isinstance(result, tuple) and len(result) == 2:
                    content, message = result
                    self.add_message(message)
                    return content
                else:
                    raise ValueError("Non-streaming function must return a tuple (content, message)")

        return wrapper

    @with_message_history
    def generate_response(self, model, user_input: str):
        """
        Obtain a response from the LLM
        :param    model: model name to use (e.g., "deepseek-chat")
        :param user_input: User input content
        :return: LLM response content and the message object
        """
        response = self.client.chat.completions.create(
            model=model, messages=self.messages, stream=False
        )
        message = response.choices[0].message

        return message.content, message

    @with_message_history
    def generate_stream_response(self, model, user_input: str):
        """
        Obtain a streaming response from the LLM
        :param model: model name to use (e.g., "deepseek-chat")
        :param user_input: User input content
        :return: A generator yielding chunks of the LLM response content
        """
        stream = self.client.chat.completions.create(
            model=model, messages=self.messages, stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield content

    def get_model_list(self):
        return self.client.models.list()
