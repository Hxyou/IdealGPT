
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=0.1, max=0.2), stop=stop_after_attempt(10))
def call_gpt(chatgpt_messages, model="gpt-3.5-turbo", temp_gpt=0.0):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=temp_gpt, max_tokens=512)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

