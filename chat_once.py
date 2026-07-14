import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "deepseek-v4-flash"
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key is None:
    raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")
base_url = os.getenv("DASHSCOPE_BASE_URL")
if base_url is None:
    raise ValueError("DASHSCOPE_BASE_URL environment variable is not set.")

def ask_model(prompt: str) -> str:
    """Send one synchronous text request and return the model's answer."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    model = os.getenv("DASHSCOPE_MODEL", DEFAULT_MODEL)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = completion.choices[0].message.content
    if answer is None:
        raise ValueError("Received empty response from the model.")
    return answer


def main() -> None:
    prompt = input("请输入你的问题：").strip()
    answer = ask_model(prompt)
    print(f"\n模型回答：\n{answer}")


if __name__ == "__main__":
    main()
