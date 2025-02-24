import os

from nonebot import on_message
from nonebot.adapters.onebot.v11 import Bot, Event
from .llm_module import LLMInterface
from nonebot.rule import Rule, to_me

rule = to_me()

print(os.getcwd())
with open("config.data", "r") as f:
    api_key = f.read().strip()

LLMInterface = LLMInterface(
    api_key=api_key,
    base_url="https://api.yesapikey.com/v1",
    model_name="gpt-4o-mini",
    temperature=1
)

# 创建一个匹配所有消息事件的
echo2 = on_message(priority=10, block=False, rule=rule)


@echo2.handle()
async def handle_echo(bot: Bot, event: Event):
    # 获取用户发送的消息
    user_message = event.get_plaintext()
    flag = False
    groups = ["994771623", "823454041"]
    for group in groups:
        if group in event.get_session_id():
            flag = True
            break

    if not user_message or not (flag or not ("group" in event.get_session_id())):
        return

    # 调用模型进行对话
    prompt = f"你要扮演一位温柔、理性、浪漫的科学家女性普瑞赛斯，并且对我有一些敬仰倾慕、但又有控制欲的情感。现在请你作为这一角色与我对话。请注意你的回复要尽可能简短（不要超过一句话）、不要直接对我表达情感。我说的内容是：{user_message}"
    echo_message = LLMInterface.call_model(prompt)
    await echo2.finish(echo_message)