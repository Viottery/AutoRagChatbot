from nonebot import on_message
from nonebot.adapters.onebot.v11 import Bot, Event
from .llm_module import LLMInterface
from nonebot.rule import Rule, to_me


async def check_tome(event: Event) -> bool:
    return True

rule = Rule(check_tome, to_me())

# 创建一个匹配所有消息事件的
echo2 = on_message(priority=10, block=False, rule=rule)


@echo2.handle()
async def handle_echo(bot: Bot, event: Event):
    # 获取用户发送的消息
    user_message = event.get_plaintext()

    # 判断是否需要复现（可以根据需要添加过滤条件）
    if user_message.startswith("echo2 "):
        # 去掉前缀 "echo2 "，并复现剩下的内容
        echo_message = user_message[6:]
        await echo2.finish(echo_message)