import os
import uuid
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from typing import Optional, List


class LLMInterface:
    """
    A class to interact with large language models using OpenAI API and LangChain.
    Supports custom API keys, base URLs, and formatted input/output.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7, system_message: str = None):
        """
        Initialize the LLM interface.

        Args:
            api_key (str): The API key for the OpenAI model.
            base_url (Optional[str]): The base URL for the API endpoint (optional).
            model_name (str): The name of the OpenAI model to use.
            temperature (float): The temperature parameter for the model.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_message
        self.role_prompt = ""

        # Configure LangChain LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url
        )

        # Initialize LangGraph
        self.memory = MemorySaver()  # 使用 MemorySaver 作为持久化存储
        self.be_init = []  # 保存初始化过的thread_id
        self.workflow = self._create_workflow()  # 创建并编译工作流

        # 初始化增量摘要
        self.summary = ""  # 用于存储压缩后的前文摘要

    def _create_workflow(self):
        """
        创建并编译 LangGraph 工作流
        """
        workflow = StateGraph(state_schema=MessagesState)

        # 定义调用模型的函数
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}

        # 定义增量摘要逻辑
        from langchain_core.messages import RemoveMessage

        def update_summary(state: MessagesState):
            """
            更新增量摘要，保留 system prompt、summary 消息和五条最新的用户消息，并删除余下的消息。
            """
            messages = state["messages"]

            # 确保消息列表中至少包含 system prompt 和 summary 消息
            if len(messages) < 3:
                # 如果消息总数少于3条，直接返回原列表（因为没有足够的消息进行压缩）
                return {"messages": messages}

            # 提取 system prompt 和 summary 消息
            system_prompt = messages[0]  # 假设 system prompt 是第一条消息
            summary_message = messages[1]  # 假设 summary 是第二条消息

            # 提取最新的五条用户消息（确保不会超出列表范围）
            recent_messages = messages[-5:] if len(messages) > 5 else messages[2:]

            # 获取需要删除的旧消息（排除 system prompt、summary 和最新的五条消息）
            # 注意边界条件：确保不会重复包含 system prompt 或 summary
            older_messages = messages[2:-5] if len(messages) > 7 else []

            # 如果有旧消息，更新摘要
            if older_messages:
                # 构造摘要更新的提示
                summary_prompt = (
                        f"根据以下对话内容更新摘要，保留你认为值得记忆的信息，舍去不值得长远记忆的信息。你的摘要应当尽可能简短和精华：\n"
                        f"旧摘要： {summary_message.content}\n"
                        f"新消息组：\n" +
                        "\n".join([msg.content for msg in older_messages]) +
                        "\n你的回复只应当包含摘要内容。以‘前文摘要：’开始"
                )
                # 调用 LLM 更新摘要
                new_summary_message = HumanMessage(content=summary_prompt)
                updated_summary = self.llm.invoke([new_summary_message]).content
                summary_message = HumanMessage(content=updated_summary)  # 更新摘要消息

            older_messages = messages[0:] if len(messages) > 7 else []
            # 按照内容重新构造 messages
            new_recent_messages = []
            if older_messages:
                for msg in recent_messages:
                    # 区分HumanMessage和AIMessage
                    if isinstance(msg, HumanMessage):
                        new_msg = HumanMessage(content=f"{msg.content}")
                    else:
                        new_msg = AIMessage(content=f"{msg.content}")
                    new_recent_messages.append(new_msg)

            # 检查system_prompt是否需要更新
            if self.role_prompt+self.system_prompt != system_prompt.content:
                system_prompt = SystemMessage(content="你的角色设定："+self.role_prompt+"\n"+self.system_prompt)

            # 返回更新后的消息列表：system prompt + 更新后的摘要 + 最新的五条消息
            # 同时返回需要删除的消息
            return {
                "messages": [RemoveMessage(id=msg.id) for msg in older_messages] + [system_prompt,
                                                                                    summary_message] + new_recent_messages
            }

        # 添加节点和边
        workflow.add_node("update_summary", update_summary)  # 添加增量摘要节点
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "update_summary")
        workflow.add_edge("update_summary", "model")  # 添加从摘要节点到模型节点的边

        # 编译工作流并传入 MemorySaver
        return workflow.compile(checkpointer=self.memory)

    def call_model_with_langchain(self, prompt: str, reply: bool = True, thread_id: str = "") -> str:
        """
        Call the large language model using LangChain and a formatted template.

        Args:
            prompt (str): The input prompt for the model.
            reply (bool): Whether to generate a response from the model.
            thread_id (str): section id

        Returns:
            str: The response from the model (if reply is True).
        """

        # if thread not init
        if thread_id not in self.be_init:
            self.be_init.append(thread_id)
            # 添加系统提示词
            system_message = SystemMessage(content=self.system_prompt)
            # 配置对话 ID
            config = {"configurable": {"thread_id": thread_id}}
            for event in self.workflow.stream({"messages": [system_message]}, config, stream_mode="values"):
                break

            # 添加摘要消息（初始为空）
            initial_summary = HumanMessage(content="前文摘要：暂无")
            # 配置对话 ID
            config = {"configurable": {"thread_id": thread_id}}
            for event in self.workflow.stream({"messages": [initial_summary]}, config, stream_mode="values"):
                break

        # 创建 HumanMessage
        input_message = HumanMessage(content=prompt)

        # 配置对话 ID
        config = {"configurable": {"thread_id": thread_id}}

        if reply:
            # 调用工作流并获取响应
            response = ""
            cnt = 0
            for event in self.workflow.stream({"messages": [input_message]}, config, stream_mode="values"):
                cnt += 1
                response = event["messages"][-1].content
                # print(f"AI: {response}")
                if cnt == 3:
                    print(f"AI11: {response}")
                    for message in event['messages']:
                        print(message.content)
                        print()

            return response
        else:
            # 仅更新记忆，不触发模型回复
            cnt = 0
            for event in self.workflow.stream({"messages": [input_message]}, config, stream_mode="values"):
                cnt += 1
                if cnt == 2:
                    for message in event['messages']:
                        print(message.content)
                        print()
                    break
            return ""

    def update_role_prompt(self, role_prompt: str):
        """
        更新角色提示词
        """
        self.role_prompt = role_prompt

    def __repr__(self):
        return (
            f"LLMInterface(api_key={self.api_key}, base_url={self.base_url}, "
            f"model_name='{self.model_name}', temperature={self.temperature})"
        )


# Example usage
if __name__ == "__main__":
    # Load API key from file
    with open("config.data", "r") as f:
        api_key = f.read().strip()

    # Initialize the LLM interface
    llm_interface = LLMInterface(
        api_key=api_key,
        base_url="https://api.yesapikey.com/v1",  # 注意检查链接的合法性
        model_name="gpt-4o-mini",
        temperature=1,
        system_message="你是一个群聊中的一员。所有用户的发言会以“名称：内容”的形式输入给你，而当你需要回复的时候，作为“GPT”的身份参与到讨论中，给出符合语境的回复。你可以向任何其他用户提问，也可以回复任何其他用户的消息。"
    )

    # Test dialogues
    test_dialogues = [
        "Alice: 我真的很喜欢早上跑步。这让我精力充沛！",
        "Bob: 跑步，很棒但我更喜欢游泳。对关节更友好。",
        "Charlie: 说到运动，你们有没有试过瑜伽？它对柔韧性帮助很大。",
        "Diana: 我同意Charlie的看法。瑜伽对缓解压力也很有帮助。",
        "Eve: 我最近在考虑尝试一种新的锻炼方式。有什么建议吗？",
        "Alice: 今天早餐我喝了一杯加了菠菜和香蕉的奶昔，很好吃。",
        "Bob: 菠菜奶昔？听起来很有趣。我通常吃鸡蛋和吐司。",
        "Charlie: 我喜欢尝试不同的食材。比如可以加点奇亚籽，增加营养。",
        "Diana: 哦，我一直想试试奇亚籽！有什么好的食谱推荐吗？",
        "Eve: 我发现了一个很棒的奇亚籽布丁食谱，做起来很简单。",
        "Alice: 我最近也在尝试烘焙，昨天做了全麦面包。",
        "Bob: 烘焙很有趣！我试过做一次酸面包，但挺有挑战性的。",
        "Charlie: 你应该试试自己做披萨面团。其实很简单。",
        "Diana: 我会试试！有什么做完美披萨皮的秘诀吗？",
        "Eve: 我听说用披萨石可以让披萨的质地更好。",
        "Alice: 这是个好主意，Eve。我会考虑买一个。"
    ]

    print("\n=== 测试记忆模型 ===\n")

    for i, user_message in enumerate(test_dialogues):
        print(f"\n用户: {user_message}")
        if i in [5, 7, 13]:  # 只在最后一句触发回复
            llm_interface.call_model_with_langchain(user_message, reply=True)
        else:
            llm_interface.call_model_with_langchain(user_message, reply=False)
