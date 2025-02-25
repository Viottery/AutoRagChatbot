import os
import uuid
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langchain.schema.messages import HumanMessage, SystemMessage
from typing import Optional


class LLMInterface:
    """
    A class to interact with large language models using OpenAI API and LangChain.
    Supports custom API keys, base URLs, and formatted input/output.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, system_message: str= None):
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

        # Configure LangChain LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url
        )

        # Initialize LangGraph
        self.memory = MemorySaver()  # 使用 MemorySaver 作为持久化存储
        self.thread_id = str(uuid.uuid4())  # 为每个会话生成唯一的 thread_id
        self.workflow = self._create_workflow()  # 创建并编译工作流

        # 添加系统提示词
        if system_message:
            system_message = SystemMessage(content=system_message)
            # 工作流
            # 配置对话 ID
            config = {"configurable": {"thread_id": self.thread_id}}
            cnt = 0
            for event in self.workflow.stream({"messages": [system_message]}, config, stream_mode="values"):
                cnt += 1
                if cnt == 1:
                    break


    def _create_workflow(self):
        """
        创建并编译 LangGraph 工作流
        """
        workflow = StateGraph(state_schema=MessagesState)

        # 定义调用模型的函数
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}

        # 添加节点和边
        workflow.add_node("start", lambda x: x)  # 添加一个简单的起始节点
        workflow.add_node("model", call_model)
        workflow.add_edge("start", "model")  # 添加从起始节点到模型节点的边
        workflow.set_entry_point("start")  # 设置入口点为 "start"

        # 编译工作流并传入 MemorySaver
        return workflow.compile(checkpointer=self.memory)

    def call_model_with_langchain(self, prompt: str, reply: bool = True) -> str:
        """
        Call the large language model using LangChain and a formatted template.

        Args:
            prompt (str): The input prompt for the model.
            reply (bool): Whether to generate a response from the model.

        Returns:
            str: The response from the model (if reply is True).
        """
        # 创建 HumanMessage
        input_message = HumanMessage(content=prompt)

        # 配置对话 ID
        config = {"configurable": {"thread_id": self.thread_id}}

        if reply:
            # 调用工作流并获取响应
            response = ""
            cnt = 0
            for event in self.workflow.stream({"messages": [input_message]}, config, stream_mode="values"):
                cnt += 1
                response = event["messages"][-1].content
                if cnt == 3:
                    print(f"AI: {response}")
            return response
        else:
            # 仅更新记忆，不触发模型回复
            cnt = 0
            for event in self.workflow.stream({"messages": [input_message]}, config, stream_mode="values"):
                cnt += 1
                if cnt == 1:
                    break
            return ""

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