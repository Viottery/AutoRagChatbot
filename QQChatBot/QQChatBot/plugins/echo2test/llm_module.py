# -*- coding: utf-8 -*-
"""
This module provides a basic interface to interact with large language models using OpenAI API and LangChain.
It supports custom API keys, base URLs, and formatted input/output.
"""

import os
from openai import OpenAI
from langchain_openai import ChatOpenAI  # 更新导入路径
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence  # 替换 LLMChain
from typing import Optional


class LLMInterface:
    """
    A class to interact with large language models using OpenAI API and LangChain.
    Supports custom API keys, base URLs, and formatted input/output.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
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

        # Configure OpenAI client
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)

        # Configure LangChain LLM
        self.llm = ChatOpenAI(  # 使用更新后的 ChatOpenAI
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url
        )

    def call_model(self, prompt: str) -> str:
        """
        Call the large language model using the provided prompt.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            str: The response from the model.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to call the model: {e}")

    def call_model_with_langchain(self, prompt: str, template: str = None) -> str:
        """
        Call the large language model using LangChain and a formatted template.

        Args:
            prompt (str): The input prompt for the model.
            template (str): The template to format the prompt (optional).

        Returns:
            str: The response from the model.
        """
        if template:
            prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
            # 替换 LLMChain 为 RunnableSequence
            chain = RunnableSequence(prompt_template, self.llm)
            response = chain.invoke({"prompt": prompt})  # 替换 chain.run 为 chain.invoke
        else:
            response = self.llm(prompt)
        return response

    def __repr__(self):
        return (
            f"LLMInterface(api_key={self.api_key}, base_url={self.base_url}, "
            f"model_name='{self.model_name}', temperature={self.temperature})"
        )


# Example usage
if __name__ == "__main__":
    # load api key from file
    with open("../../../config.data", "r") as f:
        api_key = f.read().strip()

    # Initialize the LLM interface
    llm_interface = LLMInterface(
        api_key=api_key,
        base_url="https://api.yesapikey.com/v1",
        model_name="gpt-4o-mini",
        temperature=1
    )

    # Example: Call the model directly using OpenAI
    prompt = "Tell me a joke."
    response = llm_interface.call_model(prompt)
    print("\nDirect Model Response:", response)

    # Example: Call the model using LangChain with a formatted template
    template = "You are a friendly assistant. Answer the following question: {prompt}"
    response = llm_interface.call_model_with_langchain(prompt, template=template)
    print("\nLangChain Formatted Response:", response)