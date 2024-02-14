"""Datetime Template"""

import asyncio
from gc import callbacks
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from rich import print as rprint
from tomlkit import date

from sayvai_tools.tools.date import GetDate

# Create a new LangChain instance
llm = ChatOpenAI(model="gpt-4", streaming=True)

_SYSTEM_PROMPT: str = (
    """You have access to the GetDate tool, which provides information about the current date and time. 
    Your task is to respond to user queries related to date and time using the current information from the tool. 
    Your responses should be accurate and relevant to the user's query. Use the information from the GetDate tool to provide up-to-date responses."""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks! "),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


class GetDateToolCallbackHandler(BaseCallbackHandler):

    def __init__(self) -> None:
        super().__init__()

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        inputs: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> Any:
        rprint("I am inside callback")
        return super().on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            inputs=inputs,
            **kwargs
        )


class DateTimeAgent:

    def __init__(self):
        self.llm = llm
        self.prompt = prompt
        self.tools = None
        self.memory = ConversationBufferWindowMemory(
            buffer_size=10,  # type: ignore
            window_size=5,  # type: ignore
            memory_key="history"
        )

    def initialize_tools(self, tools=None) -> str:
        self.tools = tools
        if self.tools is None:
            self.tools = [
                Tool(
                    func=GetDate()._run,
                    name="GetDateTool",
                    description="""A tool that takes no input and returns the current date and time.""",
                    callbacks=[GetDateToolCallbackHandler()],
                ),
            ]
        return "Tools Initialized"

    def initialize_agent_executor(self, verbose: bool = True) -> str:
        self.agent = agent = create_openai_functions_agent(
            llm.with_config({"tags": ["agent_llm"]}),  # type: ignore
            self.tools,  # type: ignore
            self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,  # type: ignore
            tools=self.tools,  # type: ignore
            verbose=verbose,
            # memory=self.memory
            ).with_config({"run_name": "Agent"})
        return "Agent Executor Initialized"

    def invoke(self, message) -> str:
        return self.agent_executor.invoke(input={"input": message})["output"]
    
    async def invoke_async(self, message) -> str:
        async for chunks in  self.agent_executor.astream(input={"input": message}):
            print(chunks)
            print("---")
        return "Done"
        


dateagent = DateTimeAgent()
dateagent.initialize_tools()
dateagent.initialize_agent_executor(verbose=True)

# asyncio.run(dateagent.invoke_async("What is madurai"))
while True:
    message = input("Enter your message: ")
    print(dateagent.invoke(message))
