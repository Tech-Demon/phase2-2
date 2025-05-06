from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from ..tools.base_tools import DocumentQueryTool, WebsiteQueryTool, DatabaseQueryTool
from sqlalchemy.orm import Session
from .prompts import AGENT_SYSTEM_TEMPLATE, AGENT_USER_TEMPLATE

class CollegeChatbotAgent:
    def __init__(self, db: Session):
        # Initialize LLM with specific configuration
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-16k",  # Using 16k context for handling longer conversations
            streaming=True
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=4000  # Prevent memory from growing too large
        )
        
        # Initialize tools with descriptions
        self.tools = [
            DocumentQueryTool(),
            WebsiteQueryTool(),
            DatabaseQueryTool(db)
        ]
        
        # Create the prompt from our templates
        prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", AGENT_USER_TEMPLATE),
        ])
        
        # Initialize the OpenAI Functions agent
        agent = OpenAIFunctionsAgent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        self.agent = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,  # Limit maximum number of tool invocations
            early_stopping_method="generate",  # Stop if agent starts looping
            handle_parsing_errors=True  # Better error handling
        )
        
    async def get_response(self, query: str) -> str:
        """
        Get response from the agent for a given query
        """
        try:
            response = await self.agent.arun(input=query)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
