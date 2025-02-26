from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
def create_qa_agent(df, api_key, model_version=3.5):
    """创建一个基于数据框的问答agent"""
    if model_version == 4:
        model_name = "gpt-4"
    elif model_version == 3.5:
        model_name = "gpt-3.5-turbo"
    elif model_version == 4.5:
        model_name = "gpt-4o-0125-preview"
    
    llm = ChatOpenAI(
        name=model_name,
        api_key=api_key,
        base_url=OPENAI_BASE_URL
    )
    
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
    )
    
    return agent

def get_agent_response(agent, query):
    """获取agent对用户查询的响应"""
    try:
        response = agent.run(query)
        return response
    except Exception as e:
        return f"抱歉，处理您的问题时出现错误：{str(e)}" 