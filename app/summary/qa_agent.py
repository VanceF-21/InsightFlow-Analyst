from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import datetime
import streamlit as st

OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
def create_qa_agent(df, api_key, model_version=4.5):
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
    

def generate_md_report():
    """生成 Markdown 格式的问答报告"""
    if "qa_history" not in st.session_state or not st.session_state.qa_history:
        return None
    
    report_content = "# 📊 数据分析问答报告\n\n"
    report_content += f"**生成时间：** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, entry in enumerate(st.session_state.qa_history):
        report_content += f"### ❓ 问题 {i+1}\n"
        report_content += f"**用户提问：** {entry['question']}\n\n"
        report_content += f"**🤖 AI 回答：**\n{entry['answer']}\n\n"
        report_content += "---\n"

    return report_content