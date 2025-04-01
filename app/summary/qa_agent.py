from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import datetime
import streamlit as st

OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
def create_qa_agent(df, api_key, model_version=4.5):
    """创建一个基于数据框的问答agent"""
    if model_version == 3.5:
        model_name = "gpt-3.5-turbo"
    elif model_version == 4:
        model_name = "gpt-4"
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
    

import pandas as pd
import streamlit as st
import datetime

def generate_md_report(df, qa_history):
    """生成包含文件背景、数据统计、问答和总结的 Markdown 格式报告"""
    if not qa_history:
        return None
    
    # 生成报告内容
    report_content = "# 📊 数据分析问答报告\n\n"
    report_content += f"**生成时间：** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # 文件背景介绍
    report_content += "## 📄 文件介绍\n"
    report_content += "文件内容概述：\n"
    report_content += f"- 数据包含 {df.shape[0]} 行和 {df.shape[1]} 列。\n"
    report_content += "- 主要字段：\n"
    for column in df.columns[:5]:  # 显示前五个字段名称，防止太长
        report_content += f"  - {column}\n"
    
    # 数据基本统计信息
    report_content += "\n## 📊 数据基本统计信息\n"
    basic_stats = df.describe(include='all').transpose()
    report_content += basic_stats.to_markdown()
    
    # 每一轮问题和回答
    report_content += "\n## 💬 问答记录\n"
    for i, entry in enumerate(qa_history):
        report_content += f"### 问题 {i + 1}\n"
        report_content += f"**用户提问：** {entry['question']}\n\n"
        report_content += f"**AI 回答：**\n{entry['answer']}\n\n"
        report_content += "---\n"
    
    # # 汇总总结
    # report_content += "\n## 📝 汇总总结\n"
    # report_content += "以下是针对文件内容以及问答记录的总结：\n"
    # # 假设AI能够根据前面的问题和回答给出总结（这个部分可以根据你的需要进一步扩展）
    # report_content += "数据分析过程中，用户主要关注数据的基本统计信息、趋势和可能的异常情况，AI根据这些信息给出了分析报告。\n"

    return report_content
