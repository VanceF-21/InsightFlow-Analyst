import time
import streamlit as st
from streamlit_lottie import st_lottie
from util import load_lottie, stream_data, welcome_message, introduction_message
from prediction_model import prediction_model_pipeline
from cluster_model import cluster_model_pipeline
from regression_model import regression_model_pipeline
from visualization import data_visualization
from src.util import read_file_from_streamlit
from summary.qa_agent import create_qa_agent, get_agent_response

st.set_page_config(page_title="InsightFlow Analyst", page_icon=":rocket:", layout="wide")

# TITLE SECTION
with st.container():
    st.subheader("您好 👋")
    st.title("欢迎使用 InsightFlow Analyst！")
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    if st.session_state.initialized:
        st.session_state.welcome_message = welcome_message()
        st.write(stream_data(st.session_state.welcome_message))
        time.sleep(0.5)
        st.write("[在 GitHub 上探索 InsightFlow Analyst 的神奇之处！ 🌟 > ](https://github.com/VanceF-21/InsightFlow-Analyst)")
        st.session_state.initialized = False
    else:
        st.write(st.session_state.welcome_message)
        st.write("[在 GitHub 上探索 InsightFlow Analyst 的神奇之处！ 🌟 > ](https://github.com/VanceF-21/InsightFlow-Analyst)")

# INTRO SECTION
with st.container():
    st.divider()
    if 'lottie' not in st.session_state:
        st.session_state.lottie_url1, st.session_state.lottie_url2 = load_lottie()
        st.session_state.lottie = True

    left_column_r1, right_column_r1 = st.columns([6, 4])
    with left_column_r1:
        st.header("InsightFlow Analyst 能做什么？")
        st.write(introduction_message()[0])
    with right_column_r1:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url1, height=280, key="animation1")

    left_column_r2, _, right_column_r2 = st.columns([6, 1, 5])
    with left_column_r2:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url2, height=200, key="animation2")
    with right_column_r2:
        st.header("简单易用")
        st.write(introduction_message()[1])

# MAIN SECTION
with st.container():
    st.divider()
    st.header("📊 数据探索与可视化")
    left_column, right_column = st.columns([6, 4])
    with left_column:
        API_KEY = st.text_input(
            "您的 API Key",
            placeholder="在此输入您的 API key...",
            type="password"
        )
        st.write("👆您的 API key：")
        uploaded_file = st.file_uploader("选择一个数据文件", accept_multiple_files=False, type=['csv', 'json', 'xls', 'xlsx'])
        if uploaded_file:
            if uploaded_file.getvalue():
                uploaded_file.seek(0)
                st.session_state.DF_uploaded = read_file_from_streamlit(uploaded_file)
                st.session_state.is_file_empty = False
            else:
                st.session_state.is_file_empty = True
        
    with right_column:
        SELECTED_MODEL = st.selectbox(
        '您想使用哪个 LLM 模型？',
        ('GPT-4', 'GPT-3.5-Turbo', 'GPT-4o'))

        MODE = st.selectbox(
        '选择合适的数据分析模式',
        ('分类模型', '聚类模型', '回归模型', '数据可视化'))
        
        st.write(f'选择的模型: :green[{SELECTED_MODEL}]')
        st.write(f'数据分析模式: :green[{MODE}]')

    # Proceed Button
    is_proceed_enabled = uploaded_file is not None and API_KEY != "" or uploaded_file is not None and MODE == "数据可视化"

    # Initialize the 'button_clicked' state
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    if st.button('开始分析', disabled=(not is_proceed_enabled) or st.session_state.button_clicked, type="primary"):
        st.session_state.button_clicked = True
    if "is_file_empty" in st.session_state and st.session_state.is_file_empty:
        st.caption('您的数据文件是空的！')

    # Start Analysis
    if st.session_state.button_clicked:
        # GPT_MODEL = 4 if SELECTED_MODEL == 'GPT-4-Turbo' else 3.5
        if SELECTED_MODEL == "GPT-4":
            GPT_MODEL = 4
        elif SELECTED_MODEL == "GPT-3.5-Turbo":
            GPT_MODEL = 3.5
        else:
            GPT_MODEL = 4.5
        with st.container():
            if "DF_uploaded" not in st.session_state:
                st.error("文件为空！")
            else:
                if MODE == '分类模型':
                    prediction_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == '聚类模型':
                    cluster_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == '回归模型':
                    regression_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == '数据可视化':
                    data_visualization(st.session_state.DF_uploaded)

# Q&A SECTION
with st.container():
    st.divider()
    st.header("💬 智能数据分析助手")
    
    # Initialize session states
    if 'qa_button_clicked' not in st.session_state:
        st.session_state.qa_button_clicked = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    # Q&A specific controls
    qa_left, qa_right = st.columns([6, 4])
    with qa_left:
        qa_api_key = st.text_input(
            "您的 API Key",
            placeholder="在此输入您的API key...",
            type="password"
        )
    with qa_right:
        qa_model = st.selectbox(
            '选择对话模型',
            ('GPT-4', 'GPT-3.5-Turbo', 'GPT-4o'),
            key="qa_model"
        )
    
    # File uploader for Q&A
    qa_file = st.file_uploader("上传要分析的数据文件", accept_multiple_files=False, type=['csv', 'json', 'xls', 'xlsx'], key="qa_file")
    
    # Start Q&A button
    is_qa_ready = qa_file is not None and qa_api_key != ""
    if st.button('开始数据分析对话', disabled=not is_qa_ready, type="primary", key="start_qa"):
        st.session_state.qa_button_clicked = True
        if qa_file.getvalue():
            qa_file.seek(0)
            st.session_state.qa_df = read_file_from_streamlit(qa_file)
            # Initialize Q&A agent
            if qa_model == "GPT-4":
                qa_model_version = 4
            elif qa_model == "GPT-3.5-Turbo":
                qa_model_version = 3.5
            else:
                qa_model_version = 4.5
            st.session_state.qa_agent = create_qa_agent(st.session_state.qa_df, qa_api_key, qa_model_version)
    
    # Q&A Interface
    if st.session_state.qa_button_clicked and "qa_df" in st.session_state:
        left_column, right_column = st.columns([7, 3])
        
        with left_column:
            st.write("请输入您的问题，我会基于您上传的数据为您解答。")

            # Chat interface
            chat_container = st.container()
            with chat_container:
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("在这里输入您的问题..."):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Get agent response
                    with st.chat_message("assistant"):
                        with st.spinner("思考中..."):
                            response = get_agent_response(st.session_state.qa_agent, prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
        
        with right_column:
            st.subheader("💡 提示")
            st.write("您可以询问的问题类型：")
            st.markdown("""
            - 数据基本统计信息
            - 特定列的分析
            - 数据分布情况
            - 相关性分析
            - 数据趋势
            - 异常值检测
            - 数据可视化请求
            """)
            
            # Clear chat history button
            if st.button("🗑️ 清除聊天记录", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()
    