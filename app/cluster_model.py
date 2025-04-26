import streamlit as st
from util import developer_info, developer_info_static
from src.plot import plot_clusters, correlation_matrix_plotly
from src.handle_null_value import contains_missing_value, remove_high_null, fill_null_values
from src.preprocess import convert_to_numeric, remove_duplicates, transform_data_for_clustering
from src.llm_service import decide_fill_null, decide_encode_type, decide_cluster_model
from src.pca import decide_pca, perform_PCA_for_clustering
from src.model_service import save_model, calculate_silhouette_score, calculate_calinski_harabasz_score, calculate_davies_bouldin_score, gmm_predict, estimate_optimal_clusters
from src.cluster_model import train_select_cluster_model
from src.util import contain_null_attributes_info, separate_fill_null_list, check_all_columns_numeric, non_numeric_columns_and_head, separate_decode_list, get_cluster_method_name

def start_training_model():
    st.session_state["start_training"] = True

def cluster_model_pipeline(DF, API_KEY, GPT_MODEL):
    st.divider()
    st.subheader('数据概览')
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin = DF
    st.dataframe(st.session_state.data_origin.describe(), width=1200)
    
    # Data Imputation
    st.subheader('处理和填补缺失值')
    if "contain_null" not in st.session_state:
        st.session_state.contain_null = contains_missing_value(st.session_state.data_origin)

    if 'filled_df' not in st.session_state:
        if st.session_state.contain_null:
            with st.status("Processing **missing values** in the data...", expanded=True) as status:
                st.write("正在过滤高频缺失的行和列...")
                filled_df = remove_high_null(DF)
                st.write("LLM正在分析...")
                attributes, types_info, description_info = contain_null_attributes_info(filled_df)
                fill_result_dict = decide_fill_null(attributes, types_info, description_info, GPT_MODEL, API_KEY)
                st.write("正在填补缺失值...")
                mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)
                filled_df = fill_null_values(filled_df, mean_list, median_list, mode_list, new_category_list, interpolation_list)
                # Store the imputed DataFrame in session_state
                st.session_state.filled_df = filled_df
                DF = filled_df
                status.update(label='Missing value processing completed!', state="complete", expanded=False)
            st.download_button(
                label="Download Data with Missing Values Imputed",
                data=st.session_state.filled_df.to_csv(index=False).encode('utf-8'),
                file_name="imputed_missing_values.csv",
                mime='text/csv')
        else:
            st.session_state.filled_df = DF
            st.success("没有被检测到的缺失值存在，跳过该过程。")
    else:
        st.success("缺失值处理已完成！")
        if st.session_state.contain_null:
            st.download_button(
                label="Download Data with Missing Values Imputed",
                data=st.session_state.filled_df.to_csv(index=False).encode('utf-8'),
                file_name="imputed_missing_values.csv",
                mime='text/csv')

    # Data Encoding
    st.subheader("处理数据编码")
    st.caption("*为了处理时间的考虑，当前管道中未包含**NLP特征**如**TF-IDF**，长文本属性可能会被丢弃。")
    if 'all_numeric' not in st.session_state:
        st.session_state.all_numeric = check_all_columns_numeric(st.session_state.data_origin)
    
    if 'encoded_df' not in st.session_state:
        if not st.session_state.all_numeric:
            with st.status("Encoding non-numeric data using **numeric mapping** and **one-hot**...", expanded=True) as status:
                non_numeric_attributes, non_numeric_head = non_numeric_columns_and_head(DF)
                st.write("LLM正在分析...")
                encode_result_dict = decide_encode_type(non_numeric_attributes, non_numeric_head, GPT_MODEL, API_KEY)
                st.write("正在编码数据...")
                convert_int_cols, one_hot_cols, drop_cols = separate_decode_list(encode_result_dict, "")
                encoded_df, mappings = convert_to_numeric(DF, convert_int_cols, one_hot_cols, drop_cols)
                # Store the imputed DataFrame in session_state
                st.session_state.encoded_df = encoded_df
                DF = encoded_df
                status.update(label='Data encoding completed!', state="complete", expanded=False)
            st.download_button(
                label="Download Encoded Data",
                data=st.session_state.encoded_df.to_csv(index=False).encode('utf-8'),
                file_name="encoded_data.csv",
                mime='text/csv')
        else:
            st.session_state.encoded_df = DF
            st.success("所有列都是数值的，跳过处理!")
    else:
        st.success("使用数值映射和独热编码处理数据完成!")
        if not st.session_state.all_numeric:
            st.download_button(
                label="Download Encoded Data",
                data=st.session_state.encoded_df.to_csv(index=False).encode('utf-8'),
                file_name="encoded_data.csv",
                mime='text/csv')
    
    # Correlation Heatmap
    if 'df_cleaned1' not in st.session_state:
        st.session_state.df_cleaned1 = DF
    st.subheader('属性之间的相关性')
    st.plotly_chart(correlation_matrix_plotly(st.session_state.df_cleaned1))

    # Remove duplicate entities
    st.subheader('删除重复实体')
    if 'df_cleaned2' not in st.session_state:
        st.session_state.df_cleaned2 = remove_duplicates(st.session_state.df_cleaned1)
        # DF = remove_duplicates(DF)
    st.info("重复的行已删除。")

    # Data Transformation
    st.subheader('数据转换')
    if 'data_transformed' not in st.session_state:
        st.session_state.data_transformed = transform_data_for_clustering(st.session_state.df_cleaned2)
    st.success("如果需要，数据已进行标准化和Box-Cox变换。")
    
    # PCA
    st.subheader('Principal Component Analysis')
    st.write("正在决定是否进行PCA...")
    if 'df_pca' not in st.session_state:
        _, n_components = decide_pca(st.session_state.df_cleaned2)
        st.session_state.df_pca = perform_PCA_for_clustering(st.session_state.data_transformed, n_components)
    st.success("Completed!")

    # Splitting and Balancing
    if 'test_percentage' not in st.session_state:
        st.session_state.test_percentage = 20
    if 'balance_data' not in st.session_state:
        st.session_state.balance_data = False
    if "start_training" not in st.session_state:
        st.session_state["start_training"] = False
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False

    splitting_column, balance_column = st.columns(2)
    with splitting_column:
        st.subheader(':grey[数据分割]')
        st.caption('数据分割不适用于聚类模型。')
        st.slider('测试集的比例', 1, 25, st.session_state.test_percentage, key='test_percentage', disabled=True)
    
    with balance_column:
        st.metric(label="Test Data", value="--%", delta=None)
        st.toggle('Class Balancing', value=st.session_state.balance_data, key='to_perform_balance', disabled=True)
        st.caption('Class balancing is not applicable to clustering models.')
    
    st.button("开始训练模型", on_click=start_training_model, type="primary", disabled=st.session_state['start_training'])

    # Model Training
    if st.session_state['start_training']:
        with st.container():
            st.header("建立模型")
            if not st.session_state.get("data_prepared", False): 
                st.session_state.X = st.session_state.df_pca
                st.session_state.data_prepared = True

            # Decide model types:
            if "decided_model" not in st.session_state:
                st.session_state["decided_model"] = False
            if "all_set" not in st.session_state:
                st.session_state["all_set"] = False

            if not st.session_state["decided_model"]:
                with st.spinner("正在根据数据选择模型..."):
                    shape_info = str(st.session_state.X.shape)
                    description_info = st.session_state.X.describe().to_csv()
                    cluster_info = estimate_optimal_clusters(st.session_state.X)
                    st.session_state.default_cluster = cluster_info
                    model_dict = decide_cluster_model(shape_info, description_info, cluster_info, GPT_MODEL, API_KEY)
                    model_list = list(model_dict.values())
                    if 'model_list' not in st.session_state:
                        st.session_state.model_list = model_list
                    st.session_state.decided_model = True

            # Display results
            if st.session_state["decided_model"]:
                display_results(st.session_state.X)
                st.session_state["all_set"] = True
            
            # Download models
            if st.session_state["all_set"]:
                download_col1, download_col2, download_col3 = st.columns(3)
                with download_col1:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model1, file_name=f"{st.session_state.model1_name}.joblib", mime="application/octet-stream")
                with download_col2:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model2, file_name=f"{st.session_state.model2_name}.joblib", mime="application/octet-stream")
                with download_col3:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model3, file_name=f"{st.session_state.model3_name}.joblib", mime="application/octet-stream")

    # Footer
    st.divider()
    if "all_set" in st.session_state and st.session_state["all_set"]:
        if "has_been_set" not in st.session_state:
            st.session_state["has_been_set"] = True
            developer_info()
        else:
            developer_info_static()

def display_results(X):
    st.success("根据数据成功选择模型!")

    # Data set metrics
    st.metric(label="Total Data", value=len(X), delta=None)
    
    # Model training
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        if "model1_name" not in st.session_state:
            st.session_state.model1_name = get_cluster_method_name(st.session_state.model_list[0])
        st.subheader(st.session_state.model1_name)

        # Slider for model parameters
        if st.session_state.model_list[0] == 2:
            st.caption('N-cluster不适用于DBSCAN.')
        else:
            st.caption(f'N-cluster for {st.session_state.model1_name}:')
        n_clusters1 = st.slider('N clusters', 2, 20, st.session_state.default_cluster, label_visibility="collapsed", key='n_clusters1', disabled=st.session_state.model_list[0] == 2)
        
        with st.spinner("模型训练进行中..."):
            st.session_state.model1 = train_select_cluster_model(X, n_clusters1, st.session_state.model_list[0])
            st.session_state.downloadable_model1 = save_model(st.session_state.model1)
       
        if st.session_state.model_list[0] != 3:
            label1 = st.session_state.model1.labels_
        else:
            label1 = gmm_predict(X, st.session_state.model1)

        # Visualization
        st.pyplot(plot_clusters(X, label1))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, label1)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, label1)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, label1)}**]')

    with model_col2:
        if "model2_name" not in st.session_state:
            st.session_state.model2_name = get_cluster_method_name(st.session_state.model_list[1])
        st.subheader(st.session_state.model2_name)

        # Slider for model parameters
        if st.session_state.model_list[1] == 2:
            st.caption('N-cluster不适用于DBSCAN.')
        else:
            st.caption(f'N-cluster for {st.session_state.model2_name}:')
        n_clusters2 = st.slider('N clusters', 2, 20, st.session_state.default_cluster, label_visibility="collapsed", key='n_clusters2', disabled=st.session_state.model_list[1] == 2)

        with st.spinner("模型训练进行中..."):
            st.session_state.model2 = train_select_cluster_model(X, n_clusters2, st.session_state.model_list[1])
            st.session_state.downloadable_model2 = save_model(st.session_state.model2)

        if st.session_state.model_list[1] != 3:
            label2 = st.session_state.model2.labels_
        else:
            label2 = gmm_predict(X, st.session_state.model2)

        # Visualization
        st.pyplot(plot_clusters(X, label2))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, label2)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, label2)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, label2)}**]')
        
    with model_col3:
        if "model3_name" not in st.session_state:
            st.session_state.model3_name = get_cluster_method_name(st.session_state.model_list[2])
        st.subheader(st.session_state.model3_name)

        # Slider for model parameters
        if st.session_state.model_list[2] == 2:
            st.caption('N-cluster不适用于DBSCAN.')
        else:
            st.caption(f'N-cluster for {st.session_state.model3_name}:')
        n_clusters3 = st.slider('N clusters', 2, 20, st.session_state.default_cluster, label_visibility="collapsed", key='n_clusters3', disabled=st.session_state.model_list[2] == 2)

        with st.spinner("模型训练进行中..."):
            st.session_state.model3 = train_select_cluster_model(X, n_clusters3, st.session_state.model_list[2])
            st.session_state.downloadable_model3 = save_model(st.session_state.model3)

        if st.session_state.model_list[2] != 3:
            label3 = st.session_state.model3.labels_
        else:
            label3 = gmm_predict(X, st.session_state.model3)

        # Visualization
        st.pyplot(plot_clusters(X, label3))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, label3)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, label3)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, label3)}**]')
