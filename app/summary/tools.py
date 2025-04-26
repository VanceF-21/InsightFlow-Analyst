import matplotlib.pyplot as plt
import seaborn as sns

def save_histogram(df, column_name):
    plt.figure(figsize=(10, 6))
    df[column_name].hist(bins=30, edgecolor='black')
    plt.title(f'{column_name} 直方图')
    plt.xlabel(column_name)
    plt.ylabel('频数')
    plt.grid(False)
    plt.savefig(f'{column_name}_histogram.png')
    plt.close()

def save_boxplot(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column_name])
    plt.title(f'{column_name} 箱线图')
    plt.xlabel(column_name)
    plt.savefig(f'{column_name}_boxplot.png')
    plt.close()

def save_scatter_plot(df, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column])
    plt.title(f'{y_column} vs {x_column} 散点图')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.savefig(f'{y_column}_vs_{x_column}_scatter.png')
    plt.close()

def save_pie_chart(df, column_name):
    plt.figure(figsize=(8, 8))
    df[column_name].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f'{column_name} 饼图')
    plt.ylabel('')
    plt.savefig(f'{column_name}_pie_chart.png')
    plt.close()

def save_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('相关性热力图')
    plt.savefig('correlation_heatmap.png')
    plt.close()


def create_histogram_tool(df):
    def histogram_tool(column_name):
        save_histogram(df, column_name)
        return f'![{column_name} 直方图]({column_name}_histogram.png)'
    return histogram_tool

def create_boxplot_tool(df):
    def boxplot_tool(column_name):
        save_boxplot(df, column_name)
        return f'![{column_name} 箱线图]({column_name}_boxplot.png)'
    return boxplot_tool

def create_scatter_plot_tool(df):
    def scatter_plot_tool(x_column, y_column):
        save_scatter_plot(df, x_column, y_column)
        return f'![{y_column} vs {x_column} 散点图]({y_column}_vs_{x_column}_scatter.png)'
    return scatter_plot_tool

def create_pie_chart_tool(df):
    def pie_chart_tool(column_name):
        save_pie_chart(df, column_name)
        return f'![{column_name} 饼图]({column_name}_pie_chart.png)'
    return pie_chart_tool

def create_heatmap_tool(df):
    def heatmap_tool():
        save_heatmap(df)
        return '![相关性热力图](correlation_heatmap.png)'
    return heatmap_tool