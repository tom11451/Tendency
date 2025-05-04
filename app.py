import os
import re
import json
import base64
import io
import requests
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
import traceback
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from openai import OpenAI
from tavily import TavilyClient
from urllib.parse import quote_plus
from datetime import datetime
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 从环境变量中获取配置
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_BASE = os.environ.get('API_BASE', 'https://api.deepseek.com/v1')
MODEL_NAME = os.environ.get('MODEL_NAME', 'deepseek-chat')
DEFAULT_TEMPERATURE = float(os.environ.get('DEFAULT_TEMPERATURE', '0.2'))
DEFAULT_MAX_TOKENS = int(os.environ.get('DEFAULT_MAX_TOKENS', '2000'))
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY', '')
DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
PORT = int(os.environ.get('PORT', '5000'))

# 设置API的完整URL
CHAT_COMPLETIONS_URL = f"{API_BASE}/chat/completions"

# 设置API请求头
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# 配置静态文件目录
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(os.path.join(STATIC_DIR, 'charts'), exist_ok=True)

# 使用LLM处理用户查询
def process_with_llm(query, search_results=None):
    try:
        # 创建提示词模板
        prompt_template = """
        你是一个专业的数据分析助手。请根据以下搜索结果，提取相关数据并生成适合的统计图表。
        
        ## 用户查询
        {query}
        
        ## 搜索结果
        {search_results}
        
        ## 任务
        1. 分析上述信息，提取与用户查询相关的数据
        2. 确定最适合展示这些数据的图表类型（柱状图、折线图、饼图或散点图）
        3. 返回一个JSON格式的响应，包含以下字段：
           - chart_type: 图表类型，必须是以下之一：["bar", "line", "pie", "scatter"]
           - chart_title: 图表标题
           - chart_data: 图表数据，格式为数组或对象
           - chart_labels: 数据标签（如适用）
           - analysis: 对数据的简短分析
        
        ## 返回格式
        ```json
        {{
          "chart_type": "图表类型",
          "chart_title": "图表标题",
          "chart_data": [数据值],
          "chart_labels": ["标签1", "标签2", ...],
          "analysis": "数据分析结果"
        }}
        ```
        
        请确保：
        1. 返回的是有效的JSON格式
        2. chart_data和chart_labels的长度相匹配
        3. 所有数值都是数字类型，不要使用字符串表示数字
        4. 如果无法从搜索结果中提取有效数据，请在analysis中说明原因
        """
        
        # 填充提示词模板
        prompt = prompt_template.format(
            query=query,
            search_results=search_results if search_results else "未提供搜索结果"
        )
        
        # 调用DeepseekV3 API
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=API_BASE.rstrip('/')  # 移除末尾的斜线
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的数据分析助手，擅长从文本中提取数据并生成适合的统计图表。"},
                {"role": "user", "content": prompt}
            ],
            temperature=DEFAULT_TEMPERATURE,  # 降低温度以获得更确定性的输出
            max_tokens=DEFAULT_MAX_TOKENS
        )
        
        # 提取响应
        llm_response = response.choices[0].message.content
        
        # 尝试从响应中提取JSON
        try:
            # 查找JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个响应
                json_str = llm_response
            
            # 解析JSON
            result = json.loads(json_str)
            
            # 验证必要的字段
            required_fields = ["chart_type", "chart_title", "chart_data", "analysis"]
            for field in required_fields:
                if field not in result:
                    result[field] = "未提供" if field != "chart_data" else []
            
            # 确保chart_type是有效的类型
            valid_chart_types = ["bar", "line", "pie", "scatter"]
            if result["chart_type"] not in valid_chart_types:
                result["chart_type"] = "bar"  # 默认使用柱状图
            
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回错误信息
            print(f"无法解析LLM响应为JSON: {llm_response}")
            return {
                "chart_type": "bar",
                "chart_title": "数据解析错误",
                "chart_data": [],
                "chart_labels": [],
                "analysis": "无法从LLM响应中提取有效数据。原始响应: " + llm_response[:100] + "..."
            }
            
    except Exception as e:
        print(f"调用LLM时出错: {str(e)}")
        return {
            "chart_type": "bar",
            "chart_title": "处理错误",
            "chart_data": [],
            "chart_labels": [],
            "analysis": f"处理查询时出错: {str(e)}"
        }

# 网络搜索函数
def search_web(query):
    """使用Tavily API进行网络搜索"""
    try:
        # 使用Tavily API搜索
        return tavily_search(query)
    except Exception as e:
        print(f"Tavily搜索错误: {str(e)}")
        # 抛出异常，不再使用模拟数据
        raise Exception(f"搜索失败: {str(e)}")

# 使用Tavily API搜索
def tavily_search(query):
    """使用Tavily API进行网络搜索"""
    try:
        # 初始化Tavily客户端
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # 调用Tavily搜索API
        response = client.search(
            query=query,
            search_depth="basic",  # 可选: basic 或 comprehensive
            max_results=5,  # 最多返回5个结果
            include_domains=[],  # 可选: 包含特定域名
            exclude_domains=[],  # 可选: 排除特定域名
            include_answer=True,  # 包含AI生成的摘要答案
            include_raw_content=False,  # 不包含原始内容以节省带宽
        )
        
        # 处理搜索结果
        search_results = []
        
        # 首先添加Tavily生成的答案（如果有）
        if "answer" in response and response["answer"]:
            search_results.append({
                "title": "Tavily AI 摘要",
                "link": "https://tavily.com/",
                "snippet": response["answer"]
            })
        
        # 添加搜索结果
        if "results" in response:
            for result in response["results"]:
                search_results.append({
                    "title": result.get("title", "无标题"),
                    "link": result.get("url", "#"),
                    "snippet": result.get("content", "无内容摘要")
                })
        
        # 如果没有结果，抛出异常
        if not search_results:
            raise Exception("没有找到相关信息")
            
        return search_results
    except Exception as e:
        print(f"Tavily API搜索错误: {str(e)}")
        # 如果出错，抛出异常
        raise Exception(f"搜索失败: {str(e)}")

# 从URL获取数据
def extract_data_from_url(url):
    try:
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试查找表格数据
        try:
            from io import StringIO
            tables = pd.read_html(StringIO(response.text))
            if tables:
                return {
                    "type": "table",
                    "data": tables[0].to_dict(orient='records')  # 返回第一个表格
                }
        except Exception as table_error:
            print(f"提取表格数据时出错: {str(table_error)}")
            # 继续尝试其他方法提取数据
        
        # 如果没有表格，尝试提取文本中的数字数据
        text = soup.get_text()
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            # 将字符串数字转换为浮点数
            numeric_data = [float(num) for num in numbers[:20]]  # 限制返回的数字数量
            return {
                "type": "text",
                "data": numeric_data
            }
        
        # 如果没有找到数值数据，抛出异常
        print("没有找到数据")
        raise Exception("没有找到数据")
    except Exception as e:
        print(f"数据提取错误: {str(e)}")
        # 抛出异常
        raise Exception(f"数据提取错误: {str(e)}")

# 确定最适合的图表类型
def determine_chart_type(data, user_query):
    # 尝试使用DeepseekV3 API进行更智能的分析
    try:
        # 不需要传递数据，因为我们只是想获取图表类型建议
        llm_response = process_with_llm(user_query)
        
        if 'chart_type' in llm_response:
            chart_type = llm_response['chart_type'].lower()
            # 确保返回的图表类型是支持的
            if chart_type in ['bar', 'line', 'pie', 'scatter']:
                return chart_type
    except Exception as e:
        print(f"使用LLM确定图表类型时出错: {str(e)}")
    
    # 如果LLM分析失败，使用基于数据特征的方法
    # 如果数据是字典类型
    if isinstance(data, dict):
        # 如果值的数量少于5，使用饼图
        if len(data) <= 5:
            return "pie"
        # 如果键是数字或日期，使用折线图
        elif all(isinstance(k, (int, float)) or is_date(k) for k in data.keys()):
            return "line"
        # 否则使用柱状图
        else:
            return "bar"
    
    # 如果数据是列表类型
    elif isinstance(data, list):
        # 如果列表长度少于5，使用饼图
        if len(data) <= 5:
            return "pie"
        # 否则使用柱状图
        else:
            return "bar"
    
    # 默认使用柱状图
    return "bar"

# 设置中文字体
def setup_chinese_font():
    # 尝试设置中文字体
    try:
        # Windows系统上常见的中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simkai.ttf',  # 楷体
            'C:/Windows/Fonts/STKAITI.TTF'  # 华文楷体
        ]
        
        # 检查字体文件是否存在
        font_file = None
        for path in font_paths:
            if os.path.exists(path):
                font_file = path
                break
        
        if font_file:
            # 设置全局字体
            chinese_font = fm.FontProperties(fname=font_file)
            plt.rcParams['font.family'] = chinese_font.get_name()
            # 设置全局字体属性
            mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return chinese_font
        else:
            print("未找到支持中文的字体文件")
            return None
    except Exception as e:
        print(f"设置中文字体时出错: {str(e)}")
        return None

# 生成图表
def generate_chart(data, chart_type, title):
    # 设置中文字体
    chinese_font = setup_chinese_font()
    
    plt.figure(figsize=(10, 6))
    if chinese_font:
        plt.title(title, fontproperties=chinese_font)
    else:
        plt.title(title)
    
    # 检查数据是否为空或None
    if data is None or (isinstance(data, dict) and len(data) == 0) or (isinstance(data, list) and len(data) == 0):
        # 如果数据为空，创建一个简单的图表
        plt.text(0.5, 0.5, "没有可用的数据", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        plt.close()
        return img_base64
    
    # 处理字典列表类型的数据
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        # 将字典列表转换为DataFrame
        df = pd.DataFrame(data)
        # 选择前两列作为x和y
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            data = {x_col: df[x_col].tolist(), y_col: df[y_col].tolist()}
        else:
            # 如果只有一列，创建索引作为x轴
            x_col = 'Index'
            y_col = df.columns[0]
            data = {x_col: list(range(len(df))), y_col: df[y_col].tolist()}
    
    # 根据图表类型生成不同的图表
    if chart_type == "bar":
        if isinstance(data, dict):
            # 确保数据类型正确
            try:
                keys = list(data.keys())[:10]
                values = [float(data[k]) if isinstance(data[k], (int, float, str)) else 0 for k in keys]
                plt.bar(keys, values)
                plt.xticks(rotation=45)
                # 设置中文字体
                if chinese_font:
                    plt.xticks(fontproperties=chinese_font)
                    plt.yticks(fontproperties=chinese_font)
                    plt.xlabel("数据标签", fontproperties=chinese_font)
                    plt.ylabel("数据值", fontproperties=chinese_font)
            except Exception as e:
                print(f"绘制柱状图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        elif isinstance(data, list):
            try:
                x = list(range(len(data)))
                values = [float(v) if isinstance(v, (int, float, str)) else 0 for v in data]
                plt.bar(x, values)
            except Exception as e:
                print(f"绘制柱状图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "数据类型不支持柱状图", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    elif chart_type == "line":
        if isinstance(data, dict):
            try:
                keys = list(data.keys())[:10]
                values = [float(data[k]) if isinstance(data[k], (int, float, str)) else 0 for k in keys]
                plt.plot(keys, values, marker='o')
                plt.xticks(rotation=45)
                # 设置中文字体
                if chinese_font:
                    plt.xticks(fontproperties=chinese_font)
                    plt.yticks(fontproperties=chinese_font)
                    plt.xlabel("数据标签", fontproperties=chinese_font)
                    plt.ylabel("数据值", fontproperties=chinese_font)
            except Exception as e:
                print(f"绘制折线图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        elif isinstance(data, list):
            try:
                x = list(range(len(data)))
                values = [float(v) if isinstance(v, (int, float, str)) else 0 for v in data]
                plt.plot(x, values, marker='o')
            except Exception as e:
                print(f"绘制折线图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "数据类型不支持折线图", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    elif chart_type == "pie":
        if isinstance(data, dict):
            try:
                keys = list(data.keys())[:5]
                values = [float(data[k]) if isinstance(data[k], (int, float, str)) else 0 for k in keys]
                # 确保所有值都是正数
                values = [max(0, v) for v in values]
                # 如果所有值都是0，添加一个非零值
                if sum(values) == 0:
                    values[0] = 1
                    
                if chinese_font:
                    plt.pie(values, labels=keys, autopct='%1.1f%%', textprops={'fontproperties': chinese_font})
                else:
                    plt.pie(values, labels=keys, autopct='%1.1f%%')
            except Exception as e:
                print(f"绘制饼图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        elif isinstance(data, list):
            try:
                values = [float(v) if isinstance(v, (int, float, str)) else 0 for v in data[:5]]
                # 确保所有值都是正数
                values = [max(0, v) for v in values]
                # 如果所有值都是0，添加一个非零值
                if sum(values) == 0:
                    values[0] = 1
                    
                labels = [f'项目 {i+1}' for i in range(len(values))] if chinese_font else [f'Item {i+1}' for i in range(len(values))]
                if chinese_font:
                    plt.pie(values, labels=labels, autopct='%1.1f%%', textprops={'fontproperties': chinese_font})
                else:
                    plt.pie(values, labels=labels, autopct='%1.1f%%')
            except Exception as e:
                print(f"绘制饼图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "数据类型不支持饼图", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    elif chart_type == "scatter":
        if isinstance(data, dict) and len(data) >= 2:
            try:
                keys = list(data.keys())
                x_values = data[keys[0]]
                y_values = data[keys[1]]
                
                # 确保数据类型正确
                if isinstance(x_values, list) and isinstance(y_values, list) and len(x_values) == len(y_values):
                    # 将所有值转换为浮点数
                    x_numeric = [float(x) if isinstance(x, (int, float, str)) else 0 for x in x_values]
                    y_numeric = [float(y) if isinstance(y, (int, float, str)) else 0 for y in y_values]
                    plt.scatter(x_numeric, y_numeric)
                else:
                    # 如果不是列表或长度不匹配，创建示例数据
                    plt.text(0.5, 0.5, "散点图数据类型不正确", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                
                if chinese_font:
                    plt.xlabel(keys[0], fontproperties=chinese_font)
                    plt.ylabel(keys[1], fontproperties=chinese_font)
                    plt.xticks(fontproperties=chinese_font)
                    plt.yticks(fontproperties=chinese_font)
                else:
                    plt.xlabel(keys[0])
                    plt.ylabel(keys[1])
            except Exception as e:
                print(f"绘制散点图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        elif isinstance(data, list):
            try:
                x = list(range(len(data)))
                values = [float(v) if isinstance(v, (int, float, str)) else 0 for v in data]
                plt.scatter(x, values)
                if chinese_font:
                    plt.xlabel("数据标签", fontproperties=chinese_font)
                    plt.ylabel("数据值", fontproperties=chinese_font)
            except Exception as e:
                print(f"绘制散点图时出错: {str(e)}")
                plt.text(0.5, 0.5, f"无法绘制图表: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "数据类型不支持散点图", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # 保存图表到内存
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    
    # 转换为base64编码
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close()
    
    return img_base64

# API端点 - 处理用户查询
@app.route('/api/process_query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': '查询不能为空'
            })
        
        # 记录查询
        print(f"处理查询: {user_query}")
        
        # 搜索网络获取相关信息
        search_results = search_web(user_query)
        
        if not search_results:
            return jsonify({
                'success': False,
                'error': '无法找到相关信息'
            })
        
        # 将搜索结果转换为文本格式，以便传递给LLM
        search_context = "\n\n".join([
            f"标题: {result.get('title', 'N/A')}\n链接: {result.get('link', 'N/A')}\n摘要: {result.get('snippet', 'N/A')}"
            for result in search_results
        ])
        
        # 使用DeepseekV3 API生成图表数据
        chart_data_response = process_with_llm(user_query, search_context)
        
        # 检查响应是否包含必要的字段
        if 'chart_type' not in chart_data_response or 'chart_title' not in chart_data_response or 'chart_data' not in chart_data_response:
            raise Exception("生成的图表数据不完整")
        
        # 提取图表信息
        chart_type = chart_data_response.get('chart_type', 'bar')
        chart_title = chart_data_response.get('chart_title', '数据可视化')
        chart_data = chart_data_response.get('chart_data', [])
        chart_labels = chart_data_response.get('chart_labels', [])
        analysis = chart_data_response.get('analysis', '无分析结果')
        
        # 检查图表数据是否为空
        if not chart_data:
            raise Exception("生成的图表数据为空")
        
        # 准备图表数据
        prepared_data = None
        
        # 检查数据类型并进行适当处理
        if isinstance(chart_data, list):
            # 如果数据是列表
            if chart_labels and len(chart_data) == len(chart_labels):
                # 如果有标签并且长度匹配，创建字典
                # 确保标签是可哈希类型
                valid_labels = []
                valid_data = []
                for i, label in enumerate(chart_labels):
                    if i < len(chart_data):
                        # 尝试将标签转换为字符串
                        try:
                            str_label = str(label)
                            valid_labels.append(str_label)
                            valid_data.append(chart_data[i])
                        except:
                            print(f"标签转换失败: {label}")
                
                prepared_data = dict(zip(valid_labels, valid_data))
            else:
                # 如果没有标签或长度不匹配，创建一个简单的字典
                prepared_data = {f"Item {i+1}": value for i, value in enumerate(chart_data)}
        elif isinstance(chart_data, dict):
            # 如果数据本身就是字典，确保键是字符串
            prepared_data = {str(k): v for k, v in chart_data.items()}
        else:
            # 如果数据类型不支持，抛出异常
            print(f"不支持的数据类型: {type(chart_data)}")
            raise Exception(f"不支持的数据类型: {type(chart_data)}")
        
        # 生成图表
        chart_image = generate_chart(prepared_data, chart_type, chart_title)
        
        # 返回结果
        return jsonify({
            'success': True,
            'chart_image': chart_image,
            'chart_type': chart_type,
            'chart_title': chart_title,
            'analysis': analysis
        })
        
    except Exception as e:
        print(f"处理查询时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'处理查询时出错: {str(e)}'
        })

# 静态文件服务
@app.route('/static/<path:path>')
def serve_static(path):
    return send_file(os.path.join(STATIC_DIR, path))

# 添加根路径路由，直接提供index.html
@app.route('/')
def index():
    return send_file('index.html')

# 添加其他静态文件路由
@app.route('/<path:path>')
def serve_file(path):
    if os.path.exists(path):
        return send_file(path)
    else:
        return send_file('index.html')  # 如果文件不存在，返回index.html

if __name__ == '__main__':
    print("服务器已启动，请访问: http://127.0.0.1:5000")
    app.run(debug=DEBUG, port=PORT, host='0.0.0.0')
