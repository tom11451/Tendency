# Tendency - 数据可视化工具

这是一个基于用户需求自动联网搜索、获取数据并生成图表的工具，由DeepseekV3 API提供智能分析支持，使用Tavily API进行高质量网络搜索。

## 功能特点

- 使用DeepseekV3 API进行智能数据分析
- 使用Tavily API进行高质量网络搜索
- 根据用户输入的需求自动进行网络搜索
- 从搜索结果中提取相关数据
- 根据数据特征自动选择最适合的图表类型
- 生成数据可视化图表和智能分析摘要

## 如何使用

1. 复制`.env.example`文件为`.env`并设置您的API密钥：
   ```
   cp .env.example .env
   ```
2. 在`.env`文件中设置您的DeepseekV3 API密钥和Tavily API密钥
3. 运行`start_server.bat`启动后端服务
4. 在浏览器中访问：http://127.0.0.1:5000
5. 在搜索框中输入您的需求，例如："中国近十年GDP增长趋势"
6. 点击"开始"按钮或按回车键提交需求
7. 系统将分析您的需求，自动搜索相关数据并生成图表和分析摘要

## 技术栈

- 前端：HTML, CSS, JavaScript
- 后端：Python, Flask
- 数据处理：Pandas, NumPy
- 数据可视化：Matplotlib, Seaborn
- 人工智能：DeepseekV3 API (通过OpenAI库调用)
- 网络搜索：Tavily API

## 安装要求

- Python 3.7+
- 必要的Python包（已在requirements.txt中列出）
- DeepseekV3 API密钥（需要在.env中设置）
- Tavily API密钥（需要在.env中设置）

## 环境变量配置

项目使用`.env`文件存储敏感配置信息。创建`.env`文件并设置以下变量：

```
# DeepseekV3 API配置
DEEPSEEK_API_KEY=your_deepseek_api_key
API_BASE=https://api.deepseek.com/v1

# Tavily API配置
TAVILY_API_KEY=your_tavily_api_key

# 模型配置
MODEL_NAME=deepseek-chat
DEFAULT_TEMPERATURE=0.2
DEFAULT_MAX_TOKENS=2000

# 应用配置
DEBUG=True
PORT=5000
```

## 开源说明

本项目采用MIT许可证开源。如果您想贡献代码，请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request

## 注意事项

- 第一次运行时，系统将自动安装必要的依赖包
- 请确保您的计算机已连接到互联网
- 搜索结果和数据提取可能受到网络状况和目标网站结构的影响
- 使用DeepseekV3 API和Tavily API需要有效的API密钥，请在.env中设置您的密钥
- **重要**：请勿将包含API密钥的.env文件提交到版本控制系统
