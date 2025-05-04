"""
Tendency 配置文件
包含API密钥和其他配置信息
"""

# DeepseekV3 API配置
DEEPSEEK_API_KEY = "sk-b8cadfb35d8a4666bfcf91b54c79d7f2"  # 替换为您的API密钥
API_BASE = "https://api.deepseek.com/v1"  # DeepseekV3 API基础URL

# Tavily API配置
TAVILY_API_KEY = "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"  # 请替换为您的实际Tavily API密钥

# 模型配置
MODEL_NAME = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000

# 应用配置
DEBUG = True
PORT = 5000
