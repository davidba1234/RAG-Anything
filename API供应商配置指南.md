# API 供应商配置指南

本指南将帮助您配置SiliconFlow API供应商，以便在RAG-Anything项目中使用DeepSeek和其他优质模型。

## 推荐配置：SiliconFlow

### SiliconFlow（推荐）
**特点：**
- 提供多种优质开源模型
- 支持DeepSeek-V3.1最新模型
- 价格实惠，性价比高
- API兼容OpenAI格式
- 稳定可靠的服务

**配置方法：**
在 `.env` 文件中设置：
```bash
OPENAI_API_KEY=sk-your-siliconflow-api-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Pro/deepseek-ai/DeepSeek-V3.1
OPENAI_EMBEDDING_MODEL=Pro/BAAI/bge-m3
```

**模型配置：**
```bash
# 推荐模型配置
LLM_MODEL=Pro/deepseek-ai/DeepSeek-V3.1          # 主要对话模型
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct       # 视觉理解模型
EMBEDDING_MODEL=Pro/BAAI/bge-m3                  # 嵌入模型
```

**获取API Key：**
1. 访问 [SiliconFlow官网](https://siliconflow.cn/)
2. 注册账号并完成认证
3. 在控制台创建API Key
4. 查看可用模型列表和定价
5. 新用户通常有免费额度

**模型说明：**
- **DeepSeek-V3.1**：最新的DeepSeek模型，具有优秀的推理能力和中文理解
- **Qwen2.5-VL-72B**：强大的多模态视觉理解模型
- **BGE-M3**：高质量的多语言嵌入模型，支持中英文

## 配置步骤

### 1. 获取 SiliconFlow API Key
1. 访问 [SiliconFlow官网](https://siliconflow.cn/)
2. 注册账号并完成认证
3. 在控制台创建 API Key
4. 查看可用模型列表和定价
5. 新用户通常有免费额度

### 2. 配置环境变量
编辑 `.env` 文件，设置SiliconFlow配置：

```bash
# SiliconFlow + DeepSeek-V3.1 配置
OPENAI_API_KEY=sk-your-siliconflow-api-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Pro/deepseek-ai/DeepSeek-V3.1
OPENAI_EMBEDDING_MODEL=Pro/BAAI/bge-m3

LLM_MODEL=Pro/deepseek-ai/DeepSeek-V3.1
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
EMBEDDING_MODEL=Pro/BAAI/bge-m3
```

### 3. 测试配置
运行快速测试：

```python
import os
from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("LLM_MODEL")

try:
    result = openai_complete_if_cache(
        model=model,
        prompt="你好，请介绍一下自己",
        api_key=api_key,
        base_url=base_url
    )
    print(f"✅ SiliconFlow API 配置成功！")
    print(f"模型回复: {result}")
except Exception as e:
    print(f"❌ API 配置失败: {e}")
```

## 成本优势

使用SiliconFlow + DeepSeek-V3.1的组合具有以下优势：

- **极高性价比**：比OpenAI官方API便宜90%以上
- **优质模型**：DeepSeek-V3.1具有接近GPT-4的能力
- **中文优化**：对中文理解和生成能力优秀
- **稳定服务**：SiliconFlow提供稳定可靠的API服务
- **无需翻墙**：国内可直接访问
- **丰富模型**：提供多种优质开源模型选择

## 模型性能

**DeepSeek-V3.1 性能表现：**

| 能力维度 | 性能评级 | 说明 |
|----------|----------|------|
| 中文能力 | ⭐⭐⭐⭐⭐ | 优秀的中文理解和生成能力 |
| 英文能力 | ⭐⭐⭐⭐ | 接近GPT-4的英文处理能力 |
| 代码能力 | ⭐⭐⭐⭐⭐ | 强大的代码理解和生成能力 |
| 推理能力 | ⭐⭐⭐⭐⭐ | 优秀的逻辑推理和问题解决能力 |
| 多模态 | ⭐⭐⭐⭐ | 支持文本和视觉理解 |

## 推荐配置

### 标准配置（推荐）
```bash
# SiliconFlow + DeepSeek-V3.1 - 最佳性价比
OPENAI_API_KEY=sk-your-siliconflow-api-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Pro/deepseek-ai/DeepSeek-V3.1
OPENAI_EMBEDDING_MODEL=Pro/BAAI/bge-m3

LLM_MODEL=Pro/deepseek-ai/DeepSeek-V3.1
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
EMBEDDING_MODEL=Pro/BAAI/bge-m3
```

### 高性能配置
```bash
# 使用更强大的模型组合
OPENAI_API_KEY=sk-your-siliconflow-api-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Pro/Qwen/Qwen2.5-72B-Instruct
OPENAI_EMBEDDING_MODEL=Pro/BAAI/bge-m3

LLM_MODEL=Pro/Qwen/Qwen2.5-72B-Instruct
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
EMBEDDING_MODEL=Pro/BAAI/bge-m3
```

## 常见问题

### Q: 为什么推荐使用SiliconFlow？
A: SiliconFlow提供了最佳的性价比，支持DeepSeek-V3.1等优质模型，价格比OpenAI便宜90%以上，且国内可直接访问。

### Q: DeepSeek-V3.1模型有什么优势？
A: DeepSeek-V3.1具有优秀的中文理解能力、强大的代码生成能力和推理能力，性能接近GPT-4但成本极低。

### Q: 如何获取SiliconFlow的API Key？
A: 访问 [SiliconFlow官网](https://siliconflow.cn/)，注册账号后在控制台创建API Key，新用户通常有免费额度。

### Q: API 调用失败怎么办？
A: 检查以下几点：
1. SiliconFlow API Key 是否正确
2. 账户是否有余额
3. 网络连接是否正常
4. 模型名称是否正确（注意Pro/前缀）
5. Base URL 是否为 https://api.siliconflow.cn/v1

### Q: 如何监控 API 使用量？
A: 在SiliconFlow控制台可以查看详细的使用量和费用统计，建议定期检查避免超额。

### Q: 支持哪些模型类型？
A: SiliconFlow支持文本生成、视觉理解、嵌入等多种模型，推荐使用DeepSeek-V3.1作为主要对话模型。

## 总结

**SiliconFlow + DeepSeek-V3.1** 是RAG-Anything项目的最佳选择：

### 核心优势
1. **极高性价比** - 比OpenAI便宜90%以上
2. **优秀性能** - DeepSeek-V3.1具有接近GPT-4的能力
3. **中文优化** - 专门优化的中文理解和生成能力
4. **访问便利** - 国内可直接访问，无需特殊网络环境
5. **模型丰富** - 提供多种优质开源模型选择

### 快速开始
1. 访问 [SiliconFlow官网](https://siliconflow.cn/) 获取API Key
2. 按照本指南配置环境变量
3. 运行测试脚本验证配置
4. 开始使用RAG-Anything构建您的智能应用

选择SiliconFlow，让您以最低的成本享受最优质的AI服务！