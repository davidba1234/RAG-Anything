#!/usr/bin/env python
"""
RAG-Anything 高级功能演示

这个脚本演示了 RAG-Anything 的高级功能：
1. 批处理多个文档
2. 多模态内容处理
3. 自定义查询模式
4. 性能优化技巧

使用前请确保：
1. 已配置 .env 文件中的 API 密钥
2. 准备了多个测试文档
"""

import os
import asyncio
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# 导入必要的模块
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from raganything.batch_parser import BatchParser

# 加载环境变量
load_dotenv()

def create_test_documents():
    """
    创建测试文档用于演示
    """
    test_dir = Path("./test_documents")
    test_dir.mkdir(exist_ok=True)
    
    documents = {
        "技术文档.txt": """
# 人工智能技术概述

## 机器学习
机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习模式。

### 主要类型
1. 监督学习：使用标记数据训练模型
2. 无监督学习：从未标记数据中发现模式
3. 强化学习：通过奖励机制学习最优策略

### 应用领域
- 图像识别
- 自然语言处理
- 推荐系统
- 自动驾驶

## 深度学习
深度学习使用多层神经网络模拟人脑处理信息的方式。

### 核心技术
- 卷积神经网络 (CNN)
- 循环神经网络 (RNN)
- 变换器 (Transformer)
- 生成对抗网络 (GAN)
        """,
        
        "商业报告.txt": """
# 2024年AI市场分析报告

## 执行摘要
人工智能市场在2024年继续快速增长，预计市场规模将达到5000亿美元。

## 市场趋势

### 增长驱动因素
1. **企业数字化转型**
   - 自动化需求增加
   - 效率提升要求
   - 成本控制压力

2. **技术成熟度提升**
   - 算法性能改进
   - 计算成本降低
   - 开发工具完善

3. **应用场景扩展**
   - 医疗健康
   - 金融服务
   - 制造业
   - 教育培训

## 市场细分

| 领域 | 市场份额 | 增长率 |
|------|----------|--------|
| 机器学习平台 | 35% | 25% |
| 计算机视觉 | 20% | 30% |
| 自然语言处理 | 18% | 28% |
| 机器人技术 | 15% | 22% |
| 其他 | 12% | 20% |

## 投资建议
建议重点关注以下领域的投资机会：
- 生成式AI应用
- 边缘计算AI芯片
- 行业专用AI解决方案
        """,
        
        "研究论文.md": """
# 大语言模型在文档处理中的应用研究

## 摘要
本研究探讨了大语言模型（LLM）在多模态文档处理中的应用效果，通过对比实验验证了RAG技术的有效性。

## 1. 引言

随着信息技术的快速发展，文档处理自动化成为企业提高效率的重要手段。传统的文档处理方法存在以下局限性：

- 处理格式单一
- 语义理解能力有限
- 无法处理多模态内容
- 缺乏上下文关联

## 2. 相关工作

### 2.1 检索增强生成（RAG）
RAG技术结合了检索和生成两种方法的优势：
- 提高了知识的时效性
- 增强了回答的准确性
- 支持大规模知识库

### 2.2 多模态处理
多模态文档处理需要同时理解：
- 文本内容
- 图像信息
- 表格数据
- 公式符号

## 3. 方法论

### 3.1 系统架构
我们提出的系统包含以下组件：
1. 文档解析模块
2. 多模态内容提取
3. 知识图谱构建
4. 检索增强查询

### 3.2 实验设计
- 数据集：1000份多模态文档
- 评估指标：准确率、召回率、F1分数
- 对比方法：传统检索、纯生成模型

## 4. 实验结果

实验结果表明，我们的方法在各项指标上都有显著提升：

- 准确率提升 15%
- 召回率提升 20%
- 处理速度提升 30%

## 5. 结论

本研究证明了大语言模型在文档处理中的巨大潜力，特别是在多模态内容理解方面。未来工作将重点关注：
- 模型效率优化
- 更多模态的支持
- 实时处理能力
        """,
        
        "产品手册.txt": """
# RAG-Anything 产品使用手册

## 产品概述
RAG-Anything 是一款基于大语言模型的智能文档处理系统，支持多种格式的文档解析和智能问答。

## 核心功能

### 1. 文档解析
- 支持 PDF、Word、PowerPoint、图片等格式
- 自动提取文本、图像、表格、公式
- 保持原始文档结构和格式

### 2. 智能问答
- 基于文档内容的精准问答
- 支持多轮对话
- 提供引用来源

### 3. 知识管理
- 自动构建知识图谱
- 支持知识更新和维护
- 提供可视化界面

## 使用流程

### 步骤1：环境配置
1. 安装 Python 3.8+
2. 安装依赖包
3. 配置 API 密钥

### 步骤2：文档上传
1. 选择要处理的文档
2. 设置解析参数
3. 开始解析处理

### 步骤3：智能查询
1. 输入查询问题
2. 选择查询模式
3. 获取答案结果

## 最佳实践

### 文档准备
- 确保文档清晰可读
- 避免过度复杂的格式
- 合理组织文档结构

### 查询技巧
- 使用具体明确的问题
- 提供足够的上下文
- 利用多轮对话深入探讨

### 性能优化
- 合理设置批处理大小
- 根据硬件调整并发数
- 定期清理缓存文件

## 故障排除

### 常见问题
1. **解析失败**
   - 检查文档格式
   - 确认文件完整性
   - 尝试不同解析模式

2. **查询无结果**
   - 检查问题表述
   - 确认文档已处理
   - 尝试不同查询模式

3. **性能问题**
   - 检查系统资源
   - 调整配置参数
   - 优化文档大小

## 技术支持
如需技术支持，请联系：
- 邮箱：support@raganything.com
- 文档：https://docs.raganything.com
- 社区：https://community.raganything.com
        """
    }
    
    created_files = []
    for filename, content in documents.items():
        file_path = test_dir / filename
        file_path.write_text(content, encoding='utf-8')
        created_files.append(file_path)
        print(f"✅ 创建测试文档: {filename}")
    
    return created_files

async def demo_batch_processing(rag: RAGAnything, file_paths: List[Path]):
    """
    演示批处理功能
    """
    print("\n🔄 批处理演示")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 批处理文档
        await rag.batch_process_documents(
            file_paths=[str(f) for f in file_paths],
            output_dir="./batch_output",
            batch_size=2,  # 每批处理2个文档
            concurrency=2  # 并发数为2
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ 批处理完成")
        print(f"   处理文档数: {len(file_paths)}")
        print(f"   处理时间: {processing_time:.2f} 秒")
        print(f"   平均速度: {len(file_paths)/processing_time:.2f} 文档/秒")
        
    except Exception as e:
        print(f"❌ 批处理失败: {e}")

async def demo_advanced_queries(rag: RAGAnything):
    """
    演示高级查询功能
    """
    print("\n🔍 高级查询演示")
    print("=" * 50)
    
    queries = [
        {
            "query": "人工智能的主要应用领域有哪些？",
            "mode": "hybrid",
            "description": "混合检索模式"
        },
        {
            "query": "2024年AI市场的增长驱动因素是什么？",
            "mode": "global",
            "description": "全局图谱检索"
        },
        {
            "query": "RAG技术的优势有哪些？",
            "mode": "local",
            "description": "局部图谱检索"
        },
        {
            "query": "如何优化文档处理性能？",
            "mode": "naive",
            "description": "简单向量检索"
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        print(f"\n📝 查询 {i}: {query_info['description']}")
        print(f"问题: {query_info['query']}")
        
        try:
            start_time = time.time()
            result = await rag.aquery(
                query=query_info['query'],
                mode=query_info['mode']
            )
            end_time = time.time()
            
            print(f"\n💬 回答 ({end_time - start_time:.2f}秒):")
            print(f"{result}")
            print(f"\n{'='*30}")
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")

async def demo_multimodal_features(rag: RAGAnything):
    """
    演示多模态功能（如果有相关内容）
    """
    print("\n🎨 多模态功能演示")
    print("=" * 50)
    
    # 查询包含表格信息的内容
    table_query = "AI市场各领域的市场份额和增长率是多少？"
    print(f"📊 表格查询: {table_query}")
    
    try:
        result = await rag.aquery(
            query=table_query,
            mode="hybrid"
        )
        print(f"\n💬 回答:")
        print(f"{result}")
        
    except Exception as e:
        print(f"❌ 多模态查询失败: {e}")

async def demo_performance_comparison():
    """
    演示不同配置的性能对比
    """
    print("\n⚡ 性能对比演示")
    print("=" * 50)
    
    configs = [
        {
            "name": "基础配置",
            "parser": "mineru",
            "parse_method": "auto",
            "enable_image_processing": False,
            "enable_table_processing": False,
        },
        {
            "name": "完整配置",
            "parser": "mineru",
            "parse_method": "auto",
            "enable_image_processing": True,
            "enable_table_processing": True,
        }
    ]
    
    for config_info in configs:
        print(f"\n🔧 测试配置: {config_info['name']}")
        print(f"   解析器: {config_info['parser']}")
        print(f"   图像处理: {config_info['enable_image_processing']}")
        print(f"   表格处理: {config_info['enable_table_processing']}")
        
        # 这里可以添加实际的性能测试代码
        print(f"   预估性能: {'高' if config_info['enable_image_processing'] else '中'}")

async def main():
    """
    主函数：运行所有演示
    """
    print("🚀 RAG-Anything 高级功能演示")
    print("=" * 60)
    
    # 检查环境配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        print("❌ 错误：请在 .env 文件中配置 OPENAI_API_KEY")
        return
    
    print(f"✅ 环境配置检查完成")
    
    # 创建测试文档
    print(f"\n📄 准备测试文档...")
    test_files = create_test_documents()
    
    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./advanced_demo_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # 定义模型函数
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )
    
    # 初始化 RAGAnything
    try:
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        print(f"✅ RAGAnything 初始化成功")
    except Exception as e:
        print(f"❌ RAGAnything 初始化失败: {e}")
        return
    
    # 运行演示
    try:
        # 1. 批处理演示
        await demo_batch_processing(rag, test_files)
        
        # 2. 高级查询演示
        await demo_advanced_queries(rag)
        
        # 3. 多模态功能演示
        await demo_multimodal_features(rag)
        
        # 4. 性能对比演示
        await demo_performance_comparison()
        
        print(f"\n🎉 所有演示完成！")
        print(f"\n📚 接下来你可以：")
        print(f"   1. 尝试处理自己的文档")
        print(f"   2. 调整配置参数优化性能")
        print(f"   3. 集成到自己的应用中")
        print(f"   4. 探索更多高级功能")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print(f"\n💡 可能的解决方案：")
        print(f"   1. 检查 API 密钥和网络连接")
        print(f"   2. 确认文档格式正确")
        print(f"   3. 调整批处理参数")

if __name__ == "__main__":
    print("启动 RAG-Anything 高级功能演示...")
    asyncio.run(main())