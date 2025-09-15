#!/usr/bin/env python
"""
RAG-Anything 快速入门示例

这个脚本演示了如何使用 RAG-Anything 进行文档处理和查询：
1. 基本配置和初始化
2. 处理文档
3. 执行文本查询
4. 执行多模态查询

使用前请确保：
1. 已配置 .env 文件中的 API 密钥
2. 准备了要处理的测试文档
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# 导入必要的模块
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

# 加载环境变量
load_dotenv()

async def main():
    """
    主函数：演示 RAG-Anything 的基本使用流程
    """
    
    print("🚀 RAG-Anything 快速入门")
    print("=" * 50)
    
    # 1. 检查环境配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    if not api_key:
        print("❌ 配置检查失败！")
        print("\n📋 配置帮助:")
        print("1. 复制 env.example 为 .env")
        print("2. 推荐使用 SiliconFlow + DeepSeek-V3.1 (性价比最高)")
        print("3. 运行 'python test_api_config.py' 测试配置")
        print("\n🔗 获取API Key:")
        print("• SiliconFlow: https://siliconflow.cn/")
        print("\n🌟 推荐配置:")
        print("OPENAI_API_KEY=sk-your-siliconflow-api-key")
        print("OPENAI_BASE_URL=https://api.siliconflow.cn/v1")
        print("OPENAI_MODEL=Pro/deepseek-ai/DeepSeek-V3.1")
        print("OPENAI_EMBEDDING_MODEL=Pro/BAAI/bge-m3")
        return
    
    # 识别 API 供应商
    supplier = "未配置"
    if base_url:
        if "siliconflow.cn" in base_url:
            supplier = "SiliconFlow"
        else:
            supplier = "其他供应商"
    
    print(f"✅ API 配置检查完成")
    print(f"   供应商: {supplier}")
    print(f"   Base URL: {base_url}")
    print(f"   Model: {llm_model}")
    print(f"   API Key: {api_key[:10]}...")
    
    # 2. 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=os.getenv("ENABLE_IMAGE_PROCESSING", "true").lower() == "true",
        enable_table_processing=os.getenv("ENABLE_TABLE_PROCESSING", "true").lower() == "true",
        enable_equation_processing=os.getenv("ENABLE_EQUATION_PROCESSING", "true").lower() == "true",
    )
    
    print(f"✅ RAG 配置创建完成")
    print(f"   工作目录: {config.working_dir}")
    print(f"   解析器: {config.parser}")
    print(f"   解析方法: {config.parse_method}")
    
    # 3. 定义模型函数
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        """LLM 模型函数"""
        return openai_complete_if_cache(
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        """视觉模型函数"""
        if messages:
            # 多模态 VLM 增强查询格式
            return openai_complete_if_cache(
                os.getenv("VISION_MODEL", "gpt-4o"),
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            # 传统单图像格式
            return openai_complete_if_cache(
                os.getenv("VISION_MODEL", "gpt-4o"),
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
            # 纯文本格式
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    
    # 4. 定义嵌入函数
    embedding_func = EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
        max_token_size=int(os.getenv("MAX_TOKEN_SIZE", "8192")),
        func=lambda texts: openai_embed(
            texts,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            api_key=api_key,
            base_url=base_url,
        ),
    )
    
    print(f"✅ 模型函数配置完成")
    
    # 5. 初始化 RAGAnything
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
    
    # 6. 检查是否有测试文档
    test_files = []
    for ext in ["*.pdf", "*.docx", "*.pptx", "*.txt", "*.md"]:
        test_files.extend(Path(".").glob(ext))
    
    if not test_files:
        print("\n📄 未找到测试文档")
        print("请在当前目录放置一些文档文件 (PDF, DOCX, PPTX, TXT, MD)")
        print("\n💡 你可以：")
        print("   1. 下载一些示例文档")
        print("   2. 创建一个简单的文本文件进行测试")
        
        # 创建一个示例文档
        sample_doc = Path("sample_document.txt")
        if not sample_doc.exists():
            sample_content = """
# RAG-Anything 示例文档

## 什么是 RAG-Anything？

RAG-Anything 是一个基于 LightRAG 的多模态文档处理系统，它能够：

1. **多格式支持**：处理 PDF、Word、PowerPoint、图片等多种格式
2. **智能解析**：使用 MinerU 和 Docling 进行高质量文档解析
3. **多模态查询**：支持文本和图像的混合查询
4. **图谱构建**：自动构建知识图谱以增强检索效果

## 核心特性

- 端到端文档处理流水线
- 高级多模态内容理解
- 上下文感知的检索系统
- 批处理能力
- 灵活的配置选项

## 使用场景

1. 学术研究文献分析
2. 企业文档知识管理
3. 技术文档问答系统
4. 多媒体内容分析

这是一个用于测试 RAG-Anything 功能的示例文档。
            """
            sample_doc.write_text(sample_content, encoding='utf-8')
            print(f"\n✅ 已创建示例文档: {sample_doc}")
            test_files = [sample_doc]
    
    if test_files:
        print(f"\n📄 找到 {len(test_files)} 个测试文档:")
        for i, file in enumerate(test_files[:3], 1):  # 只显示前3个
            print(f"   {i}. {file.name}")
        
        # 选择第一个文档进行处理
        test_file = test_files[0]
        print(f"\n🔄 开始处理文档: {test_file.name}")
        
        try:
            # 处理文档
            await rag.process_document_complete(
                file_path=str(test_file),
                output_dir=os.getenv("OUTPUT_DIR", "./output"),
                parse_method=config.parse_method
            )
            print(f"✅ 文档处理完成")
            
            # 执行示例查询
            print(f"\n🔍 执行示例查询...")
            
            # 文本查询示例
            text_query = "这个文档的主要内容是什么？"
            print(f"\n📝 文本查询: {text_query}")
            
            text_result = await rag.aquery(
                query=text_query,
                mode="hybrid"
            )
            print(f"\n💬 查询结果:")
            print(f"{text_result}")
            
            print(f"\n🎉 快速入门演示完成！")
            print(f"\n📚 接下来你可以：")
            print(f"   1. 尝试不同的查询问题")
            print(f"   2. 处理更多文档")
            print(f"   3. 探索多模态查询功能")
            print(f"   4. 查看 examples/ 目录中的更多示例")
            
        except Exception as e:
            print(f"❌ 文档处理失败: {e}")
            print(f"\n💡 可能的解决方案：")
            print(f"   1. 检查 API 密钥是否正确")
            print(f"   2. 确认网络连接正常")
            print(f"   3. 检查文档格式是否支持")

if __name__ == "__main__":
    print("启动 RAG-Anything 快速入门...")
    asyncio.run(main())