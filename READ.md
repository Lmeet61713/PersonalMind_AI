# PersonalMind_AI 🧠💬

一个带有**长期记忆（事实摘要）**、**性格画像分析**与**向量化对话切片检索**的智能对话伴侣。  
基于 Streamlit + DeepSeek + BGE 嵌入模型，可实现上下文感知、个性化交互。

---

## ✨ 主要功能

- 💾 **长期记忆管理**
  - 自动从对话中抽取事实（偏好、计划、经历等）生成摘要
  - 支持手动添加记忆（/remember 指令）
  - 合并相似事实，避免冗余
  - 记忆向量化（BGE）与关键词混合检索

- 🧬 **性格画像分析（五维模型）**
  - 自动提取用户性格特征：`personality_traits`, `interests`, `values`, `communication_style`, `habits`
  - 特征置信度随对话动态调整（奖励/衰减机制）
  - 跨维度去重，防止特征膨胀

- 📚 **对话切片检索**
  - 当用户提及“还记得吗”、“之前说过”等触发词时，自动检索相关历史对话片段
  - 先关键词粗筛，后向量精排，提取完整对话上下文

- 🧵 **会话管理**
  - 多会话独立保存（主对话 + 记忆 + 性格画像）
  - 历史会话列表，支持切换/删除
  - 导出记忆库或对话切片为 JSON

- 🎭 **个性化交互**
  - 可自定义 AI 昵称、性格描述
  - 回复中自动添加颜文字与表情包（情绪匹配）
  - 背景图像自定义

---

## 🏗️ 项目结构

```
PersonalMind_AI/
├── app.py                # 主 Streamlit 应用入口，包含 UI 和聊天逻辑
├── config.py             # 路径、模型加载、常量等全局配置
├── session_manager.py    # 会话的创建、保存、加载、删除
├── memory_manager.py     # 记忆事实的抽取、存储、向量检索
├── persona_manager.py    # 性格画像的合并、裁剪、去重、衰减机制
├── slice_utils.py        # 对话切片的提取与临时存储
├── llm_client.py         # 与 DeepSeek API 交互的统一抽取函数
├── utils.py              # 通用工具：关键词提取、文本相似度计算
├── 停用词.txt            # 中文停用词表（可选，无文件时使用内置）
├── session/              # 会话文件存储
├── memory/               # 记忆摘要存储
├── persona/              # 性格画像存储
├── vectors/              # 记忆向量（.npy）存储
├── emojis/               # 表情包图片（按情绪分类）
└── background/           # 自定义背景图片
```

---

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- 安装依赖：

```bash
pip install streamlit openai jieba sentence-transformers numpy
```

### 2. 配置

#### ① 设置 API Key

创建系统环境变量，或在代码中直接填写：

```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxx"
```

或修改 `config.py` 中的 `DEEPSEEK_API_KEY` 变量。

#### ② 下载或指定 BGE 嵌入模型

默认使用本地路径 `C:/Users/ASUS/bge_model`（见 `config.py`）。  
推荐下载 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) 并修改 `EMBEDDING_MODEL_PATH` 为你的实际路径。

也可取消注释 `config.py` 中的镜像下载方式，直接从 Hugging Face 拉取。

#### ③ 表情包与背景（可选）

- 表情包放在 `emojis/` 下，按情绪分子文件夹 `happy`, `sad`, `caring`, `default`
- 背景图放在 `background/` 下，默认文件名为 `bz.png`

### 3. 启动

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可。

---

## 🧪 使用说明

- **基础聊天**：在底部输入框发送消息，AI 会根据设定的性格与记忆回复。
- **手动添加记忆**：发送 `/remember 事实内容`，例如 `/remember 我孙女叫小雨`
- **触发记忆检索**：当用户消息包含“记不记得”、“之前说过”等短语时，系统自动检索相关历史对话并注入回复中。
- **侧边栏控制**：
  - 更改 AI 昵称与性格描述
  - 查看/管理历史会话
  - 开启自动性格总结/事实抽取（每 5 轮自动触发）
  - 手动执行“立即深度总结”或“立即抽取事实”
  - 查看/删除记忆摘要，导出记忆库或切片

---

## 📦 依赖项

- `streamlit`
- `openai`（用于 DeepSeek API）
- `jieba`（中文分词）
- `sentence-transformers`（BGE 嵌入）
- `numpy`
- `pathlib`，`glob`，`json`（标准库）

---

## 📌 注意事项

- 本项目为演示/学习用途，未进行并发优化，适合单用户使用。
- 记忆上限默认为 400 条（`MAX_MEMORY_SUMMARIES`），可在 `config.py` 调整。
- 性格画像衰减机制基于 `DECAY_STEP_INTERVAL`，需配合多轮对话观察效果。
- 表情包路径可能出现跨平台问题，代码内置了路径修复逻辑。

---

## 📄 License

MIT License.

---

**PersonalMind_AI** —— 让你的 AI 越来越懂你 🌱