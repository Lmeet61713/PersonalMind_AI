# PersonalMind_AI 快速开始指南

## 🎯 项目简介

PersonalMind_AI 是一个基于 Streamlit 的智能对话伴侣应用，具备：
- ✅ 长期记忆管理（向量检索）
- ✅ 用户性格画像建模（五维模型）
- ✅ 自动事实抽取与记忆更新
- ✅ 智能表情包推荐
- ✅ 多会话管理

## 📋 前置要求

- Python >= 3.10
- DeepSeek API Key

## 🚀 快速启动

### 方法一：使用启动脚本（推荐）

```bash
# Windows 系统
双击运行 run.bat

# 首次运行时会自动安装依赖
# 按提示输入 DeepSeek API Key（或提前设置环境变量）
```

### 方法二：手动启动

```bash
# 1. 进入项目目录
cd "F:\PythonProject\OPC大赛\OPC大赛"

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置 API Key（Windows PowerShell）
$env:DEEPSEEK_API_KEY="your-api-key-here"

# 4. 启动应用
streamlit run app.py
```

## ⚙️ 配置说明

### 1. DeepSeek API Key 设置

**方式A：环境变量（推荐）**
```powershell
# Windows PowerShell（临时）
$env:DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxx"

# Windows 系统环境变量（永久）
# 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
# 新建系统变量: DEEPSEEK_API_KEY = your-api-key
```

**方式B：启动时输入**
- 运行 `run.bat` 时，如果未检测到环境变量，会提示输入

### 2. 嵌入模型配置

默认使用本地 BGE 模型：
```python
# config.py 第26行
EMBEDDING_MODEL_PATH = "C:/Users/ASUS/bge_model"
```

如需修改路径，编辑 `config.py` 文件。

### 3. 其他配置项

在 `config.py` 中可以调整：
```python
MAX_MEMORY_SUMMARIES = 400  # 最大记忆数量
DECAY_STEP_INTERVAL = 2     # 衰减机制触发间隔
```

## 📁 项目结构

```
PersonalMind_AI/
├── app.py                    # 主入口（Streamlit UI）
├── config.py                 # 全局配置
├── session_manager.py        # 会话管理
├── llm_client.py             # LLM API封装
├── memory_manager.py         # 记忆管理
├── persona_manager.py        # 性格画像管理
├── slice_utils.py            # 对话切片工具
├── utils.py                  # 通用工具函数
├── requirements.txt          # 依赖清单
├── run.bat                   # 启动脚本
├── MODULE_STRUCTURE.md       # 模块详细说明
└── QUICK_START.md           # 本文件
```

## 💡 核心功能使用

### 1. 开始新对话
- 点击侧边栏「开始新的聊天」按钮
- 系统自动生成时间戳命名的会话

### 2. 切换历史会话
- 侧边栏「历史会话」区域显示所有会话
- 点击会话名称即可加载
- 点击 ❌ 删除会话

### 3. 设置伴侣信息
- **昵称**：AI的称呼（默认：西格莉卡）
- **性格特点**：AI的性格设定（默认：阳光活泼的小女孩）

### 4. 自动功能开关
- **自动性格画像总结**：每5轮对话自动更新性格画像
- **自动事实抽取**：每5轮对话自动提取新事实

### 5. 手动操作
- **立即深度总结**：手动触发性格画像分析
- **立即抽取事实**：手动从对话中提取记忆

### 6. 记忆管理
- 查看记忆摘要列表
- 删除单条记忆
- 导出记忆库为JSON

### 7. 手动记录记忆
在聊天框输入：
```
/remember 我孙女叫小雨
```

### 8. 触发记忆检索
在对话中使用关键词：
- "记不记得..."
- "还记得吗..."
- "之前说过..."

系统会自动检索相关记忆并参考回答。

## 🔧 常见问题

### Q1: 启动时报错 "ModuleNotFoundError"
**解决**：运行 `pip install -r requirements.txt` 安装依赖

### Q2: 模型加载很慢
**解决**：
- 首次运行需要下载BGE模型（约400MB）
- 已配置国内镜像源，耐心等待即可
- 后续运行会从缓存加载

### Q3: API调用失败
**检查**：
1. API Key是否正确设置
2. 网络连接是否正常
3. DeepSeek账户余额是否充足

### Q4: 如何备份数据？
**方法**：
- 会话数据：`session/` 目录
- 记忆数据：`memory/` 目录
- 性格数据：`persona/` 目录
- 向量数据：`vectors/` 目录

直接复制这些文件夹即可备份。

### Q5: 如何重置应用？
**方法**：
- 删除 `session/`, `memory/`, `persona/`, `vectors/` 目录
- 重启应用即可从头开始

## 📊 性能优化建议

1. **内存限制**：`MAX_MEMORY_SUMMARIES` 设为400，避免过多记忆占用内存
2. **向量缓存**：系统自动预加载当前会话的所有向量到RAM
3. **消息渲染**：超过20条消息时，历史消息自动折叠

## 🎨 自定义扩展

### 修改表情包
在 `emojis/` 目录下添加图片：
```
emojis/
├── happy/    # 开心表情
├── sad/      # 难过表情
├── caring/   # 关心表情
└── default/  # 默认表情
```

### 修改背景图
替换 `background/bz.png` 文件

### 修改停用词表
编辑 `停用词.txt` 文件，每行一个词

## 📞 技术支持

如遇问题，请检查：
1. [MODULE_STRUCTURE.md](模块说明.md) - 模块详细说明
2. 控制台输出的错误信息
3. Python版本是否符合要求（>=3.10）

## 📝 更新日志

**v2.0 (2026-04-15)**
- ✅ 完成模块化重构
- ✅ 代码从单文件1817行拆分为8个模块
- ✅ 提升可维护性和可扩展性
- ✅ 添加启动脚本和文档

---

**祝使用愉快！** 🎉
