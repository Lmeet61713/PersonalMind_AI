import os
from sentence_transformers import SentenceTransformer
import warnings
from openai import OpenAI
warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(__file__)
SESSION_DIR = os.path.join(BASE_DIR, "session")                     # 会话数据保存目录
MEMORY_DIR = os.path.join(BASE_DIR, "memory")                       # 记忆数据保存目录
PERSONA_DIR = os.path.join(BASE_DIR, "persona")                     # 性格画像保存目录
VECTORS_DIR = os.path.join(BASE_DIR, "vectors")                     # 向量数据库保存目录
SLICES_FILE = os.path.join(SESSION_DIR, "last_slices.json")         # 原文的切片文件
STOPWORDS_FILE = os.path.join(BASE_DIR, "停用词.txt")

# ==================== 常量配置 ====================
MAX_MEMORY_SUMMARIES = 400              #最大记忆摘要数量
DECAY_STEP_INTERVAL = 2                 #衰减机制步长

# ==================== 模型加载 ====================
# 环境变量获取API Key
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# #评审专用API接口，直接输入API
# api_key = "sk-***************************"
# client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# 嵌入模型路径（本地路径或HuggingFace模型名）
EMBEDDING_MODEL_PATH = "C:/Users/ASUS/bge_model"
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'                   # 镜像站
# embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")       # 或 bge-large-zh
# 加载嵌入模型
print("正在加载嵌入模型...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
print("模型加载完成。")

# ==================== 目录初始化 ====================
def init_directories():
    """创建必要的目录（如果不存在）"""
    for d in [SESSION_DIR, MEMORY_DIR, PERSONA_DIR, VECTORS_DIR]:
        os.makedirs(d, exist_ok=True)

# 启动时初始化目录
init_directories()
