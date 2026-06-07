import os
import re
import jieba
from config import STOPWORDS_FILE

# 全局停用词缓存
STOPWORDS_CACHE = None


def load_stopwords():
    """从文件加载中文停用词表"""
    stopwords = set()
    if STOPWORDS_CACHE is not None:
        return STOPWORDS_CACHE
    
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:  # 跳过空行
                    stopwords.add(word)
    else:
        # 如果文件不存在，使用内置基础停用词
        stopwords = {'的', '了', '是', '我', '你', '他', '她', '它', '们', '吗', '呢', '吧', '啊', '哦', '嗯',
                     '在', '和', '与', '或', '但', '而', '这', '那', '有', '也', '就', '都', '很', '要', '会',
                     '可以', '什么', '怎么', '为什么', '因为', '所以', '如果', '虽然', '但是', '而且'}
    return stopwords


def get_stopwords():
    """获取停用词集合（带缓存）"""
    global STOPWORDS_CACHE
    if STOPWORDS_CACHE is None:
        STOPWORDS_CACHE = load_stopwords()
    return STOPWORDS_CACHE


def extract_keywords_from_text(text: str, top_n: int = 5) -> list:
    """
    从文本中提取关键词，用于记忆摘要的 keywords 字段。
    【优化】使用完整停用词表 + jieba 分词
    """
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)

    try:
        # 使用 jieba 分词
        words = jieba.lcut(text)
        # 获取停用词表
        stopwords = get_stopwords()
        # 过滤停用词和单字
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        # 去重并保留前 top_n 个
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords[:top_n]
    except:
        # fallback: 简单按空格切分，取长度>1的词
        words = text.split()
        stopwords = get_stopwords()
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        return list(dict.fromkeys(keywords))[:top_n]


def compute_text_similarity(text1: str, text2: str) -> float:
    """计算两个事实文本的相似度，返回 0~1。使用 Jaccard + 包含关系"""
    if text1 == text2:
        return 1.0
    if text1 in text2 or text2 in text1:
        return 0.9  # 包含关系视为高度相似
    set1 = set(jieba.lcut(text1))
    set2 = set(jieba.lcut(text2))
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union
