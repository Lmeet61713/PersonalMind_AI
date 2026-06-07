import os
import uuid
import numpy as np
from config import BASE_DIR, VECTORS_DIR, embedding_model
import streamlit as st


def _get_vector_path(session_name: str, vector_ref: str = None):
    """根据会话名和相对路径（或直接生成新路径）返回绝对路径"""
    if vector_ref:
        return os.path.join(BASE_DIR, vector_ref)
    else:
        # 生成新路径: vectors/{session_name}/{uuid}.npy
        session_dir = os.path.join(VECTORS_DIR, session_name)
        os.makedirs(session_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.npy"
        rel_path = os.path.join("vectors", session_name, filename)
        return os.path.join(BASE_DIR, rel_path), rel_path


def _save_vector(memory_entry: dict, session_name: str):
    """为 memory_entry 生成向量并保存到文件，更新 vector_ref 字段"""
    # 【修复】检查 session_name 是否为 None
    if not session_name:
        print("[警告] session_name 为 None，跳过向量保存")
        return

    fact = memory_entry.get("fact", "")
    if not fact:
        return
    # 生成向量
    vec = embedding_model.encode(fact, normalize_embeddings=True)  # 归一化便于余弦相似度
    abs_path, rel_path = _get_vector_path(session_name)
    np.save(abs_path, vec)
    memory_entry["vector_ref"] = rel_path


def _load_vector(vector_ref: str):
    """根据相对路径加载向量"""
    if not vector_ref:
        return None
    abs_path = os.path.join(BASE_DIR, vector_ref)
    if os.path.exists(abs_path):
        return np.load(abs_path)
    return None


def _delete_vector(vector_ref: str):
    """删除向量文件（如果存在）"""
    if not vector_ref:
        return
    abs_path = os.path.join(BASE_DIR, vector_ref)
    if os.path.exists(abs_path):
        os.remove(abs_path)


def _preload_vectors_to_ram():
    """将当前会话的所有记忆向量预加载到RAM缓存中"""
    st.session_state.memory_embeddings = {}
    for idx, mem in enumerate(st.session_state.memory_summaries):
        ref = mem.get("vector_ref", "")
        if ref:
            vec = _load_vector(ref)
            if vec is not None:
                st.session_state.memory_embeddings[idx] = vec
            else:
                # 如果向量文件丢失，重新生成
                vec = embedding_model.encode(mem["fact"], normalize_embeddings=True)
                _save_vector(mem, st.session_state.current_session)
                st.session_state.memory_embeddings[idx] = vec

#关键词粗匹配
def retrieve_relevant_memories(user_message: str, top_k: int = 10) -> list:
    """
    根据用户消息中的关键词检索相关记忆摘要
    【优化】扩大候选集从3→10，增加召回率
    """
    user_lower = user_message.lower()
    scored = []
    for mem in st.session_state.memory_summaries:
        fact_lower = mem["fact"].lower()
        keywords = [k.lower() for k in mem.get("keywords", [])]
        score = 0.0
        # 关键词匹配
        for kw in keywords:
            if kw in user_lower:
                score += 0.4
        # 事实文本中的词匹配
        for word in user_lower.split():
            if len(word) > 1 and word in fact_lower:
                score += 0.2
        # 重要性加权
        score += mem["importance"] * 0.3
        scored.append((score, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored[:top_k] if _ > 0]

#向量精确匹配
def retrieve_by_vector(user_message: str, top_k: int = 5, coarse_top_n: int = 15):
    """
    先关键词粗匹配，再向量精细匹配。
    【优化】
    1. 扩大粗匹配候选集: 5→15
    2. 扩大最终返回数量: 2→5
    3. 向量相似度中加入 importance 权重
    4. 提取query关键词，给予额外加分
    """
    from utils import extract_keywords_from_text
    
    # 1. 关键词粗匹配，获取候选集（扩大到15个）
    coarse_candidates = retrieve_relevant_memories(user_message, top_k=coarse_top_n)
    if not coarse_candidates:
        return []

    # 2. 编码用户消息并提取关键词
    query_vec = embedding_model.encode(user_message, normalize_embeddings=True)
    query_keywords = extract_keywords_from_text(user_message, top_n=8)  # 提取更多关键词

    # 3. 对候选集计算向量相似度（优先使用RAM缓存）
    scored = []
    for mem in coarse_candidates:
        # 查找该记忆在memory_summaries中的索引
        mem_idx = None
        for idx, m in enumerate(st.session_state.memory_summaries):
            if m is mem:
                mem_idx = idx
                break

        # 优先从RAM缓存获取向量
        vec = st.session_state.memory_embeddings.get(mem_idx) if mem_idx is not None else None
        if vec is None:
            # 缓存未命中，从文件加载
            ref = mem.get("vector_ref", "")
            vec = _load_vector(ref) if ref else None
            if vec is None:
                # 如果文件也没有，临时生成
                vec = embedding_model.encode(mem["fact"], normalize_embeddings=True)
                _save_vector(mem, st.session_state.current_session)
            # 更新RAM缓存
            if mem_idx is not None:
                st.session_state.memory_embeddings[mem_idx] = vec

        # 计算余弦相似度
        cosine_sim = float(np.dot(query_vec, vec))

        # 【优化1】加入 importance 权重 (60%向量 + 25%importance + 15%关键词匹配)
        importance_score = mem.get("importance", 0.5)

        # 【优化3】关键词匹配加分
        keyword_bonus = 0.0
        mem_keywords = [k.lower() for k in mem.get("keywords", [])]
        for qkw in query_keywords:
            if qkw.lower() in mem_keywords or any(qkw.lower() in mk for mk in mem_keywords):
                keyword_bonus += 0.15  # 每个匹配关键词加0.15分
        keyword_bonus = min(keyword_bonus, 0.5)  # 最多加0.5分

        # 综合得分：60%向量相似度 + 25%importance + 15%关键词匹配
        final_score = 0.6 * cosine_sim + 0.25 * importance_score + 0.15 * keyword_bonus

        scored.append((final_score, mem))

    # 4. 按综合得分降序排序，返回 top_k（扩大到5个）
    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored[:top_k]]  # 返回最相关的 top_k 个记忆条目


def find_similar_memory(fact: str, threshold: float = 0.8):
    """
    在 memory_summaries 中查找与 fact 相似度最高的记忆。
    返回 (index, existing_entry, similarity) 或 (None, None, 0)
    """
    from utils import compute_text_similarity
    
    best_idx = None
    best_entry = None
    best_sim = 0.0
    for idx, entry in enumerate(st.session_state.memory_summaries):
        sim = compute_text_similarity(fact, entry["fact"])
        if sim > best_sim and sim >= threshold:
            best_sim = sim
            best_idx = idx
            best_entry = entry
    return best_idx, best_entry, best_sim


def add_memory_summary(fact: str, importance: float, keywords: list,
                       source_turn: int = -1, start_turn: int = None, end_turn: int = None):
    """添加或更新记忆摘要"""
    from datetime import datetime
    from session_manager import save_session
    from config import MAX_MEMORY_SUMMARIES
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 若未指定起止轮次，则使用 source_turn 作为起止（兼容旧调用）
    if start_turn is None:
        start_turn = source_turn
    if end_turn is None:
        end_turn = source_turn

    # 检查是否与已有记忆高度相似
    idx, existing, sim = find_similar_memory(fact, threshold=0.8)
    if idx is not None:
        # 合并：更新 importance、时间戳、关键词、事实描述，并合并轮次区间
        existing["importance"] = max(existing["importance"], importance)
        existing["timestamp"] = timestamp
        new_keywords = list(set(existing["keywords"] + keywords))
        existing["keywords"] = new_keywords
        if len(fact) > len(existing["fact"]):
            _delete_vector(existing.get("vector_ref", ""))
            _save_vector(existing, st.session_state.current_session)
            existing["fact"] = fact
        # 合并轮次区间
        existing["start_turn"] = min(existing.get("start_turn", source_turn), start_turn)
        existing["end_turn"] = max(existing.get("end_turn", source_turn), end_turn)
        # vector_ref 保留原有的（或可根据需要更新）
        print(f"[记忆更新] 合并相似事实：{existing['fact']} (importance={existing['importance']})")
        save_session()
        return

    # 新增记忆
    new_entry = {
        "fact": fact,
        "timestamp": timestamp,
        "importance": importance,
        "keywords": keywords,
        "source_turn": source_turn,  # 保留原字段（单条来源轮次）
        "start_turn": start_turn,  # 新增：覆盖起始轮
        "end_turn": end_turn,  # 新增：覆盖结束轮
        "vector_ref": ""  # 新增：张量存储位置（用户可自行填充）
    }
    st.session_state.memory_summaries.append(new_entry)
    # 新增记忆后立即生成向量
    _save_vector(new_entry, st.session_state.current_session)

    # 淘汰逻辑保持不变...
    if len(st.session_state.memory_summaries) > MAX_MEMORY_SUMMARIES:
        st.session_state.memory_summaries.sort(key=lambda x: (x["importance"], x["timestamp"]))
        excess = len(st.session_state.memory_summaries) - MAX_MEMORY_SUMMARIES
        st.session_state.memory_summaries = st.session_state.memory_summaries[excess:]
        st.session_state.memory_summaries.sort(key=lambda x: x["timestamp"])

    save_session()


def update_memory_with_new_facts(new_facts, start_turn=None, end_turn=None):
    """
    new_facts: list of dict; start_turn/end_turn: 可选，指定这批事实的轮次区间
    【简化】采用demo10的简洁逻辑，直接计算轮次区间并批量添加
    """
    from config import MAX_MEMORY_SUMMARIES
    from persona_manager import merge_similar_features_across_dims, truncate_dimensions, deduplicate_across_dimensions
    
    # 如果没有传入start_turn/end_turn，动态计算当前5轮区间
    if start_turn is None or end_turn is None:
        k = st.session_state.turn_count // 5
        start_turn = (k - 1) * 5 + 1
        end_turn = k * 5

    for fact_item in new_facts:
        fact_text = fact_item["fact"]
        importance = min(fact_item["importance"], 0.99)
        keywords = fact_item.get("keywords", [])

        # 直接使用传入的或计算的轮次区间
        add_memory_summary(
            fact=fact_text,
            importance=importance,
            keywords=keywords,
            source_turn=st.session_state.turn_count,
            start_turn=start_turn,
            end_turn=end_turn
        )

    # 淘汰逻辑（保持原有）
    if len(st.session_state.memory_summaries) > MAX_MEMORY_SUMMARIES:
        st.session_state.memory_summaries.sort(key=lambda x: (x["importance"], x["timestamp"]))
        excess = len(st.session_state.memory_summaries) - MAX_MEMORY_SUMMARIES
        st.session_state.memory_summaries = st.session_state.memory_summaries[excess:]
        st.session_state.memory_summaries.sort(key=lambda x: x["timestamp"])

    # 更新RAM缓存
    _preload_vectors_to_ram()
    from session_manager import save_session
    save_session()


def extract_new_facts_from_dialog(messages, existing_summaries, turn_count):
    """
    原有接口保持不变，内部调用统一函数
    【简化】不再预先设置start_turn和end_turn，由update_memory_with_new_facts动态计算
    """
    from llm_client import unified_extract_and_summarize
    
    result = unified_extract_and_summarize(
        messages=messages,
        current_persona={},  # 不需要画像
        existing_summaries=existing_summaries,
        turn_count=turn_count,
        mode="facts"
    )

    return result["new_facts"]
