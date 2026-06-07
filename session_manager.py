import streamlit as st
import os
import json
import shutil
from datetime import datetime
from config import SESSION_DIR, MEMORY_DIR, PERSONA_DIR, VECTORS_DIR


def generate_session():
    """生成对话名字"""
    return datetime.now().strftime("%Y_%m_%d %H_%M_%S")


def save_session():
    """保存当前会话的所有状态：主会话、记忆、性格，以及运行状态"""
    if not st.session_state.current_session:
        return

    # 1. 主会话文件（包含运行状态）
    main_data = {
        "nickname": st.session_state.nickname,
        "nature": st.session_state.nature,
        "current_session": st.session_state.current_session,
        "massage": st.session_state.massage,
        # 新增：保存会话运行状态
        "turn_count": st.session_state.turn_count,
        "auto_persona_summary": st.session_state.auto_persona_summary,
        "auto_fact_extract": st.session_state.auto_fact_extract,
        "show_persona": st.session_state.show_persona,
        # "enable_deep_summary": st.session_state.enable_deep_summary,
    }
    main_path = os.path.join(SESSION_DIR, f"{st.session_state.current_session}.json")
    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)

    # 2. 记忆文件
    memory_data = {
        "memory_summaries": st.session_state.memory_summaries,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    memory_path = os.path.join(MEMORY_DIR, f"{st.session_state.current_session}_memory.json")
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)

    # 3. 性格文件
    persona_data = {
        "user_persona": st.session_state.user_persona,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    persona_path = os.path.join(PERSONA_DIR, f"{st.session_state.current_session}_personal.json")
    with open(persona_path, "w", encoding="utf-8") as f:
        json.dump(persona_data, f, ensure_ascii=False, indent=2)


def show_session():
    """返回所有主会话文件名（不含 .json 后缀）"""
    if not os.path.exists(SESSION_DIR):
        return []
    files = [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
    # 确保只返回主会话文件（不包含 _memory、_personal 或 last_slices）
    session_names = [f[:-5] for f in files if not ("_memory" in f or "_personal" in f or "last_slices" in f)]
    session_names.sort(reverse=True)
    return session_names


def load_session(session_name):
    """从三个文件加载会话，恢复所有状态（包括运行开关和轮数），并预加载向量到RAM"""
    from memory_manager import _preload_vectors_to_ram
    
    main_path = os.path.join(SESSION_DIR, f"{session_name}.json")
    memory_path = os.path.join(MEMORY_DIR, f"{session_name}_memory.json")
    persona_path = os.path.join(PERSONA_DIR, f"{session_name}_personal.json")

    if not os.path.exists(main_path):
        return

    with open(main_path, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    st.session_state.nickname = main_data.get("nickname", "")
    st.session_state.nature = main_data.get("nature", "")
    st.session_state.current_session = session_name
    st.session_state.massage = main_data.get("massage", [])
    st.session_state.turn_count = main_data.get("turn_count", 0)
    st.session_state.auto_persona_summary = main_data.get("auto_persona_summary", False)
    st.session_state.auto_fact_extract = main_data.get("auto_fact_extract", False)
    st.session_state.show_persona = main_data.get("show_persona", True)
    # st.session_state.enable_deep_summary = main_data.get("enable_deep_summary", False)

    # 加载记忆文件
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
        st.session_state.memory_summaries = memory_data.get("memory_summaries", [])

        # 兼容旧数据：为每条记忆补充 start_turn, end_turn, vector_ref
        for mem in st.session_state.memory_summaries:
            if "start_turn" not in mem:
                mem["start_turn"] = mem.get("source_turn", -1)
            if "end_turn" not in mem:
                mem["end_turn"] = mem.get("source_turn", -1)
            if "vector_ref" not in mem:
                mem["vector_ref"] = ""
    else:
        st.session_state.memory_summaries = []

    # 加载性格文件
    if os.path.exists(persona_path):
        with open(persona_path, "r", encoding="utf-8") as f:
            persona_data = json.load(f)
        st.session_state.user_persona = persona_data.get("user_persona", {
            "personality_traits": {},
            "interests": {},
            "values": {},
            "communication_style": {},
            "habits": {}
        })
    else:
        st.session_state.user_persona = {
            "personality_traits": {},
            "interests": {},
            "values": {},
            "communication_style": {},
            "habits": {}
        }
    st.session_state["auto_persona_checkbox"] = st.session_state.auto_persona_summary
    st.session_state["auto_fact_checkbox"] = st.session_state.auto_fact_extract

    # 【优化1】预加载所有向量到RAM缓存
    _preload_vectors_to_ram()


def delete_session(session_name):
    """删除会话的三个关联文件，如果删除的是当前会话则重置界面（不自动创建新会话）"""
    main_path = os.path.join(SESSION_DIR, f"{session_name}.json")
    memory_path = os.path.join(MEMORY_DIR, f"{session_name}_memory.json")
    persona_path = os.path.join(PERSONA_DIR, f"{session_name}_personal.json")
    vectors_session_dir = os.path.join(VECTORS_DIR, session_name)

    for path in [main_path, memory_path, persona_path]:
        if os.path.exists(path):
            os.remove(path)

    # 如果删除的是当前会话，重置所有会话相关状态（不清空全局设置如昵称、性格模板）
    if st.session_state.current_session == session_name:
        st.session_state.massage = []
        st.session_state.user_persona = {
            "personality_traits": {},
            "interests": {},
            "values": {},
            "communication_style": {},
            "habits": {}
        }
        st.session_state.memory_summaries = []
        st.session_state.memory_embeddings = {}
        st.session_state.turn_count = 0
        st.session_state.current_session = None  # 表示没有当前会话
        # 注意：不要调用 save_session()，避免自动创建新会话文件
        
    # 删除向量文件夹
    if os.path.exists(vectors_session_dir):
        # 为什么不用remove
        shutil.rmtree(vectors_session_dir)
