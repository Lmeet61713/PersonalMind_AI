import streamlit as st
from datetime import datetime


def extract_conversation_slice(start_turn: int, end_turn: int, text_only: bool = False) -> str:
    """
    从 st.session_state.massage 中提取指定轮次区间内的对话原文。
    轮次定义：第1轮为用户第1条消息+AI回复（如果存在）。

    参数:
        start_turn: 起始轮次
        end_turn: 结束轮次
        text_only: 是否只返回纯文本（不包含表情包路径等信息）
    """
    if start_turn < 1 or end_turn < start_turn:
        return ""
    slices = []
    for turn in range(start_turn, end_turn + 1):
        user_idx = 2 * (turn - 1)
        if user_idx >= len(st.session_state.massage):
            break
        user_msg = st.session_state.massage[user_idx]
        if user_msg["role"] == "user":
            slices.append(f"用户: {user_msg['content']}")
        else:
            continue  # 索引错位则跳过
        assistant_idx = user_idx + 1
        if assistant_idx < len(st.session_state.massage):
            assistant_msg = st.session_state.massage[assistant_idx]
            if assistant_msg["role"] == "assistant":
                # 【优化4】如果text_only=True，只提取文本内容
                content = assistant_msg['content']
                slices.append(f"AI: {content}")
    if not slices:
        return ""
    header = f"【对话片段 第{start_turn}轮 ~ 第{end_turn}轮】\n"
    return header + "\n".join(slices)


def save_slices_to_temp_file(slices_list):
    """slices_list: list of dict, 每个 dict 包含 start_turn, end_turn, timestamp, slice_text"""
    from config import SLICES_FILE
    import json
    
    data = {
        "session": st.session_state.current_session,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "slices": slices_list
    }
    with open(SLICES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_last_slices():
    """加载上一次保存的切片整合体，用于下载按钮"""
    from config import SLICES_FILE
    import json
    import os
    
    if os.path.exists(SLICES_FILE):
        with open(SLICES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
