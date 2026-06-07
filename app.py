import streamlit as st
import os
import json
import random
import glob
import base64
from pathlib import Path
from datetime import datetime

# 导入自定义模块
from config import BASE_DIR, MAX_MEMORY_SUMMARIES
from session_manager import generate_session, save_session, show_session, load_session, delete_session
from memory_manager import (
    extract_new_facts_from_dialog, 
    update_memory_with_new_facts, 
    retrieve_by_vector,
    add_memory_summary,
    _preload_vectors_to_ram
)
from persona_manager import (
    summarize_persona_with_llm,
    merge_similar_features_across_dims,
    truncate_dimensions,
    deduplicate_across_dimensions,
    format_persona_for_display,
    apply_decay_and_reward
)
from slice_utils import extract_conversation_slice, save_slices_to_temp_file, load_last_slices
from utils import extract_keywords_from_text

# ===========================================================================================================
# 初始化session_state状态
# ===========================================================================================================

# 自动功能开关
if "auto_persona_summary" not in st.session_state:
    st.session_state.auto_persona_summary = False  # 性格画像自动总结
if "auto_fact_extract" not in st.session_state:
    st.session_state.auto_fact_extract = False  # 事实自动抽取

# 记忆摘要列表：按时间追加，越新越靠后
if "memory_summaries" not in st.session_state:
    st.session_state.memory_summaries = []

# 初始化用户性格画像
if "user_persona" not in st.session_state:
    st.session_state.user_persona = {
        "personality_traits": {},
        "interests": {},
        "values": {},
        "communication_style": {},
        "habits": {}
    }
else:
    # 确保所有维度都存在，兼容旧数据时发现有不存在的键就创建并初始为空
    for dim in ["personality_traits", "interests", "values", "communication_style", "habits"]:
        if dim not in st.session_state.user_persona:
            st.session_state.user_persona[dim] = {}

# 计数器，用来记录对话轮数
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

# 【新增】性格画像衰减机制相关状态
if "persona_base" not in st.session_state:
    st.session_state.persona_base = None  # 上一轮性格画像快照
if "decay_step_counter" not in st.session_state:
    st.session_state.decay_step_counter = 0  # 衰减步长计数器（每完成一次深度总结+1）

# 缓存记忆的向量，避免重复计算
if "memory_embeddings" not in st.session_state:
    st.session_state.memory_embeddings = {}  # key: 记忆索引或id, value: 向量

# 初始化聊天信息，存储在session_state中
if "massage" not in st.session_state:
    st.session_state.massage = []  # 包括自己和大模型的结果

# 性格自动读取
if "nature" not in st.session_state:
    st.session_state.nature = """
  阳光活泼的小女孩

  """  # 默认值为空字符串

if "show_persona" not in st.session_state:
    st.session_state.show_persona = True
    
# 昵称
if "nickname" not in st.session_state:
    st.session_state.nickname = " 西格莉卡 "

# 会话标识----气泡
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session()  # 格式化时间为年月日时分秒


# =========================================================================================
# Streamlit页面配置
# =========================================================================================
st.set_page_config(
    page_title="PersonalMind_AI",  # 网页标题
    page_icon="🍁",
    layout="wide",  # 占满全部
    initial_sidebar_state="expanded",  # 控制侧边栏状态
    menu_items={}
)

# 标题
st.title("PersonalMind_AI")


# ==================================== 工具函数 ====================================
def set_background_image(image_path):
    """
    设置Streamlit应用的背景图片
    :param image_path: 图片路径（支持本地路径或URL）
    """
    # 判断是本地文件还是网络图片
    if image_path.startswith(('http://', 'https://')):
        bg_url = image_path
    else:
        # 本地图片需要转为base64编码
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        # 根据文件扩展名确定MIME类型
        ext = Path(image_path).suffix.lower()
        mime_type = "image/png" if ext == '.png' else "image/jpeg"
        bg_url = f"data:{mime_type};base64,{img_data}"

    # CSS 样式
    bg_css = f"""
    <style>
        /* 主容器背景 */
        .stApp {{
            background: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* 确保聊天消息可读（添加半透明遮罩层） */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);  /* 半透明遮罩，可调节透明度 */
            z-index: -1;
        }}

        /* 聊天输入框背景透明 */
        [data-testid="stBottom"] > div {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 10px;
        }}

        /* 聊天消息容器透明 */
        .stChatMessage {{
            background: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            margin: 5px 0;
        }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)


def validate_and_fix_emoji_path(emoji_path):
    """
    验证并修复表情包路径
    - 如果路径存在,直接返回
    - 如果路径不存在但文件名有效,尝试在当前项目中查找
    - 如果都找不到,返回 None
    """
    if not emoji_path:
        return None
    
    # 如果路径已存在,直接返回
    if os.path.exists(emoji_path):
        return emoji_path
    
    # 提取文件名
    filename = os.path.basename(emoji_path)
    
    # 在当前项目的 emojis 目录下搜索
    emojis_root = os.path.join(BASE_DIR, "emojis")
    if os.path.exists(emojis_root):
        for root, dirs, files in os.walk(emojis_root):
            if filename in files:
                found_path = os.path.join(root, filename)
                print(f"[表情包路径修复] 找到替代路径: {found_path}")
                return found_path
    
    # 如果都找不到,返回 None
    print(f"[表情包路径修复] 无法找到表情包: {filename}")
    return None


def choose_emoji_by_keywords(text):
    """
    根据文本中的关键词,选择对应情绪文件夹中的随机一张图
    返回图片的相对路径
    """
    # 关键词到文件夹的映射
    keyword_to_category = {
        "开心": "happy",
        "笑": "happy",
        "喜欢": "happy",
        "真好": "happy",

        "难过": "sad",
        "伤心": "sad",
        "哭": "sad",
        "心疼": "sad",

        "关心": "caring",
        "注意": "caring",
        "小心": "caring",
        "没事吧": "caring"
    }

    # 默认分类
    category = "default"

    # 遍历关键词,看文本中包含哪个
    for keyword, cat in keyword_to_category.items():  # 遍历字典取得键和值
        if keyword in text:
            category = cat
            break

    print(f"[表情包] 匹配到的关键词:{category}")  # 看看分类对不对

    # 获取对应文件夹下所有图片文件
    folder_path = os.path.join(BASE_DIR, "emojis", category)

    print(f"[表情包] 文件夹路径:{folder_path}")

    if not os.path.exists(folder_path):
        print("[表情包] 文件夹不存在!")
        return None  # 文件夹不存在就不显示

    image_files = glob.glob(os.path.join(folder_path, "*.*"))  # 所有文件
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    print(f"[表情包] 找到的图片文件:{image_files}")

    if not image_files:
        print("[表情包] 文件夹内没有图片文件")
        return None
    chosen = random.choice(image_files)
    print(f"[表情包] 选中的图片:{chosen}")

    # 随机选一张
    return random.choice(image_files)


# 系统提示词模板
system_prompt_template = """
你叫 %s，现在是用户真实伴侣，请完全代入角色。
规则：1.回复的内容要充分体现伴侣性格特点
      2.回复的内容带上丰富的颜文字
      3.如果回复的内容与用户的置信度相关也可以结合用户的性格特点进行回复
      4.回复的同时可以根据用户的性格特点提问内容，勾起用户的聊天兴趣
      5.你记得下面关于用户的事实，请优先参考它们来回答，不要质疑这些事实。
你的性格是 %s
"""


# ==================================== 侧边栏UI ====================================
with st.sidebar:
    st.subheader("💬AI控制面板")
    
    # 开始新的会话
    if st.button("开始新的聊天", ...):
        # 【修复】不要先保存旧会话，直接生成新会话名并重置状态
        new_session_name = generate_session()

        # 重置所有状态
        st.session_state.massage = []
        st.session_state.current_session = new_session_name
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
        st.session_state.auto_persona_summary = False
        st.session_state.auto_fact_extract = False
        st.session_state.show_persona = True

        # 【修复】只保存一次新会话的空状态
        save_session()

        # 【优化】使用st.rerun()重新渲染，避免UI阻塞
        st.rerun()

    # 历史会话和删除
    st.text("历史会话")
    session_list = show_session()  # 返回所有会话列表
    for session in session_list:
        col1, col2 = st.columns([4, 1])  # 创建两个变量解包接收，按4：1分配大小
        with col1:
            # 加载会话信息--三元运算符判断是不是当前会话
            if st.button(session, width="stretch", icon="📃", key=f"show_session{session}",
                         type="primary" if session == st.session_state.current_session else "secondary"):
                load_session(session)  # 加载会话信息
                st.rerun()  # 重置界面
        with col2:
            if st.button("", width="stretch", icon="❌", key=f"delete_session{session}"):
                delete_session(session)
                st.rerun()

    # 分割线
    st.divider()

    st.subheader("伴侣信息")
    nick_name = st.text_input("昵称", placeholder="请输入昵称", value=st.session_state.nickname)
    # 如果前面已经赋值过，则直接赋值给st.session_state.nickname
    if nick_name:
        st.session_state.nickname = nick_name
    nature = st.text_area("你的性格特点", placeholder="请输入性格特点", value=st.session_state.nature)
    if nature:
        st.session_state.nature = nature
    
    # 分割线
    st.divider()

    if st.session_state.show_persona:
        with st.container(border=True):
            st.markdown("#### 当前用户性格画像")
            if any(st.session_state.user_persona.values()):
                display_persona = format_persona_for_display(st.session_state.user_persona)
                st.json(display_persona)
                # 【衰减机制】显示衰减状态
                st.caption(
                    f"🔄 衰减步长: {st.session_state.decay_step_counter} | 📊 特征总数: {sum(len(v) for v in st.session_state.user_persona.values())}")
            else:
                st.info("尚未提取到用户性格特征")

    st.divider()
    st.subheader("自动总结/提取")
    
    # Streamlit 要求每个 widget 的 key 在 session_state 中是唯一且权威的数据源。
    # 侧边栏中的checkbox
    st.checkbox(
        "自动性格画像总结（每5轮）",
        key="auto_persona_summary"
    )
    st.checkbox(
        "自动事实抽取（每5轮）",
        key="auto_fact_extract"
    )

    if st.button("立即深度总结"):
        with st.spinner("AI 正在分析你的性格..."):
            updated_persona = summarize_persona_with_llm(st.session_state.massage, st.session_state.user_persona)

            # 对更新的画像进行处理
            updated_persona = merge_similar_features_across_dims(updated_persona, similarity_threshold=0.8)
            updated_persona = truncate_dimensions(updated_persona, max_features=20)
            updated_persona = deduplicate_across_dimensions(updated_persona)

            # 置信度限幅与格式化
            for dim in updated_persona:
                for feat, conf in updated_persona[dim].items():
                    if conf > 0.99:
                        updated_persona[dim][feat] = 0.99
                    updated_persona[dim][feat] = round(conf, 2)

            # 【衰减机制】先增加计数器，再判断是否触发
            current_step_num = st.session_state.decay_step_counter + 1
            processed_persona, should_update = apply_decay_and_reward(
                current_persona=updated_persona,
                base_persona=st.session_state.persona_base,
                decay_step=current_step_num
            )

            st.session_state.user_persona = processed_persona

            # 【衰减机制】只有触发时才轮换 base
            if should_update:
                st.session_state.persona_base = processed_persona.copy()
                st.session_state.decay_step_counter = current_step_num
                st.success(f"性格画像已更新并刷新锚点！（当前锚点轮次: {st.session_state.decay_step_counter}）")
            else:
                st.success("性格画像已更新（本轮为过渡轮次，未刷新锚点）")

    if st.button("立即抽取事实（记忆摘要）"):
        with st.spinner("正在从对话中抽取新事实..."):
            new_facts = extract_new_facts_from_dialog(
                st.session_state.massage,
                st.session_state.memory_summaries,
                st.session_state.turn_count
            )
            if new_facts:
                # 【简化】直接调用update_memory_with_new_facts，由它动态计算轮次区间
                update_memory_with_new_facts(new_facts)
                st.success(f"已学习到 {len(new_facts)} 条新事实！")
            else:
                st.info("没有发现新的事实")

    st.sidebar.write(f"📝 记忆摘要数量：{len(st.session_state.memory_summaries)} / {MAX_MEMORY_SUMMARIES}")

    st.divider()
    st.subheader("📚 记忆管理")

    # 可折叠的记忆列表
    with st.expander(f"查看记忆摘要 ({len(st.session_state.memory_summaries)} 条)", expanded=False):
        if not st.session_state.memory_summaries:
            st.caption("暂无任何记忆。")
        else:
            # 倒序显示（最新的在上）
            for idx, mem in enumerate(reversed(st.session_state.memory_summaries)):
                col1, col2 = st.columns([5, 1])
                with col1:
                    # 显示事实文本，带重要性标记
                    importance_star = "⭐" * min(3, int(mem["importance"] * 3) + 1)
                    st.caption(f"{importance_star} {mem['fact'][:80]}")
                    st.caption(
                        f"📅 {mem['timestamp'][:16]}  |  🔑 {', '.join(mem.get('keywords', [])[:3])}  |  🎯 轮次 [{mem.get('start_turn', '?')}-{mem.get('end_turn', '?')}]")
                with col2:
                    # 删除按钮
                    if st.button("❌", key=f"del_mem_{idx}", help="删除此记忆"):
                        # 删除对应的原始索引（因为 reversed，需要映射回原列表索引）
                        original_idx = len(st.session_state.memory_summaries) - 1 - idx
                        deleted = st.session_state.memory_summaries.pop(original_idx)
                        save_session()
                        st.success(f"已删除记忆：{deleted['fact'][:50]}...")
                        st.rerun()
                st.divider()

    # 导出记忆按钮
    if st.button("📥 导出记忆库 (JSON)", use_container_width=True):
        if st.session_state.memory_summaries:
            export_data = {
                "session": st.session_state.current_session,
                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "memory_summaries": st.session_state.memory_summaries
            }
            export_json = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="点击下载 JSON 文件",
                data=export_json,
                file_name=f"memory_export_{st.session_state.current_session}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("当前没有记忆可导出")
    st.divider()

    if st.button("📥 下载上一轮切片整合体", use_container_width=True):
        slices_data = load_last_slices()
        if slices_data:
            # 将数据转为 JSON 字符串并提供下载
            json_str = json.dumps(slices_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="点击下载 JSON 文件",
                data=json_str,
                file_name=f"last_slices_{st.session_state.current_session}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("暂无切片数据，请先触发一次记忆检索。")


# ==================================== 主聊天界面 ====================================
set_background_image(os.path.join(BASE_DIR, "background", "bz.png"))  # 本地图片

# logo
logo_path = os.path.join(BASE_DIR, "fll.jpg")
if os.path.exists(logo_path):
    st.logo(logo_path)

# 展示聊天消息------根据角色进行分类输出内容记录
st.text(f"当前会话是:{st.session_state.current_session}")

# 【性能优化】只渲染最近20条对话,更早的放入可折叠区域
total_messages = len(st.session_state.massage)
if total_messages > 20:
    # 计算需要折叠的历史消息数量
    history_count = total_messages - 20

    # 将历史消息放入可折叠区域
    with st.expander(f"📜 查看历史对话 ({history_count} 条)", expanded=False):
        for massage in st.session_state.massage[:history_count]:
            with st.chat_message(massage["role"]):
                st.write(massage["content"])
                if massage["role"] == "assistant" and massage.get("emoji"):
                    fixed_emoji_path = validate_and_fix_emoji_path(massage["emoji"])
                    if fixed_emoji_path:
                        st.image(fixed_emoji_path, width=80)

    # 渲染最近20条消息(正常显示)
    for massage in st.session_state.massage[history_count:]:
        with st.chat_message(massage["role"]):
            st.write(massage["content"])
            if massage["role"] == "assistant" and massage.get("emoji"):
                fixed_emoji_path = validate_and_fix_emoji_path(massage["emoji"])
                if fixed_emoji_path:
                    st.image(fixed_emoji_path, width=80)
else:
    # 总消息数 <= 20,全部正常渲染
    for massage in st.session_state.massage:
        with st.chat_message(massage["role"]):
            st.write(massage["content"])
            if massage["role"] == "assistant" and massage.get("emoji"):
                fixed_emoji_path = validate_and_fix_emoji_path(massage["emoji"])
                if fixed_emoji_path:
                    st.image(fixed_emoji_path, width=80)


# ==================================== 聊天输入处理 ====================================
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

prompt = st.chat_input("你想说点什么？")
if prompt:

    # ========== 手动记忆命令处理 ==========
    if prompt.strip().startswith("/remember"):
        # 提取命令后面的内容
        fact_text = prompt.strip()[len("/remember"):].strip()
        if fact_text:
            # 自动生成关键词
            keywords = extract_keywords_from_text(fact_text)
            # 添加记忆，importance=0.9，source_turn 使用当前轮次+1
            add_memory_summary(
                fact=fact_text,
                importance=0.8,
                keywords=keywords,
                source_turn=st.session_state.turn_count + 1
            )
            # 构造确认回复
            confirm_msg = f"好的，我记住了：{fact_text}"
            # 直接显示回复（不调用大模型）
            with st.chat_message("assistant"):
                st.write(confirm_msg)
            # 将确认消息也存入对话历史（可选）
            st.session_state.massage.append({"role": "assistant", "content": confirm_msg, "emoji": None})
            # 保存会话（add_memory_summary 已经调用过 save_session，但确认消息没存）
            save_session()
            # 停止后续处理（不再调用大模型）
            st.rerun()
        else:
            # 命令后没有内容，提示错误
            with st.chat_message("assistant"):
                st.write("请告诉我你想让我记住什么，例如：/remember 我孙女叫小雨")
            st.session_state.massage.append(
                {"role": "assistant", "content": "请告诉我你想让我记住什么，例如：/remember 我孙女叫小雨", "emoji": None})
            save_session()
            st.rerun()
    else:
        st.chat_message("user").write(prompt)
        st.session_state.massage.append({"role": "user", "content": prompt})

        # 【重要优化3】增加对话轮次
        st.session_state.turn_count += 1

        # ========== 判断是否触发记忆检索 ==========
        memory_trigger_keywords = ["记不记得", "记得", "回忆", "之前说过", "还记不记得", "还记得吗"]
        user_msg_lower = prompt.lower()
        triggered = any(kw in user_msg_lower for kw in memory_trigger_keywords)

        # 【优化4】根据是否触发决定最近对话轮数和是否返回纯文本
        if triggered:
            recent_turns = 5  # 【优化】触发时取最近5轮（原来3轮）
            # 检索记忆并提取切片
            relevant_memories = retrieve_by_vector(prompt, top_k=5, coarse_top_n=15)  # 【优化】扩大检索范围
            slices_data = []
            slice_texts = []
            for mem in relevant_memories:
                start = mem.get("start_turn")
                end = mem.get("end_turn")
                if start is not None and end is not None and start > 0 and end >= start:
                    # 【优化】向前延伸3轮（原来2轮），向后延伸1轮
                    extended_start = max(1, start - 3)
                    extended_end = min(end + 1, st.session_state.turn_count)
                    slice_text = extract_conversation_slice(extended_start, extended_end, text_only=True)
                    if slice_text:
                        slice_texts.append(slice_text)
                        slices_data.append({
                            "start_turn": extended_start,
                            "end_turn": extended_end,
                            "original_start": start,
                            "original_end": end,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "slice_text": slice_text
                        })
            # 保存切片到临时文件（覆盖）
            if slices_data:
                save_slices_to_temp_file(slices_data)
                memory_section = "\n\n## 相关回忆的原始对话片段：\n" + "\n\n".join(slice_texts)
            else:
                memory_section = "\n\n## 提示：\n未找到相关记忆的原始对话片段。"
        else:
            recent_turns = 50  # 常态下取最近50轮
            memory_section = ""  # 常态下不加记忆切片

        # 4.构造用户性格描述文本
        user_persona_desc = "用户性格特征："
        if any(st.session_state.user_persona.values()):
            for dim, features in st.session_state.user_persona.items():
                if features:
                    # 筛选置信度>0.5的特征
                    high_conf = [f"{feat}({conf:.2f})" for feat, conf in features.items() if conf > 0.5]
                    if high_conf:
                        # 将维度名翻译成中文
                        dim_cn = {
                            "personality_traits": "性格特质",
                            "interests": "兴趣爱好",
                            "values": "价值观",
                            "communication_style": "沟通风格",
                            "habits": "生活习惯"
                        }.get(dim, dim)
                        user_persona_desc += f"{dim_cn}: {', '.join(high_conf)}; "
        else:
            user_persona_desc += "暂无"

        system_content = system_prompt_template % (st.session_state.nickname, st.session_state.nature)
        system_content += "\n" + user_persona_desc

        if memory_section:
            # 【关键】添加强制性指令，让 LLM 必须参考切片
            system_content += "\n\n【重要】以下是与当前对话相关的历史记忆片段，请务必参考这些内容来回答用户的问题：\n"
            system_content += memory_section
            system_content += "\n\n请基于以上记忆片段，结合用户的当前问题，给出连贯、准确的回复。如果记忆片段中有相关信息，请优先引用。"

        # 【优化3】构建消息历史（截取最近 recent_turns 轮，只包含纯文本）
        max_messages = recent_turns * 2
        recent_messages = st.session_state.massage[-max_messages:] if max_messages > 0 else []

        # 【优化4】如果触发了记忆检索，只保留文本内容，去除表情包等额外信息
        if triggered:
            cleaned_messages = []
            for msg in recent_messages:
                cleaned_msg = {"role": msg["role"], "content": msg["content"]}
                cleaned_messages.append(cleaned_msg)
            recent_messages = cleaned_messages

        # 【优化3】调用API（使用包含记忆切片的system_content和清理后的recent_messages）
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_content},
                *recent_messages
            ],
            stream=True
        )

        # 流式大模型输出结果
        response_massage = st.empty()

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                response_massage.chat_message("assistant").write(full_response)

        # 流式输出完成后(已经显示完文字)
        emoji_path = choose_emoji_by_keywords(full_response)
        if emoji_path:
            # 验证路径有效性
            fixed_emoji_path = validate_and_fix_emoji_path(emoji_path)
            if fixed_emoji_path:
                if fixed_emoji_path.lower().endswith('.gif'):
                    with open(fixed_emoji_path, "rb") as f:
                        gif_b64 = base64.b64encode(f.read()).decode()
                    st.markdown(f'<img src="data:image/gif;base64,{gif_b64}" width="150">', unsafe_allow_html=True)
                else:
                    st.image(fixed_emoji_path, width=150)
                # 使用修复后的路径保存到会话
                emoji_path = fixed_emoji_path
        
        # 【重要优化3】AI回复完成后,再将回复添加到massage
        st.session_state.massage.append({"role": "assistant", "content": full_response, "emoji": emoji_path})

        # 【重要优化3】现在检查是否是第5轮（AI回复后），如果是则标记延迟任务
        if st.session_state.turn_count % 5 == 0:
            need_persona = st.session_state.auto_persona_summary
            need_facts = st.session_state.auto_fact_extract

            if need_persona or need_facts:
                if need_persona and need_facts:
                    mode = "both"
                    msg = "正在后台同步更新性格画像和记忆..."
                elif need_persona:
                    mode = "persona"
                    msg = "正在后台更新性格画像..."
                else:
                    mode = "facts"
                    msg = "正在后台从对话中学习新事实..."

                from llm_client import unified_extract_and_summarize

                def deferred_update_task(messages, persona, summaries, turn_count, mode_val):
                    """延迟执行的更新任务（在下次渲染时执行）"""
                    try:
                        result = unified_extract_and_summarize(
                            messages=messages,
                            current_persona=persona,
                            existing_summaries=summaries,
                            turn_count=turn_count,
                            mode=mode_val
                        )

                        if mode_val in ("facts", "both") and result["new_facts"]:
                            return {"type": "update_complete", "result": result, "mode": mode_val}
                        elif mode_val in ("persona", "both"):
                            return {"type": "update_complete", "result": result, "mode": mode_val}
                        else:
                            return {"type": "no_update"}
                    except Exception as e:
                        return {"type": "error", "error": str(e)}

                # 【修复】计算目标轮次区间 - 简单单区间逻辑
                k = st.session_state.turn_count // 5
                target_start = (k - 1) * 5 + 1
                target_end = k * 5
                target_ranges = [(target_start, target_end)]

                if "pending_deferred_tasks" not in st.session_state:
                    st.session_state.pending_deferred_tasks = []
                st.session_state.pending_deferred_tasks.append({
                    "func": deferred_update_task,
                    "args": (
                        st.session_state.massage.copy(),
                        st.session_state.user_persona.copy(),
                        st.session_state.memory_summaries.copy(),
                        st.session_state.turn_count,
                        mode
                    ),
                    "mode": mode,
                    "submitted_at": st.session_state.turn_count
                })

                st.toast(msg, icon="⏳")

        # 【优化2】检查并执行已标记的延迟任务
        if "pending_deferred_tasks" in st.session_state and st.session_state.pending_deferred_tasks:
            tasks_to_remove = []
            for task_info in st.session_state.pending_deferred_tasks:
                try:
                    result = task_info["func"](*task_info["args"])

                    if result["type"] == "update_complete":
                        res = result["result"]
                        mode_val = result["mode"]

                        if mode_val in ("facts", "both") and res["new_facts"]:
                            # 【简化】直接使用每条fact自带的start_turn和end_turn
                            update_memory_with_new_facts(res["new_facts"])
                            st.toast(f"✅ 已学习到 {len(res['new_facts'])} 条新事实！", icon="✨")
                        elif mode_val in ("facts", "both"):
                            st.toast("ℹ️ 没有发现新的事实", icon="💭")

                        if mode_val in ("persona", "both"):
                            # 【衰减机制】获取LLM返回的原始画像
                            raw_persona = res["updated_persona"]
                            
                            # 对画像进行处理
                            raw_persona = merge_similar_features_across_dims(raw_persona, similarity_threshold=0.8)
                            raw_persona = truncate_dimensions(raw_persona, max_features=20)
                            raw_persona = deduplicate_across_dimensions(raw_persona)

                            # 置信度限幅与格式化
                            for dim in raw_persona:
                                for feat, conf in raw_persona[dim].items():
                                    if conf > 0.99:
                                        raw_persona[dim][feat] = 0.99
                                    raw_persona[dim][feat] = round(conf, 2)

                            # 【衰减机制】先增加计数器，再判断是否触发
                            current_step_num = st.session_state.decay_step_counter + 1
                            processed_persona, should_update = apply_decay_and_reward(
                                current_persona=raw_persona,
                                base_persona=st.session_state.persona_base,
                                decay_step=current_step_num
                            )

                            # 更新会话状态
                            st.session_state.user_persona = processed_persona

                            # 【衰减机制】只有触发时才更新 base 并增加计数器
                            if should_update:
                                st.session_state.persona_base = processed_persona.copy()
                                st.session_state.decay_step_counter = current_step_num
                                st.toast(f"✅ 性格画像已更新！锚点已刷新至第{current_step_num}组", icon="🎭")
                            else:
                                st.toast("✅ 性格画像已更新（本轮为过渡轮次）", icon="✨")

                            save_session()
                    elif result["type"] == "error":
                        st.toast(f"❌ 更新失败：{result['error']}", icon="⚠️")
                except Exception as e:
                    st.toast(f"❌ 任务异常：{str(e)}", icon="⚠️")

                tasks_to_remove.append(task_info)

            for task in tasks_to_remove:
                st.session_state.pending_deferred_tasks.remove(task)

        # 保存会话信息
        save_session()
