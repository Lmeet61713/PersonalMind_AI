import os
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from utils import extract_keywords_from_text
import json

# 初始化DeepSeek客户端
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def unified_extract_and_summarize(
        messages: list,
        current_persona: dict,
        existing_summaries: list,
        turn_count: int,
        mode: str = "both",  # 自定义开关 "facts", "persona", "both"
        target_turn_ranges: list = None  # 【新增】目标轮次区间列表，如 [(11, 15), (16, 17)]
) -> dict:
    """
    一次 LLM 调用同时完成事实抽取和性格画像更新。

    参数:
        messages: 完整对话历史
        current_persona: 当前五维性格画像
        existing_summaries: 当前已有的记忆摘要列表
        turn_count: 当前对话轮数
        mode: "facts" 只抽取事实, "persona" 只更新画像, "both" 两者都做

    返回:
        dict: {
            "new_facts": list,  # 新的事实列表，mode不含facts时为空列表
            "updated_persona": dict  # 更新后的性格画像，mode不含persona时为原画像
        }
    """
    # 只取最近的消息（避免上下文过长）
    recent_messages = messages[-10:] if len(messages) > 10 else messages

    # 构建已有事实的文本列表，供 LLM 去重参考
    existing_facts_text = [item["fact"] for item in existing_summaries]
    existing_facts_str = "\n".join(f"- {fact}" for fact in existing_facts_text) if existing_facts_text else "无"

    # 构建当前性格画像的 JSON 字符串
    persona_str = json.dumps(current_persona, ensure_ascii=False, indent=2)

    # 根据 mode 构建不同的系统提示和输出格式要求
    if mode == "facts":
        instruction = "你只需要抽取新的事实，不要输出性格画像相关内容。"
        output_format = """
    输出格式：只输出一个 JSON 对象，包含一个字段 "new_facts"，值为事实数组。
    每个事实对象必须包含以下字段：
    - "fact": 事实文本
    - "importance": 重要性评分(0.3-0.9)
    - "keywords": 关键词列表

    【重要】不需要返回 turn_range 字段，轮次分配由系统自动处理。

    如果没有新事实，输出 {"new_facts": []}
    """
    elif mode == "persona":
        instruction = "你只需要更新性格画像，不要抽取事实。"
        output_format = """
    输出格式：只输出一个 JSON 对象，包含一个字段 "updated_persona"，值为五维性格画像字典。
    结构必须与输入的性格画像完全一致（五个维度：personality_traits, interests, values, communication_style, habits）。
    """
    else:  # both
        instruction = "你需要同时抽取新的事实和更新性格画像。"
        output_format = """
    输出格式：输出一个 JSON 对象，包含两个字段：
    - "new_facts": 事实数组，每个元素包含 "fact", "importance", "keywords"
    - "updated_persona": 更新后的五维性格画像字典（结构同输入）

    如果没有新的事实，则 "new_facts" 为空数组。
    """

    prompt = f"""
    你是一个精准的信息提取和分析助手。请根据以下对话历史，完成指定的任务。

    {instruction}

    ## 当前已有的记忆事实（用于去重）：
    {existing_facts_str}

    ## 当前性格画像（五维）：
    {persona_str}

    ## 对话历史（最近10条消息）：
    """
    for msg in recent_messages:
        # 只保留用户的消息减少干扰：有时候 AI 的回复可能会包含一些引导性或总结性的话，如果只关注用户透露的"事实"或"性格"，只看用户说的话会更纯粹
        if msg["role"] == "user":
            prompt += f"用户: {msg['content']}\n"

    prompt += f"""
    ## 任务要求：
    {output_format}

    1. 抽取用户消息中的任何**非纯寒暄**内容，包括：
   - 明确的事实（偏好、计划、经历）
   - 情感表达（喜欢、讨厌、想念）
   - 重复行为（如多次发同一表情，可总结为"喜欢用xx表情"）
   - 简单动作（"拍照"、"吃甜点"、"散步"）
    2. 对于短消息（如"耶✌"），如果能推断用户情绪（开心、兴奋），则抽取为"用户感到开心"，importance 给 0.4。
    3. 对于纯表情或重复表情，可抽取为"用户喜欢用xx表情表达情绪"，importance 给 0.30。
    4. 尽量避免遗漏：只要不是"你好"、"嗯"、"哦"这类无信息量消息，都尝试抽取。
    5. 重要性评分：明确事实 0.70~0.90，情感/情绪 0.40~0.60，行为模式 0.50~0.70。

    ### 性格画像更新规则（如果任务包含画像更新）：
    1. 置信度范围 0.01~0.99，不允许 1.0。
    2. 【重要】新特征初始化：对于对话中首次出现的特质，置信度请严格控制在 0.68~0.72 之间，作为观察期基准。
    3. 【重要】动态调整：
       - 如果用户多次提到或表现出某特质，置信度应显著高于 0.75。
       - 如果用户只是偶尔提及，置信度保持在 0.65~0.75。
       - 如果用户表现出与该特质相反的行为，请大幅降低置信度。
    4. 避免分数通胀：不要给所有特征都打 相同分数。必须有高有低，以反映真实的性格侧重。
    5. 对于已存在的特征：如果用户在对话中没有表现出对该特征的强烈否定或改变，请尽量保持其置信度稳定（与输入画像中的值接近）。
    6. 合并相似特征，每个维度最多保留 10 个特征。
    7. 只根据用户的消息分析，不要从 AI 回复推断。

    请严格按照上述输出格式返回 JSON，不要输出任何额外内容。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个精确的信息提取和分析助手，只输出指定的 JSON 格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        print(f"[统一提取] AI返回：{result}")
        data = json.loads(result)

        # 解析结果
        new_facts = []
        updated_persona = current_persona.copy()

        if mode in ("facts", "both"):
            facts_raw = data.get("new_facts", [])
            if isinstance(facts_raw, list):
                for item in facts_raw:
                    if isinstance(item, dict) and "fact" in item:
                        # 补全缺失字段
                        item.setdefault("importance", 0.6)
                        item.setdefault("keywords", extract_keywords_from_text(item["fact"]))

                        new_facts.append(item)

        if mode in ("persona", "both"):
            from persona_manager import merge_similar_features_across_dims, truncate_dimensions, deduplicate_across_dimensions
            
            if "updated_persona" in data and isinstance(data["updated_persona"], dict):
                updated_persona = data["updated_persona"]
                # 确保所有维度存在
                for dim in current_persona.keys():
                    if dim not in updated_persona:
                        updated_persona[dim] = {}

                # 【关键优化】先合并相似特征，提高泛化能力，防止因命名微调导致的特征分裂
                updated_persona = merge_similar_features_across_dims(updated_persona, similarity_threshold=0.8)
                updated_persona = truncate_dimensions(updated_persona, max_features=20)
                updated_persona = deduplicate_across_dimensions(updated_persona)

                # 置信度限幅与格式化
                for dim in updated_persona:
                    for feat, conf in updated_persona[dim].items():
                        if conf > 0.99:
                            updated_persona[dim][feat] = 0.99
                        updated_persona[dim][feat] = round(conf, 2)

        return {
            "new_facts": new_facts,
            "updated_persona": updated_persona
        }

    except Exception as e:
        print(f"[统一提取] 出错：{e}")
        return {
            "new_facts": [],
            "updated_persona": current_persona
        }


def merge_facts_with_llm(fact_texts: list) -> str:
    """调用 LLM 将多条事实合并成一句自然语言"""
    if not fact_texts:
        return ""
    if len(fact_texts) == 1:
        return fact_texts[0]

    prompt = f"""将以下多条关于用户的事实合并成**一句**连贯、简洁的话。不要遗漏重要信息，不要添加额外解释。

    事实列表：
    {chr(10).join(f'- {t}' for t in fact_texts)}

    要求：
    1、输出只包含合并后的话，不一定一有一句，但是要求简洁。（列如："用户有橘色的猫"和"用户的猫猫叫咪咪"可以合并成"用户有橘色的猫叫咪咪")）
    2、如果事实之间有矛盾（例如"用户有猫"和"用户没有猫"），保留最新的一条（按时间顺序），并在合并时注明"曾有过但现在没有了"之类的过渡。
    3、保持主语清晰（以"用户"为主语）。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个信息合并助手，只输出合并后的一句话。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        merged = response.choices[0].message.content.strip()
        return merged
    except Exception as e:
        print(f"[合并事实] LLM调用失败：{e}，使用简单拼接")
        return "；".join(fact_texts)
