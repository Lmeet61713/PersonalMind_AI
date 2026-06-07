from difflib import SequenceMatcher
from collections import defaultdict


def merge_similar_features_across_dims(persona, similarity_threshold=0.8):
    """对每个维度内部合并相似特征，置信度取最大值"""
    new_persona = {}
    for dim, features in persona.items():
        new_persona[dim] = merge_similar_features(features, similarity_threshold)
    return new_persona


def truncate_dimensions(persona, max_features=10):
    """每个维度按置信度降序保留最多max_features个特征"""
    truncated = {}
    for dim, features in persona.items():
        if len(features) > max_features:
            sorted_items = sorted(features.items(), key=lambda x: x[1], reverse=True)[:max_features]  # 根据置信度降序排序
            truncated[dim] = dict(sorted_items)
        else:
            truncated[dim] = features
    return truncated


def deduplicate_across_dimensions(persona):
    """
    如果同一个特征名出现在多个维度，只保留置信度最高的那个维度中的特征，其他维度中删除。
    返回处理后的 persona。
    """
    # 收集所有特征名及其出现的维度及置信度
    feature_map = defaultdict(list)  # {feature: [(dim, conf), ...]}
    for dim, features in persona.items():
        for feat, conf in features.items():
            feature_map[feat].append((dim, conf))

    # 确定每个特征应保留的维度（置信度最高的）
    keep_in_dim = {}
    for feat, occurrences in feature_map.items():
        if len(occurrences) > 1:
            # 按置信度降序排序，取第一个
            best = max(occurrences, key=lambda x: x[1])
            keep_in_dim[feat] = best[0]

    # 构建新 persona，只保留允许的特征
    new_persona = {dim: {} for dim in persona}
    for dim, features in persona.items():
        for feat, conf in features.items():
            # 如果该特征需要去重且当前维度不是保留维度，则跳过
            if feat in keep_in_dim and keep_in_dim[feat] != dim:
                continue
            new_persona[dim][feat] = conf
    return new_persona


def format_persona_for_display(persona):
    """将画像中的置信度格式化为保留两位小数"""
    formatted = {}
    for dim, features in persona.items():
        formatted[dim] = {}
        for feat, conf in features.items():
            # 保留两位小数，直接格式化为浮点数
            formatted[dim][feat] = round(conf, 2)
    return formatted


def merge_similar_features(features_dict, threshold=0.8):
    """合并名称相似的特征，置信度取最大值"""
    feature_names = list(features_dict.keys())  # 提取所有特征名
    merged = {}  # 存储合并结果
    used = set()  # 一开始空集合--记录已处理的特征(防重复)
    for i, name1 in enumerate(feature_names):  # i是索引，name1是特征名
        if name1 in used:
            continue
        similar_group = [name1]
        for j, name2 in enumerate(feature_names):
            if i != j and name2 not in used:  # 避免重复
                sim = SequenceMatcher(None, name1, name2).ratio()  # SequenceMatcher().ratio()组合用法计算相似度
                if sim >= threshold:
                    # 相似度大于一定的阈值就放在相似 组
                    similar_group.append(name2)
        if len(similar_group) > 1:
            best_name = max(similar_group, key=lambda x: features_dict[x])  # 从相似组中找出置信度最高的特征名作为代表
            best_conf = max(features_dict[x] for x in similar_group)  # 找出相似组中的最大置信度
            merged[best_name] = best_conf
            used.update(similar_group)  # 更新已处理的特征
        else:
            # 没有相似特征，则直接添加
            merged[name1] = features_dict[name1]
            used.add(name1)
    return merged


def apply_decay_and_reward(current_persona, base_persona, decay_step, reward_increment=0.03, decay_decrement=0.05):
    """
    应用性格画像的衰减与奖励机制。

    参数:
        current_persona: 当前LLM提取的性格画像
        base_persona: 锚点画像快照（用于对比稳定性）
        decay_step: 当前是第几次深度总结
        reward_increment: 奖励增量（默认+0.03，用于强化稳定特征）
        decay_decrement: 衰减值（默认-0.05，用于弱化不稳定特征）

    返回:
        (processed_persona, should_update_base)
        processed_persona: 处理后的性格画像字典
        should_update_base: 布尔值，指示本轮结束后是否应该更新 base
    """
    from config import DECAY_STEP_INTERVAL
    
    # 判断是否触发衰减/奖励逻辑：(轮数 - 1) / 步长 == 0 (即余数为0)
    should_trigger = (decay_step - 1) % DECAY_STEP_INTERVAL == 0

    if not should_trigger:
        # 不触发时，直接返回当前画像，且不更新 base
        return current_persona, False

    if base_persona is None:
        # 第一次触发（通常是第1组），作为初始锚点，不应用衰减，但标记需要更新 base
        return current_persona, True

    processed_persona = {}
    decayed_features = []
    rewarded_features = []
    stability_threshold = 0.05  # 判定为"稳定"的置信度波动阈值

    # 遍历所有维度
    for dim in current_persona.keys():
        processed_persona[dim] = {}
        current_features = current_persona.get(dim, {})
        base_features = base_persona.get(dim, {})

        # 处理当前维度的每个特征
        for feat, conf in current_features.items():
            new_conf = conf
            matched_base_feat = None

            # 【匹配机制】检查 base 中是否有相同或相似的特征
            for base_feat, base_conf_val in base_features.items():
                sim = SequenceMatcher(None, feat, base_feat).ratio()
                if sim >= 0.8:
                    matched_base_feat = base_feat
                    break

            if matched_base_feat:
                base_conf = base_features[matched_base_feat]
                diff = conf - base_conf

                # 三态判定逻辑：
                # 1. 稳定强化：波动很小（<0.05），说明特征很稳，给予奖励
                if abs(diff) < stability_threshold:
                    new_conf = min(conf + reward_increment, 0.99)
                    rewarded_features.append(f"{dim}/{feat} (稳定+{reward_increment})")
                # 2. 明显衰退：置信度比 base 低了超过阈值，触发强衰减
                elif diff < -stability_threshold:
                    new_conf = max(conf - decay_decrement, 0.01)
                    decayed_features.append(f"{dim}/{feat} (衰退-{decay_decrement})")
                # 3. 明显增强：置信度比 base 高了超过阈值，说明用户近期表现突出，给予额外奖励
                else:
                    new_conf = min(conf + reward_increment * 0.5, 0.99)
                    rewarded_features.append(f"{dim}/{feat} (增强+{reward_increment * 0.5})")
            else:
                # 新特征：给予轻微的观察期衰减，防止噪声
                new_conf = max(new_conf - 0.03, 0.01)

            # 【垃圾回收】只保留置信度 > 0.1 的特征
            if new_conf > 0.1:
                processed_persona[dim][feat] = round(new_conf, 2)

    # 打印日志
    if decayed_features:
        print(f"[衰减机制] 第{decay_step}组触发衰减: {', '.join(decayed_features)}")
    if rewarded_features:
        print(f"[衰减机制] 第{decay_step}组触发强化: {', '.join(rewarded_features)}")

    # 触发衰减的轮次，结束后必须更新 base
    return processed_persona, True


def summarize_persona_with_llm(messages, current_persona):
    """原有接口保持不变，内部调用统一函数"""
    from llm_client import unified_extract_and_summarize
    
    result = unified_extract_and_summarize(
        messages=messages,
        current_persona=current_persona,
        existing_summaries=[],  # 不需要记忆
        turn_count=0,
        mode="persona",
        target_turn_ranges=None  # 性格更新不需要轮次区间
    )
    return result["updated_persona"]
