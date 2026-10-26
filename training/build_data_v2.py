"""experimental-v2: v1 records plus length-stratified originals.

The split is inherited, not recomputed. validation.jsonl and test.jsonl are copied
byte for byte from v1 so that validation loss stays comparable with the completed
9b-bf16-sft-v1 run and the held-out sets keep the same meaning. New families are
train-only; their behaviour is measured by the independent evaluation fixtures
(evaluate.py), never by their own text.
"""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import build_data
import length_scenarios
from common import LOCAL, save_json
from data_utils import read_rows


def assistant_lengths(rows):
    return [len(m['content'].strip()) for r in rows for m in r['messages']
            if m['role'] == 'assistant' and m.get('content', '').strip()]


def describe(rows):
    lengths = sorted(assistant_lengths(rows))
    return {
        'records': len(rows),
        'assistant_replies': len(lengths),
        'median_chars': statistics.median(lengths),
        'mean_chars': round(statistics.mean(lengths), 1),
        'p90_chars': lengths[int(len(lengths) * 0.9)],
        'max_chars': lengths[-1],
        'share_under_50': round(100 * sum(1 for x in lengths if x < 50) / len(lengths), 1),
        'share_over_150': round(100 * sum(1 for x in lengths if x > 150) / len(lengths), 1),
        'share_over_250': round(100 * sum(1 for x in lengths if x > 250) / len(lengths), 1),
    }


# Six v1 records ask for detail in the user turn and then answer in one or two
# sentences (53-92 chars). They teach the exact mapping this dataset exists to
# remove: "asked properly -> still answer briefly". Rewritten here instead of
# editing experimental-v1, so the change is auditable against v1's recorded hashes.
OVERRIDES = {
    'curated-048': [
        '因为它让你看到的是一个可能已经过期的世界，而你没意识到这一点。\n'
        '具体是三种错法。第一种，你改完代码或者改完数据，行为没变，于是怀疑改动没生效，又去改一遍，越改越乱——其实是缓存还拿着旧结果。第二种，在 A 环境复现不出来、在 B 环境能复现，差别可能只是 A 的缓存热、B 的冷。第三种最坑：缓存只在部分节点失效，同一个请求打两次结果不同，你会以为问题偶发，转头去查并发，方向整个跑偏。\n'
        '所以我排这类问题，第一步永远是绕过缓存直接看源数据，确认源头长什么样。第二步确认缓存在哪一层——浏览器、CDN、还是应用自己的 Redis，层不同清法完全不同。第三步才比对绕过和不绕过的结果，差异就是缓存造成的。',
        '行，按顺序来。\n'
        '第一步，确认缓存这层存不存在。直接请求源站接口，带上绕过参数，或者临时换个 key 前缀，看结果变不变。这一步只做一件事：把有缓存和没缓存分开。\n'
        '第二步，如果绕过之后正常了，说明缓存是变量，接着定位是哪一层。浏览器看强刷有没有用；CDN 看响应头里的命中标记；应用缓存就去 Redis 里按 key 直接读一次，看值和数据库对不对得上。\n'
        '第三步，对不上时别急着清。先看 key 是怎么拼的，有没有漏掉影响结果的参数——少拼一个字段，不同条件的请求就会共用一份缓存，这种清多少次都没用，得改 key。\n'
        '第四步，确认是过期策略问题就调 TTL，是写入没更新就补失效逻辑，改完一定要验证：改一次数据、读一次，值必须立刻变。四步走完还没定位到，那问题大概率不在缓存，回去看代码。',
    ],
    'curated-049': [
        '两者都让文件出现在多个地方，但目的相反：同步是为了哪儿都能用，备份是为了还能回去。\n'
        '同步盘的目标是保持一致。你在 A 电脑删掉一个文件，B 电脑上也没了，这是特性不是 bug。它也保留不了真正的历史——有些同步盘有版本记录，但通常只留一个月，而且不是为恢复设计的。所以同步盘解决的是多设备协作，不是灾难恢复。\n'
        '备份的目标是能回到过去某个时点。它必须和源数据隔离开，并且不被日常操作影响。放在同一个同步盘里的备份根本不算备份，因为误删会同步过去，加密也会同步过去。\n'
        '还有一条常被忽略：备份只有验证过恢复才算存在。没演练过的备份，你并不知道它能不能用、缺不缺文件、恢复要多久。',
        '需要一份和同步盘隔离的副本。\n'
        '做法很简单：拿一块移动硬盘，或者一个独立的对象存储桶，定期把重要目录整体拷一份进去，拷完断开。关键就在断开——只有在线的、双向同步的副本，挡不住误删和加密。\n'
        '频率看数据变化速度。文档、照片一周一次足够；代码本来就在版本库里，反而是最安全的。数据库要另算，得做一致性导出，不能直接拷文件。\n'
        '再加一件事：每季度挑一个文件真恢复一次，看着它成功打开。这步花十分钟，但能证明你那堆备份不是空的。',
    ],
    'curated-051': [
        '因为能放进去和能用得快是两回事，训练恰恰卡在后者。\n'
        '显存和内存的位置不同，带宽差一个数量级。显存焊在显卡板上，带宽动辄几百 GB 每秒；系统内存要走 PCIe，一条通道十几 GB 每秒。参数放内存里装得下，但每次前向都要把它们搬过总线，算力全耗在搬数据上，GPU 大部分时间在等。\n'
        '第二层是访问方式。很多算子是假设数据在本地显存的前提下写的，直接指着系统内存跑，会走统一寻址的回退路径，性能断崖下降，有些算子干脆不支持。\n'
        '第三层是容量并没有真的变多。共享显存是按需借用系统内存，不是一个能提前占住的池子。训练中途系统内存紧张，它就会被回收，表现是突然变慢或者直接崩掉，而不是干脆地报 OOM。\n'
        '所以对的做法是卸载而不是借用：把部分层明确放在 CPU 上算，只把当前需要的张量搬到 GPU，这样搬运有计划、可与计算重叠，而不是每次算子执行时被动触发。',
        '它主要服务于图形场景，不是为了训练。\n'
        '显示器和应用要往显存里放纹理、帧缓冲，显存不够时系统允许借一部分内存顶上，避免直接黑屏或者应用崩溃。所以那个数字是应急上限，不是可用的资源池。\n'
        '对训练来说，它的意义基本只有负面的：你以为还有余量，实际上那点余量不可靠、不可预测、带宽也差，真用上了就是性能不明不白地掉，或者进程被系统杀掉。\n'
        '所以我调训练时把它当成不存在。要么用显存上限参数把分配卡死在实际显存以内，要么显式做 CPU 卸载，让每层归属是确定的。这样至少出问题时是明确的 OOM，而不是幽灵般的变慢。',
    ],
    'curated-052': [
        '假设显卡一次只能放 1 条样本，但你想要的效果是 4 条样本一起更新一次参数。梯度累积就是干这个的。\n'
        '具体走一遍：第 1 条样本前向，算出 loss，反向得到梯度，但先不更新参数，把梯度存着。第 2、3、4 条同样处理，每次的梯度累加到同一份缓冲里。四步走完，拿累积的梯度除以 4，因为 loss 按平均算，再调用一次优化器更新参数，然后清零，进入下一轮。\n'
        '对比一下：如果显卡放得下 4 条，一次前向就出 4 条的 loss，除以 4 反向，梯度是这 4 条的均值。两条路径在数学上一样，参数更新也一样。区别只在批内统计类层，累积不等价，因为统计量是每微批单独算的。\n'
        '代价是时间。4 条样本要跑 4 次前向和反向，串行执行，显存省下来了，但耗时大约是四倍，不是免费的。',
        '对绝大多数层是一样，但有两处例外。\n'
        '第一处是带批内统计的层，比如批归一化。一次放 4 条时，均值和方差是这 4 条一起算的；累积时每微批单独算，得到 4 组统计量。这对结果有实质影响，所以做累积时通常建议把这类统计方式改掉，或者直接冻结。\n'
        '第二处是随机性。dropout 的掩码按每次前向采样，一次放 4 条和四次各放 1 条，采到的掩码不一样，所以结果不会逐位相同，只是期望一致。\n'
        '除了这两点，梯度的均值和优化器的更新步长都是等价的。所以判断标准是：模型没有批相关的层，累积就是安全的；有的话，要么改结构，要么接受它只是近似更大批。',
    ],
    'curated-053': [
        '三个集回答三个不同的问题。\n'
        '训练集回答模型能不能学会，参数更新只看它。所以它必须够大、覆盖够全，也得干净——训练集里有错误标注，模型会认真地把错误学进去。\n'
        '验证集回答学到的这版好不好、该用哪一版。它不参与梯度更新，只用来选：选超参数、选训练轮数、选第几个检查点。所以你会反复看它，看很多次。\n'
        '测试集回答最终这版在没见过的数据上怎么样。它只在方案定下来之后用一次。用多了就不叫测试集了，因为你已经根据它的结果做了选择，它变成了第二个验证集，报出来的分数会偏乐观。\n'
        '关键在隔离。同一段对话换几个名字、改几个数字，仍然算同一个样本。如果它的变体分散在训练集和验证集，验证分数就会虚高，你以为模型泛化了，其实它记住了。',
        '不算，这几乎是数据泄漏最常见的形式。\n'
        '原因很直接：模型学的是这段对话该用什么语气、按什么结构回答，名字和数字只是可替换的槽位。把小陆换成小周、青柚换成微光，学到的模式没有任何变化。所以它在验证集上碰到换过名字的同一段，等于开卷。\n'
        '正确的做法是按语义家族切分：同一段对话的所有变体必须待在同一个集合里。名字、数字、项目名可以变，只要骨架一样就是一个家族。切分按家族切，不按记录切。\n'
        '判断标准可以这么用：如果你一眼能看出两条记录讲的是同一件事，它们就是同一个家族。这个判断不需要工具，人工过一遍就能筛掉大部分泄漏。',
    ],
    'curated-055': [
        '不能，这只是必要条件里最弱的一条。\n'
        '参数变了只能说明反向传播跑通了、优化器动过。下面这些情况都会让参数变化而训练其实失败：学习率太大，参数跳进更差的区域；数据有 bug，模型在认真地学错的东西；标签全错，loss 反而能降，因为它在拟合噪声；梯度爆炸被裁剪掩盖，参数在慢慢发散。\n'
        '判断训练是否有效，我至少看四件事。一是 loss 曲线，训练和验证要一起看，只看训练 loss 下降说明不了泛化。二是梯度范数，是否有限、有没有长期贴着裁剪阈值。三是验证集表现，这是唯一能反映没见过的数据上怎么样的指标。四是真实生成，输出是不是像人话、有没有崩坏成重复。\n'
        '最容易忽略的是第四条。我见过 loss 降得漂亮、验证也在降，但生成出来是一串重复词的模型。数值指标不能替代实际效果。',
        '留四样，能覆盖绝大部分争议。\n'
        '一是验证集 loss 的原始记录，按步数存，别只留最后一行。这样别人能看出是稳定下降还是震荡。\n'
        '二是检查点本身，包含优化器状态和随机数状态。没有随机数状态的检查点，续训会走出一条新轨迹，上次跑到哪就说不清了。\n'
        '三是固定输入的生成样本，同一段提示词在基座和微调后各跑一次，原文保留。这比任何指标都直观。\n'
        '四是配置和数据清单，含随机种子、批大小、累积步数、学习率、数据文件哈希。缺了这些结果不可复现，等于没做。',
    ],
}


def apply_overrides(rows):
    applied = []
    for row in rows:
        replacement = OVERRIDES.get(row['id'])
        if not replacement:
            continue
        assistant_turns = [m for m in row['messages'] if m['role'] == 'assistant']
        if len(assistant_turns) != len(replacement):
            raise RuntimeError(f'{row["id"]} has {len(assistant_turns)} assistant turns, override has {len(replacement)}')
        for message, text in zip(assistant_turns, replacement):
            message['content'] = text
        applied.append(row['id'])
    missing = set(OVERRIDES) - set(applied)
    if missing:
        raise RuntimeError(f'Overrides target records not present in train: {sorted(missing)}')
    return applied


def main():
    source = LOCAL / 'data/experimental-v1'
    output = LOCAL / 'data/experimental-v2'
    output.mkdir(parents=True, exist_ok=True)

    v1 = {split: read_rows(source / f'{split}.jsonl') for split in ('train', 'validation', 'test')}
    # Snapshot v1's own distribution before any override mutates it, otherwise the
    # "before" column in the manifest would describe the edited data.
    v1_train_stats = describe(v1['train'])
    overridden = apply_overrides(v1['train'])

    added = []
    for category, family, turns in length_scenarios.CASES:
        row = build_data.dialogue(category, family, turns)
        row['source'] = 'original_synthetic_v2_length'
        added.append(row)
    for row in length_scenarios.tool_cases(build_data.dialogue, build_data.TOOLS):
        row['source'] = 'original_synthetic_v2_tool'
        added.append(row)

    known_ids = {r['id'] for split in v1.values() for r in split}
    known_families = {r['family'] for split in v1.values() for r in split}
    duplicate_ids = [r['id'] for r in added if r['id'] in known_ids]
    duplicate_families = [r['family'] for r in added if r['family'] in known_families]
    if duplicate_ids or duplicate_families:
        raise RuntimeError(f'id/family collide with v1: {duplicate_ids} {duplicate_families}')
    if len({r['id'] for r in added}) != len(added):
        raise RuntimeError('Duplicate id inside the v2 addition')

    train = v1['train'] + added
    splits = {'train': train, 'validation': v1['validation'], 'test': v1['test']}

    hashes = {}
    for split, rows in splits.items():
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (output / f'{split}.jsonl').write_text(text, encoding='utf-8')
        hashes[split] = hashlib.sha256(text.encode()).hexdigest()

    v1_hashes = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))['sha256']
    held_out_identical = all(hashes[s] == v1_hashes[s] for s in ('validation', 'test'))
    if not held_out_identical:
        raise RuntimeError('validation/test are not byte-identical to v1; refusing to publish')

    manifest = {
        'name': 'experimental-v2',
        'seed': 3407,
        'total': sum(len(r) for r in splits.values()),
        'counts': {s: {'records': len(r), 'categories': dict(Counter(x['category'] for x in r)),
                       'families': len({x['family'] for x in r})} for s, r in splits.items()},
        'sha256': hashes,
        'inherited_from': 'experimental-v1',
        'held_out_identical_to_v1': held_out_identical,
        'overridden_v1_records': overridden,
        'override_reason': 'v1 detail records asked for detail but answered in 53-92 chars; rewritten to remove the contradicting signal',
        'added_records': len(added),
        'added_categories': dict(Counter(r['category'] for r in added)),
        'length_compare': {'v1_train': v1_train_stats, 'v2_train': describe(train),
                           'v2_additions_only': describe(added)},
        'sources': ['training/.training-local/data/experimental-v1 train split',
                    'training/length_scenarios.py (original synthetic long-form, authored for this round)'],
        'private_chat_accessed': False,
        'real_chat_validated': False,
        'human_reviewed': False,
        'limitations': [
            'validation/test unchanged on purpose, so new long-form behaviour is not covered by held-out loss',
            'Length target is conditional, not distributional: median reply length stays low because short banter is correct',
            'All additions are synthetic and authored by the same generator; no human quality review',
            'Direct duplicates of validation/test/eval topics were avoided by hand, not by an automatic check',
            'Residual adjacency: example.invalid is still reused as a placeholder domain, and "cannot read the screen" is adjacent to the evaluation fixture about an unseen photo',
        ],
    }
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sources'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
