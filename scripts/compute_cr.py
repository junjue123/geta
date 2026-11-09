"""按论文 BOPs 口径计算压缩率 CR, 并给出到 ep100 的投影。

论文 (DACO.tex L389-395):  CR = BOPs_orig / BOPs_comp
逐层展开等价于
    CR = 32*32 / ( (1 - s) * w_bit * a_bit )
其中 s 为结构化剪枝带来的运算/参数缩减比例, w_bit / a_bit 为权重/激活平均位宽。

本脚本从 history.jsonl 读取 w_bit 与 s:
  - 若已有 avg_act_bit 字段(打过补丁的运行), 用真实激活位宽
  - 否则退回 a_bit = w_bit, 并在输出里标记 est(估计), 只作量级参考
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "outputs", "framework_compare")
# 现在的 train.py 把两个 mode 写在同一个 out 目录下(平铺), 不再分 ours/origin 子目录
RUNS = [("daco", os.path.join(ROOT, "daco_history.jsonl")),
        ("origin", os.path.join(ROOT, "baseline_history.jsonl"))]


def cr(s, w, a):
    if not (w > 0 and a > 0):
        return float("nan")
    return (32.0 * 32.0) / max(1e-9, (1.0 - s) * w * a)


def main():
    show = [int(x) for x in sys.argv[1:]] or None
    for tag, path in RUNS:
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        has_act = "avg_act_bit" in recs[0]
        print(f"\n===== {tag}  ({len(recs)} epochs, 激活位宽: "
              f"{'实测' if has_act else '缺失→用 w 估计'}) =====")
        print(f"{'ep':>4} {'spars':>7} {'w_bit':>6} {'a_bit':>6} "
              f"{'CR_bops':>9} {'旧口径':>8} {'val_acc':>8}")
        eps = show or [r["epoch"] for r in recs]
        last = recs[-1]
        for r in recs:
            if r["epoch"] not in eps:
                continue
            w = r["avg_weight_bit"]
            a = r.get("avg_act_bit") or w
            c = cr(r["group_sparsity"], w, a)
            tag_a = f"{a:.2f}" + ("" if has_act else "*")
            print(f"{r['epoch']:>4} {r['group_sparsity']:>7.4f} {w:>6.2f} {tag_a:>6} "
                  f"{c:>8.2f}x {r['overall_compress']:>7.2f}x {r['val_acc']:>8.4f}")

        # ---- 投影到 ep100: 稀疏度按最近 6 个 epoch 的平均速率线性外推 ----
        tail = recs[-6:]
        rate = (tail[-1]["group_sparsity"] - tail[0]["group_sparsity"]) / max(
            tail[-1]["epoch"] - tail[0]["epoch"], 1)
        s_end = min(0.5, last["group_sparsity"] + rate * (100 - last["epoch"]))
        w = last["avg_weight_bit"]
        a = last.get("avg_act_bit") or w
        print(f"  当前速率 {rate:+.5f} spars/ep → ep100 稀疏度≈{s_end:.3f} "
              f"(target 0.5), CR_bops(位宽不变)≈{cr(s_end, w, a):.2f}x")
        print(f"  参考: 论文 ResNet-20 DACO 行 w=4.05 a=5.13 CR=80.40x → "
              f"反推 (1-s)=1024/(80.40*4.05*5.13)="
              f"{1024.0/(80.40*4.05*5.13):.3f} 即 s≈{1-1024.0/(80.40*4.05*5.13):.3f}")


if __name__ == "__main__":
    main()
