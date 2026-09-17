# -*- coding: utf-8 -*-
"""Mark the retracted parts of section 15.10 in the V11 report.

Section 15.11 showed that the per-channel split in 15.10.3 was a mid-training
artifact (it reverses by ep49), which removes the mechanism behind 15.10.4's
"D1 is top priority". A reader who lands on 15.10 must see that immediately,
so we insert pointers rather than rewriting history.
"""
import io
import sys

P = "outputs/V11_角色条件化_诊断与方案.md"

HEAD = "## §15.10 早期读数（09-17，B2 中途）—— 一个方法论警示 + 一个可操作发现\n"
H3 = "### 15.10.3 形状诊断：B2 换来的是**相关度**，不是 delta 精度\n"
H4 = "### 15.10.4 直接推论：**D1（B2 + C1a）应升为最高优先级**\n"

BANNER = HEAD + """
> ⚠️ **本节 15.10.3 / 15.10.4 的结论已被 §15.11 撤回（09-17 16:00）**：ep38 的
> 「加性项损害常数通道」是**训练中途现象**，到 ep49 已反号（常数通道从落后 5.9% 变为**领先 8.1%**）。
> 由此推出的「D1 升为最高优先级」随之失去机制依据，D1 降级为普通组合臂。
> §15.10.1 / §15.10.2 的方法论警示**不但成立，还被 §15.11.2 加强**
> （训练 val 不只水平不可比，**排序与下降斜率也不可靠**）。
> 本节原文保留，以留下发现过程与当时的推理痕迹。
"""

M3 = H3 + (
    "> ⚠️ **本小节的通道分解已被 §15.11.4 撤回**（那是 ep38 的中途状态）。\n"
)
M4 = H4 + (
    "> ⚠️ **本小节的推论已被 §15.11.4 撤回**：D1 不再具有「互补」的机制依据。\n"
)


def main() -> int:
    s = io.open(P, encoding="utf-8").read()
    for old, new, tag in ((HEAD, BANNER, "15.10"), (H3, M3, "15.10.3"), (H4, M4, "15.10.4")):
        n = s.count(old)
        if n != 1:
            print("ABORT: anchor %s occurs %d times (expected 1)" % (tag, n))
            return 1
        s = s.replace(old, new)
        print("patched %s" % tag)
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("written %s" % P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
