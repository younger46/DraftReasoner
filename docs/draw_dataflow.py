# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont

W, H = 1210, 1120
IMG = Image.new("RGB", (W, H), "#ffffff")
D = ImageDraw.Draw(IMG)
FY = "C:/Windows/Fonts/msyh.ttc"; FYB = "C:/Windows/Fonts/msyhbd.ttc"
def f(sz, b=False): return ImageFont.truetype(FYB if b else FY, sz)
F_T = f(27, True); F_H = f(19, True); F_N = f(16, True); F_R = f(11); F_S = f(11); F_XS = f(10)

def wrap(t, font, mw):
    lines, cur = [], ""
    for seg in t.split("\n"):
        cur = ""
        for ch in seg:
            if D.textlength(cur + ch, font=font) <= mw or cur == "":
                cur += ch
            else:
                lines.append(cur); cur = ch
        if cur: lines.append(cur)
    return lines

def block(cx, y0, t, font, mw, color, lh=None, anchor="center"):
    lines = wrap(t, font, mw); lh = (font.size + (lh if lh is not None else 5))
    ys = y0 if anchor == "top" else y0 - len(lines) * lh / 2
    for i, ln in enumerate(lines):
        tw = D.textlength(ln, font=font)
        if anchor == "center": D.text((cx - tw / 2, ys + i * lh), ln, font=font, fill=color)
        else: D.text((cx, ys + i * lh), ln, font=font, fill=color)
    return len(lines) * lh

def rrect(x0,y0,x1,y1,r,fill,outline,ow=2):
    D.rounded_rectangle([x0,y0,x1,y1], radius=r, fill=fill, outline=outline, width=ow)

BLUE="#2e78c2"; BLUE_F="#e4eefb"; GREEN="#3a9e5c"; GREEN_F="#e6f6ec"
ORANGE="#dd8a2b"; ORANGE_F="#fdf0df"; PUR="#7a5bbf"; PUR_F="#efe9fa"
NAVY="#0b3b5c"; GREY="#3a4653"; GREY2="#8a99a8"

D.text((W/2, 25), "MechAgent ReAct 数据流向（用户输入图纸图片 + 问题）", font=F_T, fill="#1f2933", anchor="mm")
rrect(28, 62, 430, 94, 10, NAVY, NAVY); D.text((229, 78), "处理流程（ReAct 循环）", font=F_H, fill="#fff", anchor="mm")
rrect(450, 62, 1110, 94, 10, "#5b6b7a", "#5b6b7a"); D.text((780, 78), "流动的数据 / 产物", font=F_H, fill="#fff", anchor="mm")

steps = [
    ("1  用户输入", "上传 图纸图片 + 问题", {"main": BLUE, "fill": BLUE_F}, ["image_path", "question"]),
    ("2  MechAgent(入口)", "读 subcategory → plan_for", {"main": PUR, "fill": PUR_F}, ["metadata.subcategory", "plan=[engineer]"]),
    ("3  agent 节点(LLM)", "结合 question+图+观测 → 决定动作", {"main": GREEN, "fill": GREEN_F}, ["AIMessage(thought)", "tool_call 或 content=<answer>"]),
    ("4  ToolNode", "执行所选工具 → 观测回传", {"main": ORANGE, "fill": ORANGE_F}, ["ToolMessage(observation)", "json{ok,data,evidence,error}"]),
    ("5  循环决策(条件边)", "有 tool_calls 且 < MAX_TOOL_ROUNDS?", {"main": ORANGE, "fill": ORANGE_F}, ["统计已用轮数", "→ 回 agent(步骤3) 或 → END"]),
    ("6  输出 & 评测", "extract_answer → LLM judge", {"main": PUR, "fill": PUR_F}, ["<think>/<answer>", "score 0/1 → 聚合统计"]),
]

LX0, LX1 = 28, 430
RX0, RX1 = 450, 1110
y = 104
SBH = 118
loop_from = None
loop_to = None
for i, (nm, sub, pal, datas) in enumerate(steps):
    rrect(LX0, y, LX1, y + SBH, 14, pal["fill"], pal["main"], 2)
    block((LX0+LX1)/2, y + 24, nm, F_N, (LX1-LX0)-36, pal["main"], lh=6)
    block((LX0+LX1)/2, y + 48, sub, F_R, (LX1-LX0)-46, GREY, lh=4, anchor="top")
    rrect(RX0, y, RX1, y + SBH, 14, "#fbfcfe", pal["main"], 2)
    xx = RX0 + 18
    chip_w = (RX1-RX0)-36
    for j, d in enumerate(datas):
        rrect(xx, y + 16 + j * 30, xx + chip_w, y + 16 + j * 30 + 22, 8, "#ffffff", pal["main"], 1)
        txt = (d[:46] + "…") if D.textlength(d, font=F_XS) > chip_w - 22 else d
        D.text((xx + 12, y + 16 + j * 30 + 6), txt, font=F_XS, fill=GREY)
    rrect(LX0+12, y + SBH - 30, LX0+52, y + SBH - 8, 8, pal["main"], pal["main"])
    D.text((LX0+32, y + SBH - 19), str(i+1), font=F_S, fill="#fff", anchor="mm")
    if i == 2: loop_to = y + SBH/2
    if i == 4: loop_from = y + SBH/2
    ny = y + SBH + 2
    if i < len(steps)-1:
        ax = (LX0+LX1)/2
        D.line([(ax, y + SBH), (ax, ny + 6)], fill=GREY2, width=2)
        D.polygon([(ax, ny+8), (ax-6, ny+2), (ax+6, ny+2)], fill=GREY2)
    y = ny + 12

# loop rail on the right: from step5 back up to step3
railx = 1145
D.line([(RX1, loop_from), (railx, loop_from), (railx, loop_to), (RX1, loop_to)], fill=ORANGE, width=2)
D.polygon([(RX1, loop_to), (RX1-14, loop_to-6), (RX1-14, loop_to+6)], fill=ORANGE)
D.text((railx, (loop_from+loop_to)/2), "回环\n≤MAX_TOOL_ROUNDS", font=F_XS, fill=ORANGE, anchor="mm")

fy = y - 4
rrect(LX0, fy, RX1, fy + 128, 12, "#f5f7fa", "#c4ccd4", 2)
D.text((LX0+16, fy + 12), "说明", font=F_S, fill=NAVY)
D.text((LX0+16, fy + 38), "· ReAct：LLM 自行判断“要不要调工具、调哪个”；ToolNode 执行并把 observation 回灌给 LLM，直到它给出最终 <answer>。", font=F_XS, fill=GREY)
D.text((LX0+16, fy + 62), "· 工具分两类：VLM 感知(AnnotationExtract/ViewAlign) 与 确定性(FigureParse/GeometrySolve-OCR/StdKB)。", font=F_XS, fill=GREY)
D.text((LX0+16, fy + 86), "· 注意：ReAct 的 LLM 可能覆盖工具的精确值(如 GC 把 Φ61 读成 Φ67)；建议“确定性锚点”强制采纳工具的高置信度数值。", font=F_XS, fill=GREY)

IMG.save("docs/dataflow_architecture.png")
print("saved", IMG.size)
