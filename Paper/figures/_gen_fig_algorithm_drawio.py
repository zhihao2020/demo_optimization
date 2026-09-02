#!/usr/bin/env python
"""Emit Paper/figures/fig_algorithm.drawio on a 12x8 grid. Unique actor_online id."""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "fig_algorithm.drawio"

# Canvas: 12 col x 8 row, ~1.85:1
CW, CH = 100, 90
OX, OY = 40, 20
PAGE_W, PAGE_H = 1280, 840


def G(c: float, r: float, w: float, h: float) -> tuple[int, int, int, int]:
    return (
        round(OX + c * CW),
        round(OY + r * CH),
        round(w * CW),
        round(h * CH),
    )


def geom(c, r, w, h) -> str:
    x, y, ww, hh = G(c, r, w, h)
    return f'<mxGeometry x="{x}" y="{y}" width="{ww}" height="{hh}" as="geometry"/>'


def geom_xy(x, y, w, h) -> str:
    return f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'


# ----- boxes (col, row, w, h) : keep Agent right edge < FMU left -----
BOX = {
    "agent": (0.12, 0.08, 9.40, 8.55),
    "title": (0.28, 0.14, 3.20, 0.26),
    "state": (0.28, 0.54, 2.08, 1.50),
    "oracle": (0.28, 2.24, 2.08, 1.58),
    "actor": (2.56, 0.54, 3.48, 2.72),
    "decoder": (6.24, 0.54, 2.72, 1.56),
    "givesafe": (6.52, 2.30, 2.16, 1.20),
    "retry": (6.24, 3.70, 2.72, 0.66),
    "replay": (6.24, 4.54, 2.72, 1.12),
    "fmu": (9.80, 1.52, 2.32, 1.78),
    "batch": (4.28, 4.62, 1.80, 0.86),
    "q1": (0.36, 6.22, 1.80, 1.00),
    "q2": (2.28, 6.22, 1.80, 1.00),
    "loss": (0.36, 7.38, 3.72, 0.60),
    "tgt_actor": (4.64, 6.22, 2.24, 0.92),
    "tgt_critics": (7.04, 6.22, 2.16, 0.92),
    "td_y": (4.64, 7.32, 2.24, 0.66),
    "min_q": (7.04, 7.32, 2.16, 0.66),
    "jphi": (3.70, 3.38, 2.30, 0.58),
    "critics_lbl": (0.36, 5.72, 2.40, 0.26),
    "caption": (0.16, 8.78, 11.4, 0.32),
}

STYLE = {
    "agent": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFBFC;strokeColor=#B8BFC7;strokeWidth=1.2;arcSize=5;pointerEvents=0;",
    "title": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontFamily=Times New Roman;fontSize=14;fontStyle=1;fontColor=#1A3A5C;",
    "state": "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F1F8;strokeColor=#0072B2;strokeWidth=1.2;fontFamily=Times New Roman;fontSize=11;fontColor=#1A3A5C;align=left;spacingLeft=10;arcSize=10;",
    "oracle": "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F6F0;strokeColor=#009E73;strokeWidth=1.4;fontFamily=Times New Roman;fontSize=11;fontColor=#1A3A5C;align=center;arcSize=10;",
    "actor": "rounded=1;whiteSpace=wrap;html=1;fillColor=#D7E8F5;strokeColor=#1A3A5C;strokeWidth=1.8;fontFamily=Times New Roman;fontColor=#1A3A5C;verticalAlign=top;spacingTop=6;arcSize=10;",
    "decoder": "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F1F8;strokeColor=#0072B2;strokeWidth=1.3;fontFamily=Times New Roman;fontSize=11;fontColor=#1A3A5C;arcSize=10;",
    "givesafe": "rhombus;whiteSpace=wrap;html=1;fillColor=#FDECEC;strokeColor=#C2410C;strokeWidth=1.5;fontFamily=Times New Roman;fontSize=11;fontColor=#9A3412;",
    "retry": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF7F5;strokeColor=#B91C1C;dashed=1;dashPattern=6 4;fontFamily=Times New Roman;fontSize=10;fontColor=#B91C1C;arcSize=10;",
    "replay": "rounded=1;whiteSpace=wrap;html=1;fillColor=#F3EEF8;strokeColor=#6B4C9A;strokeWidth=1.4;fontFamily=Times New Roman;fontSize=11;fontColor=#1A3A5C;arcSize=10;",
    "fmu": "rounded=1;whiteSpace=wrap;html=1;fillColor=#DFF3EA;strokeColor=#009E73;strokeWidth=1.6;fontFamily=Times New Roman;fontSize=12;fontColor=#1A3A5C;arcSize=10;",
    "batch": "rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F4FB;strokeColor=#8B7AB8;fontFamily=Times New Roman;fontSize=11;fontColor=#4C3A6B;arcSize=12;",
    "q": "rounded=1;whiteSpace=wrap;html=1;fillColor=#EDE6F5;strokeColor=#6B4C9A;strokeWidth=1.3;fontFamily=Times New Roman;fontSize=11;fontColor=#1A3A5C;arcSize=10;",
    "loss": "rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F4FB;strokeColor=#6B4C9A;fontFamily=Times New Roman;fontSize=10;fontColor=#4C3A6B;arcSize=10;",
    "tgt": "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8F5FC;strokeColor=#B8A7D4;strokeWidth=1;dashed=1;dashPattern=4 3;fontFamily=Times New Roman;fontSize=11;fontColor=#6B4C9A;opacity=85;arcSize=10;",
    "chip": "rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F4FB;strokeColor=#8B7AB8;fontFamily=Times New Roman;fontSize=10;fontColor=#4C3A6B;arcSize=10;",
    "jphi": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontFamily=Times New Roman;fontSize=10;fontColor=#6B4C9A;",
    "lbl": "text;html=1;strokeColor=none;fillColor=none;align=left;fontFamily=Times New Roman;fontSize=11;fontStyle=1;fontColor=#6B4C9A;",
    "caption": "text;html=1;strokeColor=none;fillColor=none;align=left;fontFamily=Times New Roman;fontSize=12;fontColor=#1A3A5C;",
    "head": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#0072B2;fontFamily=Times New Roman;fontSize=10;fontColor=#1A3A5C;",
    "node": "ellipse;fillColor=#FFFFFF;strokeColor=#0072B2;strokeWidth=1.1;",
    "nedge": "endArrow=none;strokeColor=#7AA3C4;strokeWidth=0.7;",
}

E_BLACK = "endArrow=block;html=1;rounded=0;strokeColor=#374151;strokeWidth=1.2;fontFamily=Times New Roman;fontSize=10;fontColor=#374151;"
E_GREEN = "endArrow=block;html=1;rounded=0;strokeColor=#009E73;strokeWidth=1.4;fontFamily=Times New Roman;fontSize=10;fontColor=#009E73;"
E_RED = "endArrow=block;dashed=1;html=1;rounded=0;strokeColor=#B91C1C;strokeWidth=1.1;dashPattern=5 5;fontFamily=Times New Roman;fontSize=10;fontColor=#B91C1C;"
E_PURP = "endArrow=block;html=1;rounded=0;strokeColor=#6B4C9A;strokeWidth=1.2;fontFamily=Times New Roman;fontSize=10;fontColor=#6B4C9A;"
E_POLY = "endArrow=block;dashed=1;html=1;rounded=0;strokeColor=#8B7AB8;strokeWidth=1.05;dashPattern=4 3;fontFamily=Times New Roman;fontSize=9;fontColor=#6B4C9A;"


def cell(cid, value, style, g, *, parent="1", vertex=True, extra=""):
    v = ' vertex="1"' if vertex else ""
    val = f' value="{value}"' if value is not None else ""
    return f'        <mxCell id="{cid}" parent="{parent}"{v}{val} style="{style}"{extra}>\n          {g}\n        </mxCell>'


def edge(eid, source, target, style, value="", points=None, extra=""):
    val = f' value="{value}"' if value else ""
    pts = ""
    if points:
        arr = "\n".join(f'              <mxPoint x="{x}" y="{y}"/>' for x, y in points)
        pts = f'\n            <Array as="points">\n{arr}\n            </Array>'
    return (
        f'        <mxCell id="{eid}" parent="1" edge="1" source="{source}" target="{target}"{val} style="{style}"{extra}>\n'
        f'          <mxGeometry relative="1" as="geometry">{pts}\n          </mxGeometry>\n'
        f"        </mxCell>"
    )


def boxes_overlap(a, b, gap=4):
    ax, ay, aw, ah = G(*BOX[a])
    bx, by, bw, bh = G(*BOX[b])
    return not (
        ax + aw + gap <= bx
        or bx + bw + gap <= ax
        or ay + ah + gap <= by
        or by + bh + gap <= ay
    )


def main() -> None:
    skip_pairs = {
        frozenset({"agent", k}) for k in BOX if k != "agent" and k not in {"fmu", "caption"}
    }
    skip_pairs |= {
        frozenset({"title", "agent"}),
        frozenset({"critics_lbl", "agent"}),
        frozenset({"jphi", "agent"}),
        frozenset({"caption", "agent"}),
        frozenset({"critics_lbl", "q1"}),
        frozenset({"critics_lbl", "q2"}),
    }
    # Title sits in the Agent header strip; ignore it vs. boxes below.
    keys = [k for k in BOX if k not in {"agent", "title"}]
    overlaps = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if frozenset({a, b}) in skip_pairs:
                continue
            if boxes_overlap(a, b, gap=6):
                overlaps.append((a, b, G(*BOX[a]), G(*BOX[b])))
    if overlaps:
        msg = "\n".join(f"  {a} vs {b}: {ga} {gb}" for a, b, ga, gb in overlaps)
        raise SystemExit("overlap:\n" + msg)

    ax, ay, aw, ah = G(*BOX["actor"])
    # heads on the right, stacked
    heads = [
        ("head_th", "u&lt;sub&gt;t&lt;/sub&gt;&lt;sup&gt;th&lt;/sup&gt;", 256, 56),
        ("head_bat", "u&lt;sub&gt;t&lt;/sub&gt;&lt;sup&gt;bat&lt;/sup&gt;", 256, 94),
        ("head_ell", "&amp;ell;&lt;sub&gt;t&lt;/sub&gt;", 256, 132),
        ("head_z", "z&lt;sub&gt;t&lt;/sub&gt;", 256, 170),
    ]
    nodes = [
        ("n11", 40, 64),
        ("n12", 40, 96),
        ("n13", 40, 128),
        ("n21", 108, 56),
        ("n22", 108, 88),
        ("n23", 108, 120),
        ("n24", 108, 152),
        ("n31", 176, 64),
        ("n32", 176, 96),
        ("n33", 176, 128),
    ]

    # waypoints in pixels
    # gradient corridor: below oracle / left of batch
    # polyak corridor: 20px gap between actor-right and decoder-left
    actor_r = ax + aw  # ~628
    dec_x = G(*BOX["decoder"])[0]  # ~664
    mid_gap_x = (actor_r + dec_x) // 2
    q1x, q1y, q1w, _ = G(*BOX["q1"])
    jx, jy, jw, jh = G(*BOX["jphi"])
    fmx, fmy, fmw, fmh = G(*BOX["fmu"])
    stx, sty, stw, _ = G(*BOX["state"])
    rpx, rpy, rpw, rph = G(*BOX["replay"])
    lsx, lsy, lsw, lsh = G(*BOX["loss"])
    tcx, tcy, tcw, tch = G(*BOX["tgt_critics"])

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<mxfile host="app.diagrams.net" agent="PC-HybridTD3 Fig.2 grid" version="22.1.0">',
        '  <diagram id="pc-hybrid-td3-fig2" name="PC-HybridTD3">',
        f'    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{PAGE_W}" pageHeight="{PAGE_H}" math="0" shadow="0" background="#FFFFFF">',
        "      <root>",
        '        <mxCell id="0"/>',
        '        <mxCell id="1" parent="0"/>',
        "",
        cell("agent_box", "", STYLE["agent"], geom(*BOX["agent"])),
        cell("agent_title", "PC-HybridTD3 Agent", STYLE["title"], geom(*BOX["title"])),
        "",
        cell(
            "state",
            "&lt;b&gt;State s&lt;sub&gt;t&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;SOCs&lt;br&gt;unit powers&lt;br&gt;load / RES&lt;br&gt;TOU price&lt;br&gt;24 h forecast",
            STYLE["state"],
            geom(*BOX["state"]),
        ),
        cell(
            "oracle",
            "&lt;b&gt;Feasibility Oracle&lt;/b&gt;&lt;br&gt;K(s&lt;sub&gt;t&lt;/sub&gt;), M&lt;sub&gt;k&lt;/sub&gt;(s&lt;sub&gt;t&lt;/sub&gt;)&lt;br&gt;joint feasible support&lt;br&gt;Â&lt;sub&gt;f&lt;/sub&gt;(s) = A&lt;sub&gt;dev&lt;/sub&gt;(s) ∩ A&lt;sub&gt;grid&lt;/sub&gt;(s)",
            STYLE["oracle"],
            geom(*BOX["oracle"]),
        ),
        cell(
            "actor_online",
            "&lt;div style=&quot;font-weight:bold;font-size:13px;&quot;&gt;Online Hybrid Actor &amp;mu;&lt;sub&gt;&amp;phi;&lt;/sub&gt;&lt;/div&gt;&lt;div style=&quot;font-size:10px;color:#5B6570;&quot;&gt;[s&lt;sub&gt;t&lt;/sub&gt;, Â&lt;sub&gt;f&lt;/sub&gt;(s&lt;sub&gt;t&lt;/sub&gt;)]&lt;/div&gt;",
            STYLE["actor"],
            geom(*BOX["actor"]),
            extra=' connectable="1"',
        ),
    ]

    for nid, nx, ny in nodes:
        parts.append(
            cell(nid, None, STYLE["node"], geom_xy(nx, ny, 14, 14), parent="actor_online", extra=' connectable="0"')
        )
    nlinks = [
        ("nl1", "n11", "n21"),
        ("nl2", "n12", "n22"),
        ("nl3", "n13", "n23"),
        ("nl4", "n21", "n31"),
        ("nl5", "n22", "n32"),
        ("nl6", "n24", "n33"),
    ]
    for eid, s, t in nlinks:
        parts.append(
            f'        <mxCell id="{eid}" parent="actor_online" edge="1" source="{s}" target="{t}" style="{STYLE["nedge"]}" connectable="0">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
    for hid, hval, hx, hy in heads:
        parts.append(
            cell(hid, hval, STYLE["head"], geom_xy(hx, hy, 78, 30), parent="actor_online", extra=' connectable="0"')
        )

    parts += [
        "",
        cell(
            "decoder",
            "&lt;b&gt;Hybrid Decoder&lt;/b&gt;&lt;br&gt;m&lt;sub&gt;t&lt;/sub&gt; = argmax(&amp;ell;&#771;&lt;sub&gt;t&lt;/sub&gt;)&lt;br&gt;a&lt;sub&gt;t&lt;/sub&gt; = decode(&lt;br&gt;u&lt;sub&gt;t&lt;/sub&gt;&lt;sup&gt;th&lt;/sup&gt;, u&lt;sub&gt;t&lt;/sub&gt;&lt;sup&gt;bat&lt;/sup&gt;, m&lt;sub&gt;t&lt;/sub&gt;, z&lt;sub&gt;t&lt;/sub&gt;)",
            STYLE["decoder"],
            geom(*BOX["decoder"]),
        ),
        cell(
            "givesafe",
            "&lt;b&gt;GiveSafe&lt;/b&gt;&lt;br&gt;Physics / residual check",
            STYLE["givesafe"],
            geom(*BOX["givesafe"]),
        ),
        cell(
            "retry",
            "&lt;b&gt;Safety audit / retry&lt;/b&gt;&lt;br&gt;no FMU execution · no economic Bellman update",
            STYLE["retry"],
            geom(*BOX["retry"]),
        ),
        cell(
            "fmu",
            "&lt;b&gt;Sysplorer FMU / Plant Twin&lt;/b&gt;&lt;br&gt;thermal + battery + CAES",
            STYLE["fmu"],
            geom(*BOX["fmu"]),
        ),
        cell(
            "replay",
            "&lt;b&gt;Physical Replay Buffer D&lt;sub&gt;B&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;store accepted physical transitions only&lt;br&gt;(s&lt;sub&gt;t&lt;/sub&gt;, a&lt;sub&gt;t&lt;/sub&gt;, r&lt;sub&gt;t&lt;/sub&gt;, s&lt;sub&gt;t+1&lt;/sub&gt;, d&lt;sub&gt;t&lt;/sub&gt;)",
            STYLE["replay"],
            geom(*BOX["replay"]),
        ),
        cell(
            "batch",
            "Sample mini-batch&lt;br&gt;B ∼ D&lt;sub&gt;B&lt;/sub&gt;",
            STYLE["batch"],
            geom(*BOX["batch"]),
        ),
        cell("critics_lbl", "Online Twin Critics", STYLE["lbl"], geom(*BOX["critics_lbl"])),
        cell(
            "q1",
            "&lt;b&gt;Q&lt;sub&gt;&amp;theta;1&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;Q&lt;sub&gt;&amp;theta;1&lt;/sub&gt;(s, a)",
            STYLE["q"],
            geom(*BOX["q1"]),
        ),
        cell(
            "q2",
            "&lt;b&gt;Q&lt;sub&gt;&amp;theta;2&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;Q&lt;sub&gt;&amp;theta;2&lt;/sub&gt;(s, a)",
            STYLE["q"],
            geom(*BOX["q2"]),
        ),
        cell(
            "critic_loss",
            "L&lt;sub&gt;Q&lt;/sub&gt; = Σ&lt;sub&gt;j=1&lt;/sub&gt;&lt;sup&gt;2&lt;/sup&gt; E[(Q&lt;sub&gt;&amp;theta;j&lt;/sub&gt;(s,a) − y)&lt;sup&gt;2&lt;/sup&gt;]&lt;br&gt;gradient descent on &amp;theta;&lt;sub&gt;1&lt;/sub&gt;, &amp;theta;&lt;sub&gt;2&lt;/sub&gt;",
            STYLE["loss"],
            geom(*BOX["loss"]),
        ),
        cell(
            "target_actor",
            "&lt;b&gt;Target Actor &amp;mu;&lt;sub&gt;&amp;phi;&#772;&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;a′ = &amp;mu;&lt;sub&gt;&amp;phi;&#772;&lt;/sub&gt;(s′) + &amp;epsilon;&lt;br&gt;m′ = argmax &amp;ell;&#771;, noise on z′ only",
            STYLE["tgt"],
            geom(*BOX["tgt_actor"]),
        ),
        cell(
            "target_critics",
            "&lt;b&gt;Target Critics&lt;/b&gt;&lt;br&gt;Q&lt;sub&gt;&amp;theta;&#772;1&lt;/sub&gt;(s′, a′)&lt;br&gt;Q&lt;sub&gt;&amp;theta;&#772;2&lt;/sub&gt;(s′, a′)",
            STYLE["tgt"],
            geom(*BOX["tgt_critics"]),
        ),
        cell(
            "min_q",
            "min(Q&#772;&lt;sub&gt;&amp;theta;1&lt;/sub&gt;, Q&#772;&lt;sub&gt;&amp;theta;2&lt;/sub&gt;)",
            STYLE["chip"],
            geom(*BOX["min_q"]),
        ),
        cell(
            "td_y",
            "y = r + &amp;gamma;(1−d) min&lt;sub&gt;j&lt;/sub&gt; Q&#772;&lt;sub&gt;&amp;theta;j&lt;/sub&gt;(s′, a′)",
            STYLE["q"],
            geom(*BOX["td_y"]),
        ),
        cell(
            "jphi",
            "J(&amp;phi;) = E[Q&lt;sub&gt;&amp;theta;1&lt;/sub&gt;(s, &amp;mu;&lt;sub&gt;&amp;phi;&lt;/sub&gt;(s))]&lt;br&gt;&lt;i&gt;delayed policy update, every d steps&lt;/i&gt;",
            STYLE["jphi"],
            geom(*BOX["jphi"]),
        ),
        cell(
            "caption",
            "Fig. 2. Architecture and data flow of the proposed physics-constrained hybrid TD3 (PC-HybridTD3) algorithm.",
            STYLE["caption"],
            geom(*BOX["caption"]),
        ),
        "",
        # --- online path ---
        edge(
            "e_state_oracle",
            "state",
            "oracle",
            E_GREEN + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;",
            "build Â&lt;sub&gt;f&lt;/sub&gt;(s&lt;sub&gt;t&lt;/sub&gt;)",
        ),
        edge(
            "e_state_actor",
            "state",
            "actor_online",
            E_BLACK + "exitX=1;exitY=0.28;entryX=0;entryY=0.18;",
            "s&lt;sub&gt;t&lt;/sub&gt;",
        ),
        edge(
            "e_oracle_actor",
            "oracle",
            "actor_online",
            E_GREEN + "exitX=1;exitY=0.42;entryX=0;entryY=0.58;",
            "Â&lt;sub&gt;f&lt;/sub&gt;(s&lt;sub&gt;t&lt;/sub&gt;)",
        ),
        edge(
            "e_actor_decoder",
            "actor_online",
            "decoder",
            E_BLACK + "exitX=1;exitY=0.42;entryX=0;entryY=0.45;fontSize=9;",
            "[u&lt;sup&gt;th&lt;/sup&gt;, u&lt;sup&gt;bat&lt;/sup&gt;, &amp;ell;, z] = &amp;mu;&lt;sub&gt;&amp;phi;&lt;/sub&gt;(s&lt;sub&gt;t&lt;/sub&gt;, Â&lt;sub&gt;f&lt;/sub&gt;)",
        ),
        edge(
            "e_decoder_gs",
            "decoder",
            "givesafe",
            E_BLACK + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;",
            "candidate a&lt;sub&gt;t&lt;/sub&gt;",
        ),
        edge(
            "e_gs_fmu",
            "givesafe",
            "fmu",
            E_GREEN + "exitX=1;exitY=0.5;entryX=0;entryY=0.45;fontStyle=1;strokeWidth=1.6;",
            "accepted a&lt;sub&gt;t&lt;/sub&gt;",
        ),
        edge(
            "e_gs_reject",
            "givesafe",
            "retry",
            E_RED + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;",
            "reject",
        ),
        edge(
            "e_retry_actor",
            "retry",
            "actor_online",
            E_RED + "exitX=0;exitY=0.15;entryX=0.96;entryY=1;",
            "retry",
        ),
        edge(
            "e_fmu_replay",
            "fmu",
            "replay",
            E_GREEN + "exitX=0.5;exitY=1;entryX=1;entryY=0.45;strokeWidth=1.5;",
            "(s&lt;sub&gt;t&lt;/sub&gt;, a&lt;sub&gt;t&lt;/sub&gt;) → (r&lt;sub&gt;t&lt;/sub&gt;, s&lt;sub&gt;t+1&lt;/sub&gt;)",
            points=[
                (fmx + fmw // 2, fmy + fmh + 24),
                (fmx + fmw // 2, rpy + rph // 2),
            ],
        ),
        edge(
            "e_fmu_state",
            "fmu",
            "state",
            E_GREEN + "exitX=0.55;exitY=0;entryX=0.55;entryY=0;strokeWidth=1.1;",
            "s&lt;sub&gt;t&lt;/sub&gt; ← s&lt;sub&gt;t+1&lt;/sub&gt;",
            points=[
                (fmx + fmw // 2, 10),
                (stx + stw // 2, 10),
            ],
        ),
        # --- learning path ---
        edge(
            "e_replay_batch",
            "replay",
            "batch",
            E_PURP + "exitX=0;exitY=0.5;entryX=1;entryY=0.5;",
            "B ∼ D&lt;sub&gt;B&lt;/sub&gt;",
        ),
        edge(
            "e_batch_q1",
            "batch",
            "q1",
            E_PURP + "exitX=0.12;exitY=1;entryX=0.5;entryY=0;rounded=1;",
            "(s, a)",
            points=[
                (G(*BOX["q1"])[0] + G(*BOX["q1"])[2] // 2, G(*BOX["q1"])[1] - 14),
            ],
        ),
        edge(
            "e_batch_q2",
            "batch",
            "q2",
            E_PURP + "exitX=0.28;exitY=1;entryX=0.5;entryY=0;rounded=1;",
            points=[
                (G(*BOX["q2"])[0] + G(*BOX["q2"])[2] // 2, G(*BOX["q2"])[1] - 14),
            ],
        ),
        edge(
            "e_batch_tgt",
            "batch",
            "target_actor",
            E_PURP + "exitX=0.55;exitY=1;entryX=0.35;entryY=0;",
            "s&lt;sub&gt;t+1&lt;/sub&gt;",
        ),
        edge(
            "e_tgt_tc",
            "target_actor",
            "target_critics",
            E_POLY + "exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
            "a′",
        ),
        edge(
            "e_tc_min",
            "target_critics",
            "min_q",
            E_PURP + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;",
        ),
        edge(
            "e_min_y",
            "min_q",
            "td_y",
            E_PURP + "exitX=0;exitY=0.5;entryX=1;entryY=0.5;",
        ),
        edge(
            "e_y_loss",
            "td_y",
            "critic_loss",
            E_PURP + "exitX=0;exitY=0.5;entryX=1;entryY=0.5;",
            "y",
        ),
        edge(
            "e_q1_loss",
            "q1",
            "critic_loss",
            E_PURP + "exitX=0.5;exitY=1;entryX=0.28;entryY=0;",
        ),
        edge(
            "e_q2_loss",
            "q2",
            "critic_loss",
            E_PURP + "exitX=0.5;exitY=1;entryX=0.72;entryY=0;",
        ),
        edge(
            "e_q1_actor",
            "q1",
            "actor_online",
            E_PURP + "exitX=0.55;exitY=0;entryX=0.22;entryY=1;strokeWidth=1.6;fontSize=11;fontStyle=1;rounded=1;",
            "∇&lt;sub&gt;&amp;phi;&lt;/sub&gt; J(&amp;phi;)",
            points=[
                (q1x + q1w // 2, jy + jh + 18),
                (ax + 70, jy + jh + 18),
            ],
        ),
        edge(
            "e_polyak_actor",
            "actor_online",
            "target_actor",
            E_POLY + "exitX=1;exitY=0.92;entryX=0.15;entryY=0;",
            "&amp;phi;&#772; ← &amp;tau;&amp;phi; + (1−&amp;tau;)&amp;phi;&#772;&lt;br&gt;Polyak",
            points=[
                (mid_gap_x, ay + ah + 8),
                (mid_gap_x, G(*BOX["tgt_actor"])[1] - 18),
            ],
        ),
        edge(
            "e_polyak_q",
            "critic_loss",
            "target_critics",
            E_POLY + "exitX=0.85;exitY=1;entryX=0.55;entryY=1;",
            "&amp;theta;&#772;&lt;sub&gt;j&lt;/sub&gt; ← &amp;tau;&amp;theta;&lt;sub&gt;j&lt;/sub&gt; + (1−&amp;tau;)&amp;theta;&#772;&lt;sub&gt;j&lt;/sub&gt;",
            points=[
                (lsx + lsw - 20, lsy + lsh + 16),
                (tcx + tcw // 2, lsy + lsh + 16),
            ],
        ),
        "      </root>",
        "    </mxGraphModel>",
        "  </diagram>",
        "</mxfile>",
        "",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")
    for k, spec in BOX.items():
        print(f"  {k:12s} {G(*spec)}")


if __name__ == "__main__":
    main()
