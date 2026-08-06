#!/usr/bin/env python
"""Build Fig. 1/4/5/6 as journal-ready vector concept diagrams.

Text, formulae and arrows remain editable vector elements.  Image-2 source
assets are documented in Paper/figures/image2_sources/README.md and can be
inserted later without changing the paper interface.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Paper" / "figures"
ASSETS = OUT / "image2_sources" / "final"
plt.rcParams.update({"font.family":"DejaVu Sans", "mathtext.fontset":"dejavusans", "font.size":9, "pdf.fonttype":42, "ps.fonttype":42})
C={"ink":"#243244","muted":"#667085","blue":"#1D7ED8","bluef":"#E9F4FF","teal":"#149B9A","tealf":"#E6F7F5","orange":"#E9782F","orangef":"#FFF1E8","purple":"#805AD5","purplef":"#F2ECFF","green":"#3A9D68","greenf":"#EAF7EE","coral":"#D85C6A","coralf":"#FDECEF","line":"#B8C3D1","panel":"#F7F9FC"}

def box(ax,x,y,w,h,t,fc="white",ec=None,fs=8,weight="normal"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025,rounding_size=0.09",fc=fc,ec=ec or C["ink"],lw=1.15,zorder=3))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=C["ink"],fontweight=weight,linespacing=1.15,zorder=4)
def arr(ax,a,b,color=None,label=None,dash=False,rad=0):
    col=color or C["ink"]
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=11,lw=1.35,color=col,linestyle="--" if dash else "-",connectionstyle=f"arc3,rad={rad}",zorder=2))
    if label:
        x=(a[0]+b[0])/2; y=(a[1]+b[1])/2+(0.16 if rad>=0 else -0.16)
        ax.text(x,y,label,ha="center",va="center",fontsize=6.7,color=col,bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=.94),zorder=5)
def title(ax,t,sub):
    ax.text(.35,9.45,t,fontsize=14,fontweight="bold",color=C["ink"]); ax.text(.35,9.08,sub,fontsize=7.6,color=C["muted"])
def save(fig,n):
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/f"{n}.pdf",bbox_inches="tight",pad_inches=.03); fig.savefig(OUT/f"{n}.png",dpi=320,bbox_inches="tight",pad_inches=.03); plt.close(fig); print("wrote",n)
def canvas(h=10):
    f,a=plt.subplots(figsize=(10.2,5.9)); a.set(xlim=(0,14),ylim=(0,h)); a.axis("off"); return f,a
def asset(ax,name,xy,zoom):
    path=ASSETS/name
    if path.is_file():
        ax.add_artist(AnnotationBbox(OffsetImage(plt.imread(path),zoom=zoom),xy,frameon=False,zorder=6))

def topology():
    f,a=canvas(); title(a,"Multi-energy plant and Modelica–FMU digital twin","Physical energy pathways are advanced in the twin; controller commands and observations are exchanged every hour.")
    for y,n,s,col in [(7.2,"SUPPLY","renewables + grid interface",C["blue"]),(5.05,"CONVERSION","thermal unit + CAES operation",C["orange"]),(2.9,"STORAGE","battery + CAES inventories",C["purple"]),(.65,"DEMAND / MARKET","electric load and TOU settlement",C["teal"])]:
        a.add_patch(FancyBboxPatch((.35,y),13.3,1.35,boxstyle="round,pad=.02,rounding_size=.08",fc="white",ec=C["line"],lw=.8)); a.add_patch(Rectangle((.35,y),.34,1.35,fc=col,ec="none")); a.text(.88,y+.83,n,fontsize=8.7,fontweight="bold",color=C["ink"]); a.text(.88,y+.40,s,fontsize=6.8,color=C["muted"])
    box(a,2,7.48,2,.76,"Wind + PV\nrenewable supply",C["bluef"],C["blue"]); box(a,5,7.48,2,.76,"Grid interface\nTOU buy / sell",C["panel"],fs=7.7); box(a,9,7.48,2.5,.76,"Digital twin\nModelica–FMU",C["tealf"],C["teal"])
    box(a,2,5.36,2.4,.84,"Thermal unit\n$u_{\mathrm{tp}}$",C["orangef"],C["orange"]); box(a,5.1,5.36,2.7,.84,"CAES plant\nmode + magnitude",C["purplef"],C["purple"]); box(a,9,5.36,2.8,.84,"Electric power balance\ninside FMU",C["panel"])
    box(a,2.2,3.20,2.5,.84,"Battery\n$u_{\mathrm{bat}}$, SoC",C["greenf"],C["green"]); box(a,6,3.20,2.9,.84,"CAES inventory\ngas / hot / cold SoC",C["purplef"],C["purple"]); box(a,4.1,1.00,5,.88,"Electric load + market settlement",C["tealf"],C["teal"],8.5)
    for u,v,c,l in [((3,7.48),(3.2,6.2),C["blue"],"electricity"),((6,7.48),(6.4,6.2),C["blue"],None),((5,5.36),(5.6,4.04),C["orange"],"thermal / electric"),((7,5.36),(7.3,4.04),C["purple"],"gas + thermal"),((3.4,3.2),(5.0,1.88),C["green"],None),((7.4,3.2),(7.4,1.88),C["purple"],None),((10,7.48),(10.3,6.2),C["teal"],"state / command")]: arr(a,u,v,c,l,dash=l=="state / command")
    asset(a,"renewables_cluster.png",(12.3,7.85),.105); asset(a,"caes_storage_module.png",(12.2,5.80),.078); asset(a,"digital_twin_terminal.png",(11.65,3.55),.060)
    a.text(9.3,.25,"Solid: physical energy flow     Dashed: information / control",fontsize=6.8,color=C["muted"]); save(f,"fig_topology")

def reward():
    f,a=canvas(); title(a,"Reward composition and constraint-aware hierarchy","Economic return, inventory recovery and terminal feasibility are separated before layered learning updates.")
    box(a,.6,6.65,3.2,1.05,"Market cash-flow\n" + r"$\Delta J_t/C_{\mathrm{ref}}\rightarrow r_t^{\mathrm{econ}}$",C["bluef"],C["blue"]); box(a,5.1,6.65,3.65,1.05,"Inventory potential shaping\n$\kappa_t(L_{1,t-1}-L_{1,t})-\kappa_t^{\mathrm{abs}}L_{1,t}$",C["orangef"],C["orange"],7.2); box(a,10.05,6.65,3.3,1.05,"Weekly terminal recovery gate\n" + r"$+B$ if $L_1^e\leq\varepsilon$; else $-pL_1^e$",C["coralf"],C["coral"],7.2)
    box(a,3.25,4.6,7.5,1.15,"$r_t^{\mathrm{ext}}=\mathrm{clip}(r_t^{\mathrm{econ}}+r_t^{\mathrm{shape}}+r_t^{\mathrm{term}},r_{\min},r_{\max})$","white",C["ink"],11,"bold")
    for u,v,c in [((2.2,6.65),(5,5.75),C["blue"]),((6.9,6.65),(7,5.75),C["orange"]),((11.7,6.65),(9.1,5.75),C["coral"])]:arr(a,u,v,c)
    box(a,.7,2.55,4.3,1.12,"Low-level TD3\n" + r"$r_t^{\mathrm{int}}=-\|e_t\|_w+\alpha r_t^{\mathrm{ext}}$" + "\ntracks residual goal + immediate feedback",C["greenf"],C["green"],7.7); box(a,8.95,2.55,4.35,1.12,"High-level TD3\n$R_k^{\mathrm{hi}}=c^{-1}\sum_{j=0}^{c-1}r_{t+j}^{\mathrm{ext}}$\nupdates every $c$ steps",C["purplef"],C["purple"],7.7); arr(a,(5.3,4.6),(2.9,3.67),C["green"],"extrinsic term"); arr(a,(8.7,4.6),(11.1,3.67),C["purple"],"c-step aggregation")
    box(a,3.25,.65,7.5,1.02,"Hard feasibility layer: GiveSafe projection  $a\in\mathcal{F}(s)$  +  FMU post-step checks\nlimits • CAES minimum-run locks • inventory and terminal recovery",C["panel"],C["ink"],8); asset(a,"feasibility_emblem.png",(11.75,1.18),.052); save(f,"fig_reward_structure")

def algorithm():
    f,a=canvas(); title(a,"Safe Market-GHTD3: learning and closed-loop execution","A high-level market/recovery intent is executed by a feasibility-filtered low-level controller on the FMU twin.")
    box(a,.55,5.8,2.35,1.52,"Environment observation\n$s_t$: prices, resources, load,\npower and inventory states",C["bluef"],C["blue"],7.8); box(a,3.55,5.8,2.55,1.52,"High-level TD3\n$\mu^{\mathrm{hi}}(s_t)$ + MSGP\nmarket / recovery prior",C["orangef"],C["orange"],7.8); box(a,6.75,5.8,2.72,1.52,"Absolute goal every $c$ steps\n$g_t=[\Delta\mathrm{bat},\Delta\mathrm{gas},\Delta\mathrm{th},u_{\mathrm{tp}},\mathrm{arb}]$\nMS-HER relabeling",C["purplef"],C["purple"],6.8); box(a,10.1,5.8,2.55,1.52,"Low-level TD3\n$\pi_{\mathrm{lo}}(s_i,\kappa g_i)$\nabsolute hybrid action",C["greenf"],C["green"],7.8)
    box(a,8.2,3.25,2.45,1.18,"GiveSafe\n$\Pi_{\mathcal{F}(s)}(a_i)$",C["coralf"],C["coral"],8.4); box(a,11.15,3.25,2.18,1.18,"Modelica–FMU\nplant step",C["tealf"],C["teal"]); box(a,4.4,3.25,2.8,1.18,"Market reward + state\n$r_i^{\mathrm{ext}},r_i^{\mathrm{int}},s_{i+1}$\nSoC recovery gate",C["panel"],fs=7.6)
    box(a,.8,1,3.15,1.05,"Feasible rule trajectories\nF-MLE warm-start",C["panel"],C["muted"]); box(a,4.6,.82,3.15,1.4,"Low-level replay buffer\n$(s_i,g_i,a_i,r_i^{\mathrm{int}},s_{i+1},g_{i+1})$",C["greenf"],C["green"],7.2); box(a,8.45,.82,3.15,1.4,"High-level replay buffer\n$(s_t,g_t,\sum r^{\mathrm{ext}},s_{t+c})$\nMS-HER / TD3 updates",C["purplef"],C["purple"],7.2)
    for u,v,c,l,d in [((2.9,6.55),(3.55,6.55),C["blue"],"$s_t$",False),((6.1,6.55),(6.75,6.55),C["orange"],"$g_t$",False),((9.47,6.55),(10.1,6.55),C["purple"],"$(s_i,g_i)$",False),((11.35,5.8),(9.45,4.43),C["green"],"$a_i$",False),((10.65,3.84),(11.15,3.84),C["coral"],"safe $a_i$",False),((11.15,3.25),(7.2,3.84),C["teal"],"state + rewards",False),((5.7,3.25),(6.1,2.22),C["green"],"hourly transition",False),((6.95,3.25),(9.52,2.22),C["purple"],"each $c$ steps",False),((6.2,2.22),(11.3,5.8),C["green"],"TD3 update",True),((10,2.22),(4.83,5.8),C["purple"],"TD3 update",True)]:arr(a,u,v,c,l,d)
    asset(a,"digital_twin_terminal.png",(12.45,8.0),.048); asset(a,"feasibility_emblem.png",(9.3,4.0),.035)
    save(f,"fig_algorithm")

def cstep():
    f,a=canvas(); title(a,"$c$-step interaction between goals, hybrid actions, and the FMU","The high-level policy sets an inventory/market intent; the low level executes a safe action every hour and updates the residual goal.")
    xs=[1.35,3.75,6.15,8.55,11.55]; labs=["$t$","$t{+}1$","$\cdots$","$t{+}c{-}1$","$t{+}c$"]; a.plot([.85,12.7],[.85,.85],color=C["ink"],lw=1.25)
    for x,l in zip(xs,labs):a.plot([x,x],[.62,1.1],color=C["ink"],lw=1);a.text(x,.28,l,ha="center",fontsize=9.2)
    box(a,.65,6.08,2.4,1.05,"High-level policy\nobserve $s_t$ and issue $g_t$",C["orangef"],C["orange"]); box(a,4.08,6.08,4.2,1.05,"Residual goal update\n$g_{i+1}=h(s_i,g_i,s_{i+1})=s_i^{\mathrm{int}}+g_i-s_{i+1}^{\mathrm{int}}$",C["purplef"],C["purple"],7.7); box(a,10,6.08,2.65,1.05,"High-level update\n$\sum_{i=t}^{t+c-1}r_i^{\mathrm{ext}}$",C["orangef"],C["orange"])
    box(a,.75,3.42,11.95,1.3,r"Hourly closed-loop action:   $(s_i,g_i)\to a_i\to\Pi_{\mathcal{F}(s)}(a_i)\to(s_{i+1},r_i^{\mathrm{ext}},r_i^{\mathrm{int}})$" + "\npolicy  →  GiveSafe  →  FMU",C["greenf"],C["green"],8.6)
    for x in xs[:-1]:box(a,x-.66,2.08,1.32,.62,"$a_i$","white",C["green"]);arr(a,(x,3.42),(x,2.7),C["green"])
    arr(a,(2.98,6.55),(4.08,6.55),C["orange"],"$g_t$");arr(a,(8.28,6.55),(10,6.55),C["purple"],"aggregate return");arr(a,(1.85,6.08),(1.35,4.72),C["orange"],"$g_t$");arr(a,(6.15,6.08),(6.15,4.72),C["purple"],"$g_i$");arr(a,(11.34,4.72),(11.34,6.08),C["orange"],"terminal state")
    a.text(.87,7.55,"slow decision scale",fontsize=7,color=C["orange"],fontweight="bold");a.text(.87,5.30,"fast execution scale",fontsize=7,color=C["green"],fontweight="bold");save(f,"fig_cstep")

if __name__=="__main__":
    topology(); reward(); algorithm(); cstep()
