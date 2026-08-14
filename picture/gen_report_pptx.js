/**
 * 风光火储经济调度 · 训练结果汇报（领导口径）
 * Run:  node picture/gen_report_pptx.js
 * Out:  picture/HMSD训练结果汇报.pptx
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.join(__dirname, "report_figures");
const OUT = path.join(__dirname, "HMSD训练结果汇报.pptx");
const OUT_ALT = path.join(__dirname, "HMSD训练结果汇报_v2.pptx");

const C = {
  green: "005035",
  greenBright: "00703C",
  greenSoft: "E8F0EB",
  red: "802F2D",
  gold: "A49665",
  ink: "101820",
  body: "333333",
  muted: "5C6570",
  label: "79808A",
  line: "E2E4E6",
  cream: "F7F4EC",
  creamDeep: "EAE4D4",
  offWhite: "FAFAF7",
  white: "FFFFFF",
  coral: "C45C3E",
};

const F = {
  title: "Microsoft YaHei",
  body: "Microsoft YaHei",
  label: "Microsoft YaHei",
};

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE13", width: 13.333, height: 7.5 });
pres.layout = "WIDE13";
pres.title = "风光火储综合能源系统：一周经济调度训练结果";
pres.author = "optimal_demo";
pres.subject = "冬 / 过渡 / 夏考试周，同一厂站同一本账";

const W = 13.333;
const H = 7.5;
const N = 20;

function fig(name) {
  return path.join(FIG, name);
}

function addFooter(slide, page) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 7.18, w: W, h: 0.32,
    fill: { color: C.offWhite }, line: { color: C.offWhite, width: 0 },
  });
  slide.addShape(pres.shapes.LINE, {
    x: 0.45, y: 7.18, w: W - 0.9, h: 0,
    line: { color: C.line, width: 0.75 },
  });
  slide.addText("风光火储经济调度  ·  冬 / 过渡 / 夏考试周", {
    x: 0.45, y: 7.2, w: 8.5, h: 0.26,
    fontFace: F.label, fontSize: 10, color: C.label, margin: 0, valign: "middle",
  });
  slide.addText(`${page}  /  ${N}`, {
    x: W - 2.1, y: 7.2, w: 1.6, h: 0.26,
    fontFace: F.label, fontSize: 10, color: C.label, align: "right", margin: 0, valign: "middle",
  });
}

function addTitleBar(slide, title, eyebrow) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.92,
    fill: { color: C.green }, line: { color: C.green, width: 0 },
  });
  if (eyebrow) {
    slide.addText(eyebrow, {
      x: 0.45, y: 0.06, w: 12.4, h: 0.24,
      fontFace: F.label, fontSize: 10, color: C.gold, charSpacing: 1.2, margin: 0,
    });
    slide.addText(title, {
      x: 0.45, y: 0.30, w: 12.4, h: 0.52,
      fontFace: F.title, fontSize: 22, bold: true, color: C.white, margin: 0,
    });
  } else {
    slide.addText(title, {
      x: 0.45, y: 0.22, w: 12.4, h: 0.52,
      fontFace: F.title, fontSize: 22, bold: true, color: C.white, margin: 0,
    });
  }
}

function th(text) {
  return {
    text,
    options: {
      fill: { color: C.green }, color: C.white, bold: true,
      align: "center", valign: "middle", fontFace: F.body, fontSize: 10, margin: 2,
    },
  };
}
function td(text, opt = {}) {
  return {
    text: String(text),
    options: {
      fill: { color: opt.fill || C.white },
      color: opt.color || C.ink,
      bold: !!opt.bold,
      align: opt.align || "center",
      valign: "middle",
      fontFace: F.body,
      fontSize: opt.fontSize || 10,
      margin: 2,
    },
  };
}



// =====================================================================
// 1 封面
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.green };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: H,
    fill: { color: C.gold }, line: { color: C.gold, width: 0 },
  });
  s.addText("训练结果汇报  ·  2026-08-14", {
    x: 0.7, y: 1.55, w: 11, h: 0.35,
    fontFace: F.label, fontSize: 14, color: C.gold, charSpacing: 2, margin: 0,
  });
  s.addText("风光火储综合能源系统\n一周经济调度", {
    x: 0.7, y: 2.05, w: 12, h: 1.7,
    fontFace: F.title, fontSize: 36, bold: true, color: C.white, margin: 0,
  });
  s.addText("分层调度：上层管库存，下层管机组；不安全指令进不了仿真", {
    x: 0.7, y: 3.95, w: 11.5, h: 0.4,
    fontFace: F.body, fontSize: 16, color: C.cream, margin: 0,
  });
  s.addText([
    { text: "考试  ", options: { color: C.gold, bold: true } },
    { text: "冬、过渡、夏各留一周验收    ", options: { color: C.white } },
    { text: "训练  ", options: { color: C.gold, bold: true } },
    { text: "每季约五千回合    ", options: { color: C.white } },
    { text: "数字  ", options: { color: C.gold, bold: true } },
    { text: "预先定好的一次，不事后挑最好", options: { color: C.white } },
  ], {
    x: 0.7, y: 5.7, w: 12, h: 0.35,
    fontFace: F.body, fontSize: 13, margin: 0,
  });
  s.addText("冬天把机组用起来；夏天循环偏多；过渡季赚钱最多，但周末库存没回到规定范围", {
    x: 0.7, y: 6.15, w: 12, h: 0.3,
    fontFace: F.body, fontSize: 13, color: C.creamDeep, margin: 0,
  });
}

// =====================================================================
// 2 口径
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "先看口径：本汇报只认这一套数字", "怎么读这份汇报");
  const cards = [
    { t: "考试周", d: "冬第 5 周 / 过渡第 18 周 / 夏第 31 周\n前面几周用来训练，这一周留着验收" },
    { t: "看哪笔账", d: "综合收益：售电现金扣掉碳、弃电、缺供、电池磨损\n周末库存：对照崔文，回到初值附近加分" },
    { t: "方法怎么叫", d: "本方法 = 分层调度\n对照 = 单层强化学习、厂站规则、峰谷规则、滚动规划" },
    { t: "曲线怎么看", d: "调度图是一周形态示意\n比高低以考试周表上的数为准" },
  ];
  cards.forEach((c, i) => {
    const x = 0.45 + (i % 4) * 3.2;
    const y = 1.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.05, h: 2.35,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.1, h: 2.35,
      fill: { color: C.green }, line: { color: C.green, width: 0 },
    });
    s.addText(c.t, {
      x: x + 0.25, y: y + 0.18, w: 2.65, h: 0.4,
      fontFace: F.title, fontSize: 16, bold: true, color: C.green, margin: 0,
    });
    s.addText(c.d, {
      x: x + 0.25, y: y + 0.7, w: 2.65, h: 1.45,
      fontFace: F.body, fontSize: 13, color: C.body, margin: 0,
    });
  });

  const notes = [
    ["写进主表", "预先定好的这一次考试周，冬 / 过渡 / 夏各一周"],
    ["不混进主表", "以前较短训练、事后挑过的结果，都不进这套数"],
    ["三季怎么说", "冬天分层最好；过渡赚钱但库存没过门；夏天单层强化学习账更好"],
  ];
  notes.forEach((row, i) => {
    const y = 3.8 + i * 1.0;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.45, y, w: 12.4, h: 0.9,
      fill: { color: i === 2 ? C.cream : C.white },
      line: { color: C.line, width: 0.75 },
    });
    s.addText(row[0], {
      x: 0.65, y: y + 0.22, w: 2.4, h: 0.46,
      fontFace: F.title, fontSize: 15, bold: true, color: C.green, margin: 0, valign: "middle",
    });
    s.addText(row[1], {
      x: 3.2, y: y + 0.22, w: 9.4, h: 0.46,
      fontFace: F.body, fontSize: 15, color: C.ink, margin: 0, valign: "middle",
    });
  });
  addFooter(s, 2);
}

// =====================================================================
// 3 问题
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "问题：分时电价下的一周闭环经济调度", "要解决什么");
  s.addText("电价给定、厂站用高保真仿真。每个小时决定火电、电池、压空怎么动，一周算清一笔账，同时守住设备和电网边界。", {
    x: 0.45, y: 1.12, w: 12.4, h: 0.55,
    fontFace: F.body, fontSize: 15, color: C.body, margin: 0,
  });

  const items = [
    { k: "对象", v: "风电 + 光伏 + 火电 + 电池 + 压缩空气储能\n只做厂站调度，不参与电网出清" },
    { k: "时间", v: "一小时下一道指令\n一次完整调度 = 168 小时（一周）" },
    { k: "动作", v: "火电出力\n电池充放\n压空充电 / 待机 / 放电" },
    { k: "看得见的信息", v: "电池和气库还剩多少、机组状态\n未来 24 小时风光荷预报 + 分时电价" },
  ];
  items.forEach((it, i) => {
    const x = 0.45 + (i % 2) * 6.4;
    const y = 1.85 + Math.floor(i / 2) * 2.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 6.15, h: 2.15,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 6.15, h: 0.08,
      fill: { color: C.green }, line: { color: C.green, width: 0 },
    });
    s.addText(it.k, {
      x: x + 0.25, y: y + 0.25, w: 5.65, h: 0.4,
      fontFace: F.title, fontSize: 16, bold: true, color: C.green, margin: 0,
    });
    s.addText(it.v, {
      x: x + 0.25, y: y + 0.75, w: 5.65, h: 1.2,
      fontFace: F.body, fontSize: 16, color: C.ink, margin: 0,
    });
  });
  addFooter(s, 3);
}

// =====================================================================
// 4 优化目标
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "怎么算账：综合收益，不是只看卖了多少电", "优化目标");

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 1.15, w: 12.4, h: 1.55,
    fill: { color: C.white }, line: { color: C.line, width: 1 },
  });
  s.addText("综合收益  =  经营现金  −  碳成本  −  弃电与缺供  −  电池磨损", {
    x: 0.7, y: 1.28, w: 12.0, h: 0.55,
    fontFace: F.title, fontSize: 20, bold: true, color: C.green, align: "center", margin: 0,
  });
  s.addText("训练时按这一笔账打分。周末库存对照崔文：回到周一初值附近给加分，回不去不加分", {
    x: 0.7, y: 1.9, w: 12.0, h: 0.5,
    fontFace: F.body, fontSize: 15, color: C.body, align: "center", margin: 0,
  });

  const kpis = [
    { n: "80", u: "元 / 吨二氧化碳", l: "碳价" },
    { n: "300", u: "元 / 兆瓦时", l: "弃电惩罚" },
    { n: "1000", u: "元 / 兆瓦时", l: "缺供惩罚" },
    { n: "约 15.7 万", u: "元 / 小时", l: "记账标尺" },
    { n: "加分", u: "周末回到初值附近", l: "库存软门（崔文）" },
  ];
  kpis.forEach((k, i) => {
    const x = 0.45 + i * 2.52;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.95, w: 2.4, h: 1.85,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addText(k.n, {
      x, y: 3.1, w: 2.4, h: 0.7,
      fontFace: F.title, fontSize: 20, bold: true, color: C.green, align: "center", margin: 0,
    });
    s.addText(k.u, {
      x: x + 0.1, y: 3.8, w: 2.2, h: 0.3,
      fontFace: F.body, fontSize: 12, color: C.muted, align: "center", margin: 0,
    });
    s.addText(k.l, {
      x: x + 0.1, y: 4.2, w: 2.2, h: 0.4,
      fontFace: F.body, fontSize: 14, bold: true, color: C.ink, align: "center", margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 5.05, w: 12.4, h: 1.9,
    fill: { color: C.greenSoft }, line: { color: C.greenSoft, width: 0 },
  });
  s.addText("分项含义", {
    x: 0.7, y: 5.18, w: 12, h: 0.32,
    fontFace: F.title, fontSize: 14, bold: true, color: C.green, margin: 0,
  });
  s.addText("现金按购售电价结算。碳按火电和购电折算。弃电、缺供按没送出去或没供上的电量计价。磨损只计电池放电。电池荷电运行中只许大约一成到九成。周末对照崔文：回到初值附近给加分，回不去不加分、不按偏差重罚。", {
    x: 0.7, y: 5.52, w: 11.9, h: 1.25,
    fontFace: F.body, fontSize: 14, color: C.ink, margin: 0,
  });
  addFooter(s, 4);
}

// =====================================================================
// 5 约束
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "约束不靠加大惩罚糊弄：不可行指令进不了仿真", "硬约束");

  const blocks = [
    { t: "设备与电网", d: "火电出力上下限与爬坡\n电池充放功率边界\n联络线功率" },
    { t: "压空只能三段", d: "只能放电、待机、充电\n中间档位不合法\n模式切换有最短运行时间" },
    { t: "安全过滤", d: "先判这条指令能不能执行\n只有安全指令才进厂站仿真\n被拦下的指令记一次负分，状态不动" },
    { t: "周末库存（崔文）", d: "运行中：电池大约 10%–90%\n周末：回到初值附近给加分\n回不去不加分，不改电池指令" },
  ];
  blocks.forEach((b, i) => {
    const x = 0.45 + (i % 4) * 3.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.25, w: 3.05, h: 3.55,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.25, w: 3.05, h: 0.1,
      fill: { color: i === 1 ? C.coral : C.green },
      line: { color: i === 1 ? C.coral : C.green, width: 0 },
    });
    s.addText(String(i + 1).padStart(2, "0"), {
      x: x + 0.2, y: 1.5, w: 2.65, h: 0.55,
      fontFace: F.title, fontSize: 26, bold: true, color: C.gold, margin: 0,
    });
    s.addText(b.t, {
      x: x + 0.2, y: 2.15, w: 2.65, h: 0.7,
      fontFace: F.title, fontSize: 16, bold: true, color: C.ink, margin: 0,
    });
    s.addText(b.d, {
      x: x + 0.2, y: 2.9, w: 2.65, h: 1.65,
      fontFace: F.body, fontSize: 13, color: C.body, margin: 0,
    });
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 5.05, w: 12.4, h: 1.9,
    fill: { color: C.green }, line: { color: C.green, width: 0 },
  });
  s.addText("一句话", {
    x: 0.7, y: 5.2, w: 12, h: 0.32,
    fontFace: F.label, fontSize: 12, color: C.gold, margin: 0,
  });
  s.addText("安全是边学边挡，不是训完再检查。压空不能停在不充不放的中间区；策略可以给出连续数字，真正执行时只落到充、停、放三档。", {
    x: 0.7, y: 5.55, w: 11.9, h: 1.15,
    fontFace: F.body, fontSize: 16, color: C.white, margin: 0,
  });
  addFooter(s, 5);
}

// =====================================================================
// 6 电池库存（崔文做法）
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "电池库存对照崔文：运行卡住，周末只打分", "电池荷电怎么管");

  const cols = [
    {
      t: "运行中：硬盒子",
      d: "电池荷电只许大约 10% 到 90%，防过充过放。越界的充放指令进不了仿真。这一层和崔文一样，不能松。",
    },
    {
      t: "周末：只加分",
      d: "一周结束时，电池（和气库）相对周一初值足够近，给一笔固定加分。回不去：不加分，也不按偏差再罚，更不把这一周作废。",
    },
    {
      t: "不改电池指令",
      d: "周末不抢策略的手、不强行充放把电池扭回去。气库仍可用回收窗。对照崔文：周末回归是奖励项，不是改动作。",
    },
  ];
  cols.forEach((c, i) => {
    const x = 0.45 + i * 4.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.2, w: 4.0, h: 4.0,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.2, w: 4.0, h: 0.1,
      fill: { color: i === 0 ? C.coral : C.green },
      line: { color: i === 0 ? C.coral : C.green, width: 0 },
    });
    s.addText(String(i + 1).padStart(2, "0"), {
      x: x + 0.22, y: 1.45, w: 3.55, h: 0.5,
      fontFace: F.title, fontSize: 24, bold: true, color: C.gold, margin: 0,
    });
    s.addText(c.t, {
      x: x + 0.22, y: 2.05, w: 3.55, h: 0.7,
      fontFace: F.title, fontSize: 18, bold: true, color: C.ink, margin: 0,
    });
    s.addText(c.d, {
      x: x + 0.22, y: 2.85, w: 3.55, h: 2.1,
      fontFace: F.body, fontSize: 14, color: C.body, margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 5.4, w: 12.4, h: 1.55,
    fill: { color: C.green }, line: { color: C.green, width: 0 },
  });
  s.addText("一句话", {
    x: 0.7, y: 5.55, w: 12, h: 0.3,
    fontFace: F.label, fontSize: 12, color: C.gold, margin: 0,
  });
  s.addText("电池荷电：周中用上下限卡住，周末用加分引导回到初值。不靠周末改指令，也不靠回不去就重罚。", {
    x: 0.7, y: 5.9, w: 11.9, h: 0.85,
    fontFace: F.body, fontSize: 16, color: C.white, margin: 0,
  });
  addFooter(s, 6);
}

// =====================================================================
// 7 原理
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "上层管未来几小时库存，下层管这一小时怎么发", "方法");
  s.addImage({
    path: fig("fig_architecture.png"),
    x: 0.35, y: 1.05, w: 12.6, h: 12.6 / 1.923,
    sizing: { type: "contain", w: 12.6, h: 5.7 },
  });
  addFooter(s, 7);
}

// =====================================================================
// 7 训练安排
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "对比方法与训练安排：同一厂站、同一本账、同一考试周", "怎么比");

  s.addTable(
    [
      [th("季节"), th("训练周（轮换）"), th("考试周")],
      [td("冬"), td("第 0–4 周"), td("第 5 周", { bold: true })],
      [td("过渡", { fill: C.cream }), td("第 13–17 周", { fill: C.cream }), td("第 18 周", { bold: true, fill: C.cream })],
      [td("夏"), td("第 26–30 周"), td("第 31 周", { bold: true })],
    ],
    { x: 0.45, y: 1.2, w: 6.2, h: 2.35, colW: [1.6, 2.5, 2.1], border: [{ pt: 0.5, color: C.line }], valign: "middle" },
  );

  s.addTable(
    [
      [th("方法"), th("角色")],
      [td("分层调度", { bold: true, color: C.coral }), td("本方法：上层管库存，下层管机组", { align: "left" })],
      [td("单层 / 对照强化学习", { fill: C.cream }), td("不分层的学习对照", { align: "left", fill: C.cream })],
      [td("厂站 / 峰谷规则"), td("现场保守规则 / 按峰谷电价规则", { align: "left" })],
      [td("滚动规划 / 粒子群", { fill: C.cream }), td("短视域线性规划 / 参数搜索", { align: "left", fill: C.cream })],
    ],
    { x: 6.9, y: 1.2, w: 5.95, h: 2.85, colW: [2.35, 3.6], border: [{ pt: 0.5, color: C.line }] },
  );

  const specs = [
    { n: "5000", l: "回合 / 每一季" },
    { n: "84 万", l: "有效交互步" },
    { n: "每 8 小时", l: "上层下一道库存意图" },
    { n: "预先指定", l: "只报这一次，不挑最好" },
  ];
  specs.forEach((sp, i) => {
    const x = 0.45 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 4.3, w: 3.05, h: 1.55,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addText(sp.n, {
      x, y: 4.45, w: 3.05, h: 0.7,
      fontFace: F.title, fontSize: 24, bold: true, color: C.green, align: "center", margin: 0,
    });
    s.addText(sp.l, {
      x: x + 0.1, y: 5.2, w: 2.85, h: 0.4,
      fontFace: F.body, fontSize: 13, color: C.body, align: "center", margin: 0,
    });
  });
  s.addText("各方法同一厂站、同一本账、同一考试周。分季训练已经跑完；本机没有分季逐步日志，所以不画学习曲线。", {
    x: 0.45, y: 6.05, w: 12.4, h: 0.35,
    fontFace: F.body, fontSize: 13, italic: true, color: C.muted, margin: 0,
  });
  addFooter(s, 8);
}

// =====================================================================
// 8 总览
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "总览：冬、过渡经济领先；夏天单层强化学习账更好", "结果总览");
  s.addImage({
    path: fig("fig_kpi_jgen_bars.png"),
    x: 0.3, y: 1.05, w: 7.5, h: 7.5 / 2.461,
    sizing: { type: "contain", w: 7.5, h: 3.05 },
  });
  s.addImage({
    path: fig("fig_kpi_reward_soc.png"),
    x: 0.3, y: 4.1, w: 7.5, h: 7.5 / 2.214,
    sizing: { type: "contain", w: 7.5, h: 2.7 },
  });

  const takeaways = [
    { h: "冬季", t: "分层调度  综合收益 1600 万元\n比单层多 1570 万元\n单层几乎不干活" },
    { h: "过渡", t: "分层调度仍最能赚钱\n但周末库存没过门\n粒子群现金接近、周没跑满" },
    { h: "夏季", t: "单层周评分 97.9，分层 61.2\n分层多循环、多烧火电\n外部成本几乎翻倍" },
  ];
  takeaways.forEach((tk, i) => {
    const y = 1.15 + i * 1.85;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 8.05, y, w: 4.85, h: 1.7,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 8.05, y, w: 0.1, h: 1.7,
      fill: { color: i === 2 ? C.coral : C.green },
      line: { color: i === 2 ? C.coral : C.green, width: 0 },
    });
    s.addText(tk.h, {
      x: 8.35, y: y + 0.12, w: 4.4, h: 0.35,
      fontFace: F.title, fontSize: 16, bold: true, color: C.ink, margin: 0,
    });
    s.addText(tk.t, {
      x: 8.35, y: y + 0.5, w: 4.4, h: 1.1,
      fontFace: F.body, fontSize: 13, color: C.body, margin: 0,
    });
  });
  addFooter(s, 9);
}

// =====================================================================
// 9 购售电
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "购售电：冬天分层在卖电，夏天单层卖得更多", "市场结算");
  s.addImage({
    path: fig("fig_market_buysell.png"),
    x: 0.25, y: 1.05, w: 12.8, h: 12.8 / 2.6,
    sizing: { type: "contain", w: 12.8, h: 4.0 },
  });
  const mk = [
    { t: "冬季", d: "分层售电 2.29 万兆瓦时，购电仅 81。单层售电 559——闲置之后几乎不参与市场。" },
    { t: "过渡", d: "分层购 5145 / 售 1.64 万。单层购售都极低。滚动规划购 7507、售 8656，综合账已超过厂规。" },
    { t: "夏季", d: "单层售电 1.90 万，分层 1.50 万。不能说分层一定更会卖电。" },
  ];
  mk.forEach((row, i) => {
    const y = 5.15 + i * 0.55;
    s.addText(row.t, {
      x: 0.45, y, w: 1.3, h: 0.5,
      fontFace: F.title, fontSize: 13, bold: true, color: C.green, margin: 0, valign: "middle",
    });
    s.addText(row.d, {
      x: 1.8, y, w: 11.0, h: 0.5,
      fontFace: F.body, fontSize: 13, color: C.ink, margin: 0, valign: "middle",
    });
  });
  addFooter(s, 10);
}

// =====================================================================
// 10 冬季
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "冬季：分层把库存跨天用起来，单层几乎闲置", "冬季 · 第 5 周");
  s.addTable(
    [
      [th("方法"), th("周评分"), th("综合收益 / 万元"), th("库存"), th("售电"), th("火电")],
      [td("厂站规则"), td("54.28"), td("627"), td("过"), td("—"), td("25200")],
      [td("滚动规划"), td("67.78"), td("894"), td("过"), td("12772"), td("20682")],
      [td("对照强化学习", { fill: C.cream }), td("102.92", { fill: C.cream }), td("1482", { fill: C.cream }), td("过", { fill: C.cream }), td("20866", { fill: C.cream }), td("12440", { fill: C.cream })],
      [td("单层强化学习"), td("−1.26", { color: C.red }), td("29"), td("未过", { color: C.red }), td("559"), td("398")],
      [td("分层调度", { bold: true, fill: "F3E6E0" }), td("113.61", { bold: true, fill: "F3E6E0" }), td("1599", { bold: true, fill: "F3E6E0" }), td("过", { bold: true, fill: "F3E6E0" }), td("22877", { fill: "F3E6E0" }), td("9700", { fill: "F3E6E0" })],
    ],
    { x: 0.45, y: 1.15, w: 12.4, h: 2.7, colW: [2.5, 1.6, 2.3, 1.4, 2.3, 2.3], border: [{ pt: 0.5, color: C.line }] },
  );
  const pts = [
    { t: "相对单层", d: "综合收益多 1570 万元。单层火电只发了 398 兆瓦时，几乎不发电、不储电。" },
    { t: "相对滚动规划", d: "修好后的滚动规划综合收益 894 万、过门、火电 2.07 万。分层仍多 705 万，火电只要 9700。" },
    { t: "相对对照", d: "对照强化学习也过门，综合收益 1482 万，接近分层；但电池几乎不动，压空吞吐更大。" },
  ];
  pts.forEach((p, i) => {
    const y = 4.1 + i * 0.9;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.45, y, w: 12.4, h: 0.82,
      fill: { color: C.white }, line: { color: C.line, width: 0.75 },
    });
    s.addText(p.t, {
      x: 0.65, y, w: 2.2, h: 0.82,
      fontFace: F.title, fontSize: 14, bold: true, color: C.green, margin: 0, valign: "middle",
    });
    s.addText(p.d, {
      x: 2.9, y, w: 9.7, h: 0.82,
      fontFace: F.body, fontSize: 14, color: C.ink, margin: 0, valign: "middle",
    });
  });
  addFooter(s, 11);
}

// =====================================================================
// 11 过渡
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "过渡季：最能赚钱，但周末库存必须写没过门", "过渡季 · 第 18 周");
  s.addTable(
    [
      [th("方法"), th("周评分"), th("综合收益 / 万元"), th("库存"), th("购电"), th("售电")],
      [td("厂站规则"), td("8.40"), td("−89"), td("过"), td("—"), td("—")],
      [td("滚动规划", { fill: C.cream }), td("26.77", { fill: C.cream }), td("270", { fill: C.cream }), td("过", { fill: C.cream }), td("7507", { fill: C.cream }), td("8656", { fill: C.cream })],
      [td("单层强化学习"), td("−0.29"), td("13"), td("未过", { color: C.red }), td("374"), td("834")],
      [td("分层调度", { bold: true, fill: "F3E6E0" }), td("50.11", { bold: true, fill: "F3E6E0" }), td("843", { bold: true, fill: "F3E6E0" }), td("未过", { bold: true, fill: "F3E6E0", color: C.red }), td("5145", { fill: "F3E6E0" }), td("16427", { fill: "F3E6E0" })],
    ],
    { x: 0.45, y: 1.15, w: 12.4, h: 2.7, colW: [2.5, 1.6, 2.3, 1.6, 2.2, 2.2], border: [{ pt: 0.5, color: C.line }] },
  );

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 4.15, w: 6.05, h: 2.7,
    fill: { color: C.white }, line: { color: C.line, width: 1 },
  });
  s.addText("读数", {
    x: 0.7, y: 4.3, w: 5.6, h: 0.35,
    fontFace: F.title, fontSize: 15, bold: true, color: C.green, margin: 0,
  });
  s.addText("分层调度综合账仍最高。滚动规划已过门、超过厂规（270 万对 −89 万），但视界只有一天，跨日库存仍不如分层。单层再次近乎闲置。", {
    x: 0.7, y: 4.75, w: 5.55, h: 1.85,
    fontFace: F.body, fontSize: 14, color: C.ink, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.7, y: 4.15, w: 6.15, h: 2.7,
    fill: { color: "F8EEEA" }, line: { color: C.line, width: 1 },
  });
  s.addText("必须写进汇报的限制", {
    x: 6.95, y: 4.3, w: 5.7, h: 0.35,
    fontFace: F.title, fontSize: 15, bold: true, color: C.red, margin: 0,
  });
  s.addText("周末库存相对初值偏了约 8%，门槛是约 6%。分层本该把一周库存收回来，这一次过渡季没有收住。赚钱最多不等于约束过关。", {
    x: 6.95, y: 4.75, w: 5.7, h: 1.85,
    fontFace: F.body, fontSize: 14, color: C.ink, margin: 0,
  });
  addFooter(s, 12);
}

// =====================================================================
// 12 奖励拆分
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "评分拆开看：过渡季经营账仍最高，被周末库存扣住", "评分从哪来");
  s.addImage({
    path: fig("fig_reward_parts.png"),
    x: 0.3, y: 1.05, w: 12.7, h: 12.7 / 2.55,
    sizing: { type: "contain", w: 12.7, h: 3.7 },
  });
  const rp = [
    { t: "冬季", d: "分层经营项 102.1，周末库存加 15 分。单层经营项几乎为 0，周末也没有加分。" },
    { t: "过渡", d: "分层经营项 53.8 仍是全场最高，但周末库存没过门，这 15 分没拿到。周评分 50.1 是被库存拖住后的数。" },
    { t: "夏季", d: "两边都过门、都拿到周末加分。单层经营项 88.5，分层 50.7——差距来自经营账，不是库存门。" },
  ];
  rp.forEach((row, i) => {
    const y = 4.95 + i * 0.62;
    s.addText(row.t, {
      x: 0.45, y, w: 1.3, h: 0.56,
      fontFace: F.title, fontSize: 13, bold: true, color: C.green, margin: 0, valign: "middle",
    });
    s.addText(row.d, {
      x: 1.8, y, w: 11.0, h: 0.56,
      fontFace: F.body, fontSize: 13, color: C.ink, margin: 0, valign: "middle",
    });
  });
  addFooter(s, 13);
}

// =====================================================================
// 13 夏季
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "夏季：单层少循环、少烧火电，账反而更好", "夏季 · 第 31 周");
  s.addTable(
    [
      [th("方法"), th("周评分"), th("综合收益 / 万元"), th("库存"), th("售电"), th("火电"), th("外部成本")],
      [td("厂站规则"), td("22.06"), td("121"), td("过"), td("—"), td("25200"), td("—")],
      [td("滚动规划"), td("42.51"), td("520"), td("过"), td("9979"), td("20556"), td("172 万")],
      [td("对照强化学习", { fill: C.cream }), td("79.30", { fill: C.cream }), td("1119", { fill: C.cream }), td("过", { fill: C.cream }), td("17095", { fill: C.cream }), td("12473", { fill: C.cream }), td("99 万", { fill: C.cream })],
      [td("单层强化学习", { bold: true, fill: "E8F0EB" }), td("97.89", { bold: true, fill: "E8F0EB" }), td("1386", { bold: true, fill: "E8F0EB" }), td("过", { fill: "E8F0EB" }), td("18997", { fill: "E8F0EB" }), td("9025", { fill: "E8F0EB" }), td("72 万", { fill: "E8F0EB" })],
      [td("分层调度", { fill: "F3E6E0" }), td("61.22", { fill: "F3E6E0" }), td("794", { fill: "F3E6E0" }), td("过", { fill: "F3E6E0" }), td("15000", { fill: "F3E6E0" }), td("15624", { fill: "F3E6E0" }), td("148 万", { fill: "F3E6E0" })],
    ],
    { x: 0.35, y: 1.15, w: 12.6, h: 2.7, colW: [2.2, 1.4, 2.1, 1.2, 1.7, 1.7, 1.7], border: [{ pt: 0.5, color: C.line }] },
  );

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 4.15, w: 12.4, h: 2.7,
    fill: { color: C.green }, line: { color: C.green, width: 0 },
  });
  s.addText("夏天不要写成全面第一", {
    x: 0.7, y: 4.35, w: 12, h: 0.4,
    fontFace: F.title, fontSize: 18, bold: true, color: C.gold, margin: 0,
  });
  s.addText("分层过了库存门，但电池吞吐 4328 对 644、火电 15624 对 9025、售电 1.50 万对 1.90 万。循环多、火电烧得多，综合账输了。分层的价值在冬天和过渡季的跨日库存，不在每一季都第一。", {
    x: 0.7, y: 4.85, w: 11.9, h: 1.75,
    fontFace: F.body, fontSize: 15, color: C.white, margin: 0,
  });
  addFooter(s, 14);
}

// =====================================================================
// 14 拒绝率
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "安全过滤：单层冬天六成指令被拦，和闲置对得上", "指令拦得住拦不住");
  s.addImage({
    path: fig("fig_reject_td3_sac.png"),
    x: 0.25, y: 1.05, w: 8.15, h: 8.15 / 2.54,
    sizing: { type: "contain", w: 8.15, h: 3.4 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.55, y: 1.15, w: 4.35, h: 3.3,
    fill: { color: C.white }, line: { color: C.line, width: 1 },
  });
  s.addText("考试周能说的", {
    x: 8.75, y: 1.3, w: 4.0, h: 0.35,
    fontFace: F.title, fontSize: 14, bold: true, color: C.green, margin: 0,
  });
  s.addText("单层冬天 63.8%、过渡 51.9%、夏天 29.3% 的指令被安全层拦住。对照强化学习四季都在四到五成。分层调度这张考试周没有单独记下拦截次数，表上留空。", {
    x: 8.75, y: 1.75, w: 3.95, h: 2.5,
    fontFace: F.body, fontSize: 13, color: C.ink, margin: 0,
  });

  s.addTable(
    [
      [th("另一批典型周（旁证）"), th("被拦比例"), th("指令被改了多少"), th("仿真跑飞")],
      [td("分层调度"), td("14.3%"), td("0.13"), td("0")],
      [td("单层强化学习", { fill: C.cream }), td("65.8%", { fill: C.cream }), td("0.71", { fill: C.cream }), td("0.65%", { fill: C.cream })],
      [td("厂站规则"), td("4.6%"), td("0.03"), td("0")],
    ],
    { x: 0.45, y: 4.65, w: 12.4, h: 1.55, colW: [3.6, 2.9, 2.9, 3.0], border: [{ pt: 0.5, color: C.line }] },
  );
  s.addText("下表是另一批典型周，用来看方向：单层更常撞到不能执行的指令。分层考试周的拦截次数仍以空白为准。", {
    x: 0.45, y: 6.35, w: 12.4, h: 0.45,
    fontFace: F.body, fontSize: 13, italic: true, color: C.muted, margin: 0,
  });
  addFooter(s, 15);
}

// =====================================================================
// 15 冬季调度
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "冬季调度：火电压在下限，白天风光加电池跟电价吞吐", "冬季一周形态");
  s.addImage({
    path: fig("fig_dispatch_winter_ghtd3.png"),
    x: 0.35, y: 1.05, w: 12.6, h: 5.2,
    sizing: { type: "contain", w: 12.6, h: 5.2 },
  });
  s.addText("示意一周形态，表上数字以考试周为准。电池跟着电价吞吐，气库几乎不动。", {
    x: 0.45, y: 6.35, w: 12.4, h: 0.35,
    fontFace: F.body, fontSize: 13, italic: true, color: C.muted, margin: 0,
  });
  addFooter(s, 16);
}

// =====================================================================
// 16 夏季对照
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "夏季调度对照：两边都能消纳风光，差在火电和储能用法", "夏季一周形态");
  s.addImage({
    path: fig("fig_dispatch_compare_summer.png"),
    x: 0.35, y: 1.02, w: 12.6, h: 5.2,
    sizing: { type: "contain", w: 12.6, h: 5.2 },
  });
  s.addText("示意一周形态。考试周表上：单层火电更少、电池吞吐更低；本图用来看风光消纳长什么样。", {
    x: 0.45, y: 6.35, w: 12.4, h: 0.35,
    fontFace: F.body, fontSize: 13, italic: true, color: C.muted, margin: 0,
  });
  addFooter(s, 17);
}

// =====================================================================
// 17 弃电
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "弃风弃光：本批考试周拉不开差距", "消纳");
  s.addImage({
    path: fig("fig_renewable_util.png"),
    x: 0.3, y: 1.05, w: 12.7, h: 12.7 / 3.059,
    sizing: { type: "contain", w: 12.7, h: 3.15 },
  });

  s.addTable(
    [
      [th("季节"), th("分层弃电"), th("单层弃电"), th("缺供"), th("结论")],
      [td("冬"), td("约 0"), td("0"), td("0"), td("可发尽发", { align: "left" })],
      [td("过渡", { fill: C.cream }), td("约 0", { fill: C.cream }), td("0", { fill: C.cream }), td("0", { fill: C.cream }), td("可发尽发", { fill: C.cream, align: "left" })],
      [td("夏"), td("约 0"), td("约 0"), td("0"), td("可发尽发", { align: "left" })],
    ],
    { x: 0.45, y: 4.35, w: 12.4, h: 2.0, colW: [1.8, 2.3, 2.3, 1.8, 4.2], border: [{ pt: 0.5, color: C.line }] },
  );
  s.addText("账里写了弃电惩罚，是为了兜底。本批考试周联络线和负荷够用，各方法弃电都是数值噪声。方法之间的差距在火电、购售电和库存，不在弃风弃光。", {
    x: 0.45, y: 6.45, w: 12.4, h: 0.55,
    fontFace: F.body, fontSize: 13, color: C.body, margin: 0,
  });
  addFooter(s, 18);
}

// =====================================================================
// 18 账本拆细
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "账本拆细：冬天靠卖电，夏天输在燃料和碳", "钱从哪来、花到哪去");
  s.addImage({
    path: fig("fig_cash_split.png"),
    x: 0.35, y: 1.02, w: 12.6, h: 12.6 / 2.6,
    sizing: { type: "contain", w: 12.6, h: 3.5 },
  });
  const rows = [
    ["冬天分层电网结算约 +425 万，火电现金约 −388 万；单层几乎不参与电网。"],
    ["碳和电池磨损记为支出：冬天分层合计约 108 万，仍远小于多出来的售电现金。"],
    ["夏天分层火电现金约 −625 万，单层约 −361 万；碳 124 万对 69 万。综合账输在燃料和碳。"],
  ];
  rows.forEach((r, i) => {
    const y = 4.65 + i * 0.75;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.45, y, w: 12.4, h: 0.68,
      fill: { color: C.white }, line: { color: C.line, width: 0.75 },
    });
    s.addShape(pres.shapes.OVAL, {
      x: 0.65, y: y + 0.18, w: 0.32, h: 0.32,
      fill: { color: C.green }, line: { color: C.green, width: 0 },
    });
    s.addText(String(i + 1), {
      x: 0.65, y: y + 0.18, w: 0.32, h: 0.32,
      fontFace: F.body, fontSize: 11, bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
    });
    s.addText(r[0], {
      x: 1.15, y, w: 11.5, h: 0.68,
      fontFace: F.body, fontSize: 14, color: C.ink, margin: 0, valign: "middle",
    });
  });
  addFooter(s, 19);
}

// =====================================================================
// 19 结论
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  addTitleBar(s, "结论：分层对准的是跨日库存，不是每一季都第一", "带走三句话");

  const left = [
    { n: "01", t: "问题清楚", d: "一小时下一道指令，一周算清综合收益。电池荷电周中卡住、周末对照崔文只加分。碳、弃电、磨损都进账。" },
    { n: "02", t: "冬天机制成立", d: "上层把库存跨天用起来。单层闲置，厂站规则满发火电。" },
    { n: "03", t: "过渡能赚钱", d: "综合账最高，但周末库存没过门。这一季没有把库存收回来。" },
  ];
  const right = [
    { n: "04", t: "夏天如实写", d: "单层少循环、少烧火电、售电更多。不要包装成全面第一。" },
    { n: "05", t: "弃电不是故事", d: "弃电、缺供都是零。单层冬天六成指令被拦，能解释闲置。" },
    { n: "06", t: "数字从哪来", d: "考试周是预先定好的，不事后挑。调度图是一周形态示意。电池做法对照崔文。" },
  ];
  left.forEach((it, i) => {
    const y = 1.2 + i * 1.8;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.45, y, w: 6.1, h: 1.65,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addText(it.n, {
      x: 0.65, y: y + 0.15, w: 0.7, h: 0.4,
      fontFace: F.title, fontSize: 16, bold: true, color: C.gold, margin: 0,
    });
    s.addText(it.t, {
      x: 1.4, y: y + 0.18, w: 4.9, h: 0.38,
      fontFace: F.title, fontSize: 16, bold: true, color: C.ink, margin: 0,
    });
    s.addText(it.d, {
      x: 0.65, y: y + 0.65, w: 5.7, h: 0.85,
      fontFace: F.body, fontSize: 13, color: C.body, margin: 0,
    });
  });
  right.forEach((it, i) => {
    const y = 1.2 + i * 1.8;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.75, y, w: 6.1, h: 1.65,
      fill: { color: C.white }, line: { color: C.line, width: 1 },
    });
    s.addText(it.n, {
      x: 6.95, y: y + 0.15, w: 0.7, h: 0.4,
      fontFace: F.title, fontSize: 16, bold: true, color: C.gold, margin: 0,
    });
    s.addText(it.t, {
      x: 7.7, y: y + 0.18, w: 4.9, h: 0.38,
      fontFace: F.title, fontSize: 16, bold: true, color: C.ink, margin: 0,
    });
    s.addText(it.d, {
      x: 6.95, y: y + 0.65, w: 5.7, h: 0.85,
      fontFace: F.body, fontSize: 13, color: C.body, margin: 0,
    });
  });
  addFooter(s, 20);
}

pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Wrote", OUT);
}).catch((err) => {
  if (err && err.code === "EBUSY") {
    return pres.writeFile({ fileName: OUT_ALT }).then(() => {
      console.log("Locked, wrote", OUT_ALT);
    });
  }
  console.error(err);
  process.exit(1);
});
