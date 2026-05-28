import os
import json
from pathlib import Path

import requests


LLM_CONFIG = {
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "api_key_env": "DEEPSEEK_API_KEY",
    "model": "deepseek-chat",
    "enabled": True,
}

_STRATEGY_DESC = {
    "risk_parity": "风险平价：权重∝1/vol，低波动品种仓位更高。纯风控，不直接用模型评分。",
    "score_risk": "评分风险平价：权重∝score/vol²，评分高且波动低的品种权重更高，波动惩罚为平方。",
    "score_risk_v1": "评分风险平价V1：权重∝score/vol，与score_risk类似但波动惩罚降为线性，分布更均匀。",
    "equal": "等权：所有选中品种权重相同。",
    "softmax": "Softmax概率加权，temperature参数控制集中度。",
    "rank_linear": "线性排名：按排名等差递减分配权重。",
    "kelly": "Kelly最优增长：基于评分和波动率计算最优下注比例。",
    "liquidity": "流动性优先：按成交额分配权重。",
}

_MODEL_ARCH = {
    "patchtst": "PatchTST — Transformer分块时序，将序列分段为patch后输入Transformer编码器",
    "tcn": "TCN — 时序卷积网络，带空洞卷积和残差连接",
    "dlinear": "DLinear — 线性分解为趋势项+季节项",
    "gru": "GRU — 门控循环单元+跨股票注意力机制",
    "itransformer": "iTransformer — 倒置Transformer(特征维→token)",
    "timesnet": "TimesNet — 多周期建模，2D卷积",
    "nlinear": "NLinear — 标准化线性层",
    "mamba": "Mamba — 状态空间模型",
    "lightgbm": "LightGBM — 梯度提升树",
}


def _load_env():
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("\"'")
                if k not in os.environ:
                    os.environ[k] = v


def build_llm_analysis_section(report_data: dict) -> str:
    if not LLM_CONFIG["enabled"]:
        return _placeholder_section()
    _load_env()
    api_key = os.environ.get(LLM_CONFIG["api_key_env"], "")
    if not api_key:
        return _placeholder_section(msg=f"未设置 {LLM_CONFIG['api_key_env']}")
    prompt = _build_prompt(report_data)
    try:
        analysis = _call_deepseek(prompt, api_key)
        return _format_section(analysis)
    except Exception as e:
        return _placeholder_section(msg=f"AI 分析暂不可用: {e}")


def _build_prompt(data: dict) -> str:
    m = data.get("metrics", {})
    h = data.get("holdings", [])
    w = data.get("windows", {})
    ms = data.get("market_stats", {})
    mr = data.get("market_regime", {})
    mb = data.get("market_breadth", {})
    er = data.get("etf_rankings", {})
    tv = data.get("total_value", 0)
    ca = data.get("cash", 0)
    ir = data.get("is_rebalance_day", False)
    nr = data.get("next_rebalance_date", "")
    tc = data.get("trades_count", 0)

    # ---------- 区块1: 组合概览 ----------
    leverage = (1 - ca / tv) * 100 if tv > 0 else 0
    block1 = (
        f"总资产 {tv:,.0f} | 现金 {ca:,.0f} | 仓位 {leverage:.1f}%\n"
        f"累计 {m.get('strategy_return_pct', 0):+.2f}% | "
        f"HS300 {m.get('hs300_return_pct', 0):+.2f}% | "
        f"超额 {m.get('excess_return_pct', 0):+.2f}%\n"
        f"年化 {m.get('annualized_return_pct', 0):+.1f}% | "
        f"波动 {m.get('annualized_volatility_pct', 0):.1f}% | "
        f"夏普 {m.get('sharpe_ratio', 0):.2f} | "
        f"卡玛 {m.get('calmar_ratio', 0):.1f}\n"
        f"日胜率 {m.get('daily_win_rate', 0)*100:.0f}% | "
        f"盈亏比 {m.get('profit_factor', 0):.2f} | "
        f"VaR95 {m.get('var_95', 0):.2f}% | "
        f"CVaR95 {m.get('cvar_95', 0):.2f}%\n"
        f"最大回撤 {m.get('max_drawdown_pct', 0):.2f}% "
        f"(修复 {m.get('max_recovery_days', 0)}天)"
    )

    # ---------- 区块2: 持仓明细 ----------
    _er_map = {e.get("code", e.get("stock_id", "")): e for e in er.get("holdings", [])}
    h_rows = []
    for x in h:
        code = x.get("code", x.get("stock_id", ""))
        name = x.get("name", "")
        wt = x.get("weight", 0)
        px = x.get("price", 0)
        bp = x.get("buy_price", 0)
        ret = (px / bp - 1) * 100 if bp > 0 else 0
        pnl = x.get("pnl", 0)
        _r = _er_map.get(code, {})
        rk = _r.get("rank", "")
        r5 = _r.get("return", None)
        rk_str = f"{rk}/74" if rk else "-"
        r5_str = f"{r5:+.2f}%" if r5 is not None else "-"
        h_rows.append(f"{code:>10} {name:<12} {wt:>6.1f}% {px:>6.3f} {bp:>6.3f} {ret:>+6.2f}% {pnl:>+8.0f} {rk_str:>6} {r5_str:>8}")
    header = f"{'代码':>10} {'名称':<12} {'仓位':>6} {'现价':>6} {'成本':>6} {'盈亏%':>6} {'今日盈亏':>8} {'5日排名':>6} {'5日涨跌':>8}"
    sep = "-" * 80
    block2 = "\n".join([header, sep] + h_rows) if h_rows else "无持仓"

    # ---------- 区块3: 市场状态 ----------
    reg = mr.get("regime", "?")
    r20 = mr.get("rolling_20d_return", 0)
    rv = mr.get("rolling_vol", 0)
    bpct = mb.get("bull_pct", 0)
    spct = mb.get("sideways_pct", 0)
    bpct2 = mb.get("bear_pct", 0)
    total_etf = mb.get("total", 74)
    block3_lines = [f"当前: {reg} | 20日滚动 {r20:+.2f}% | 波动率 {rv:.1f}%"]
    block3_lines.append(f"宽度: 牛{bpct:.0f}% / 震{spct:.0f}% / 熊{bpct2:.0f}% ({total_etf}只)")
    for regime_key, label in [("bull", "牛市"), ("sideways", "震荡"), ("all", "总计")]:
        s = ms.get(regime_key, {})
        days = s.get("days", 0)
        mr_ret = s.get("model_return", 0)
        hs_ret = s.get("hs300_return", 0)
        wr = s.get("model_win_rate", 0)
        exc = mr_ret - hs_ret
        if days:
            block3_lines.append(f"  {label}({days}天): 模型{mr_ret:+.2f}% vs HS300{hs_ret:+.2f}% | 超额{exc:+.2f}% | 日胜率{wr*100:.0f}%")
    block3 = "\n".join(block3_lines)

    # ---------- 区块4: ETF排行榜 ----------
    block4_lines = []
    top_list = er.get("top", [])
    bot_list = er.get("bottom", [])
    period = er.get("period", "")
    if top_list:
        block4_lines.append(f"强势板块(5日涨幅Top5, {period}):")
        for e in top_list[:5]:
            _c = e.get("code", e.get("stock_id", ""))
            _n = e.get("name", "")
            _r = e.get("return", 0)
            _tag = " ← 当前持仓" if any(h.get("code", h.get("stock_id", "")) == _c for h in h) else ""
            block4_lines.append(f"  {_c} {_n} {_r:+.2f}%{_tag}")
    if bot_list:
        block4_lines.append(f"弱势板块(Bottom5, {period}):")
        for e in bot_list[:5]:
            _c = e.get("code", e.get("stock_id", ""))
            _n = e.get("name", "")
            _r = e.get("return", 0)
            block4_lines.append(f"  {_c} {_n} {_r:+.2f}%")
    block4 = "\n".join(block4_lines) if block4_lines else "暂无排行数据"

    # ---------- 区块5: 近期窗口 ----------
    win_str = (
        f"近3天 {w.get('3d', 0):+.2f}% | "
        f"近5天 {w.get('5d', 0):+.2f}% | "
        f"近20天 {w.get('1m', 0):+.2f}%"
    )
    rb_tag = "是" if ir else "否"
    strategy_info = data.get("strategy_info", "")
    block5 = f"{win_str}\n今日调仓: {rb_tag} | 下次调仓: {nr}\n{strategy_info}"

    # ---------- 区块6: 交易概况 ----------
    block6 = f"累计交易: {tc}笔"

    # ---------- 区块7: 模型表现 ----------
    _seqs = data.get("sequences_summary", {})
    _pri_key = data.get("model_key", "")
    _pri_display = data.get("model_display", _pri_key)

    def _short_name(key):
        if key in ("average", "voting", "juejin"):
            return {"average": "平均", "voting": "投票", "juejin": "掘金"}.get(key, key)
        parts = key.split("_")
        if len(parts) >= 2 and parts[-2] == "exp":
            return f"{parts[1]}_exp{parts[-1]}"
        if len(parts) >= 2:
            return f"{parts[1]}_{parts[-1]}"
        return key

    block7_lines = [f"主模型: {_pri_display} ({data.get('strategy_info', '')})"]

    # 主模型详细统计
    pri = _seqs.get(_pri_key, {})
    if pri:
        _wr = pri.get("win_rate", 0)
        _avg = pri.get("avg_return", 0)
        _trades = pri.get("total_trades", 0)
        _l3a = pri.get("last_3_avg", 0)
        _l3w = pri.get("last_3_win_rate", 0)
        _ic = pri.get("rank_ic")
        _ndcg = pri.get("ndcg")
        _hs = pri.get("health_score", 0)
        block7_lines.append(f"交易: {_trades}笔 | 胜率: {_wr:.1f}% | 平均收益: {_avg:+.2f}%")
        block7_lines.append(f"近3笔平均: {_l3a:+.2f}% | 近3笔胜率: {_l3w:.1f}%")
        _ic_str = f"{_ic:.3f}" if _ic is not None else "N/A"
        _ndcg_str = f"{_ndcg:.3f}" if _ndcg is not None else "N/A"
        block7_lines.append(f"Rank IC: {_ic_str} | NDCG: {_ndcg_str} | 健康分: {_hs:.0f}")

    # 各模型收益对比
    if len(_seqs) >= 2:
        _cmp = []
        for _sk, _sv in _seqs.items():
            _sr = _sv.get("strategy_return_pct")
            if _sr is not None:
                _tag = " ← 主" if _sk == _pri_key else ""
                _cmp.append(f"{_short_name(_sk)} {_sr:+.2f}%{_tag}")
        if _cmp:
            block7_lines.append(f"模型对比: {' | '.join(_cmp)}")

    # 健康分明细
    _raw_health = data.get("health_scores", {})
    if isinstance(_raw_health, dict) and _raw_health:
        _h_details = []
        for _sk, _sh in _raw_health.items():
            if isinstance(_sh, dict):
                _hs = _sh.get("score", 0)
                _d = _sh.get("details", {})
                _wr_h = _d.get("wr", "?")
                _ar_h = _d.get("avgret", "?")
                _h_details.append(f"{_short_name(_sk)}: {_hs:.0f}分(近3日{_ar_h} / 10日胜率{_wr_h})")
        if _h_details:
            block7_lines.append("健康分: " + " | ".join(_h_details))

    block7 = "\n".join(block7_lines)

    # ---------- 区块8: 预测信号对比 ----------
    _pred_today = data.get("pred_signals_today", [])
    _pred_today_date = data.get("pred_signals_today_date", "")
    _pred_rb = data.get("pred_signals_rb", [])
    _pred_rb_date = data.get("pred_signals_rb_date", "")
    _top_k = data.get("top_k", 3)
    block8_lines = []
    if _pred_today and _pred_rb and _pred_today_date and _pred_rb_date:
        if _pred_today_date != _pred_rb_date:
            block8_lines.append(f"调仓日({_pred_rb_date}) vs 当前({_pred_today_date}) Top{_top_k}预测信号对比：")
            _today_scores = {p["stock_id"]: p["score"] for p in _pred_today[:_top_k]}
            _today_ranks = {p["stock_id"]: p["rank"] for p in _pred_today}
            _rb_scores = {p["stock_id"]: p["score"] for p in _pred_rb}
            _rb_ranks = {p["stock_id"]: p["rank"] for p in _pred_rb}
            _all_codes = list(_today_scores.keys()) + [p["stock_id"] for p in _pred_rb if p["stock_id"] not in _today_scores]
            _all_codes = _all_codes[:_top_k * 2]
            for _code in _all_codes:
                _ts = _today_scores.get(_code)
                _tr = _today_ranks.get(_code)
                _rs = _rb_scores.get(_code)
                _rr = _rb_ranks.get(_code)
                if _ts is not None and _rs is not None:
                    _chg = _ts - _rs
                    block8_lines.append(f"  {_code}: 得分{_rs:+.4f}→{_ts:+.4f}({_chg:+.4f}) | 排名{_rr}→{_tr}")
                elif _ts is not None:
                    block8_lines.append(f"  {_code}: 得分{_ts:+.4f} | 排名{_tr} ← 新增入榜")
                elif _rs is not None:
                    block8_lines.append(f"  {_code}: 得分{_rs:+.4f} | 排名{_rr} → 已出榜")
    block8 = "\n".join(block8_lines) if block8_lines else ""

    # ---------- 区块0: 系统背景 ----------
    ws = data.get("weight_strategy", "")
    tk = data.get("top_k", 3)
    rb = data.get("rebalance_days", 5)
    tm = data.get("trade_mode", "open")
    pp = data.get("position_pct", 0.95)
    tm_desc = "次日开盘" if tm == "open" else "当日收盘"
    ws_desc = _STRATEGY_DESC.get(ws, ws)
    _model_types = set()
    for _sk in _seqs:
        if _sk in ("average", "voting", "juejin"):
            continue
        _parts = _sk.split("_")
        if len(_parts) >= 2:
            _model_types.add(_parts[1])
    _model_archs = []
    for _mt in sorted(_model_types):
        _desc = _MODEL_ARCH.get(_mt, _mt)
        _model_archs.append(f"  {_desc}")
    bg_parts = ["【系统背景】"]
    bg_parts.append("▎交易策略")
    bg_parts.append(f"  选股：模型对74只ETF输出排序分 → 取前{tk}名")
    bg_parts.append(f"  加权：{ws} — {ws_desc}")
    bg_parts.append(f"  调仓：每{rb}交易日，{tm_desc}交易")
    bg_parts.append(f"  仓位：{pp:.0%}，费率+滑点合计约0.13%")
    if _model_archs:
        bg_parts.append("▎模型架构（预测目标：次日开盘→5日后开盘涨幅，排序学习训练）")
        bg_parts.extend(_model_archs)
    bg_parts.append("▎ETF池")
    bg_parts.append("  74只主流ETF（宽基指数/行业主题/跨境QDII/商品期货）")
    bg_parts.append("")
    bg_parts.append("⚠️ 持仓明细中的「5日排名」是全池74只ETF按近5日涨跌幅的排名（1最强/74最弱），并非模型预测评分排名。")
    bg_parts.append("  模型评分排名决定选股（取前top_k名建仓），两者含义不同，请勿混淆。")
    bg_block = "\n".join(bg_parts)

    # ---------- 拼装 ----------
    _blocks = [
        bg_block,
        f"【组合概览】\n{block1}",
        f"【持仓明细】\n{block2}",
        f"【市场状态】\n{block3}",
        f"【ETF排行榜】\n{block4}",
        f"【近期窗口】\n{block5}",
        f"【交易概况】\n{block6}",
        f"【模型表现】\n{block7}",
    ]
    if block8:
        _blocks.append(f"【预测信号对比】\n{block8}")
    return (
        f"你是一个A股ETF量化分析助手。根据以下日报数据和系统背景详细分析：\n\n"
        + "\n\n".join(_blocks) + "\n\n"
        + f"请从以下四个方面综合分析（**必须引用模型对比数据**）：\n"
        f"1)仓位与持仓合理性（集中度、个股盈亏、持仓在全池中的排名强弱）\n"
        f"2)近期市场风格与策略适应性（结合市场宽度、分市场表现、ETF排行榜热点）\n"
        f"3)风险暴露（回撤、波动、VaR、排名靠后的持仓、板块集中度、模型健康分）\n"
        f"4)调仓前瞻（下次调仓日、排行榜中值得关注的板块轮动信号、模型近期趋势、不同模型收益分化）"
    )


def _call_deepseek(prompt: str, api_key: str) -> str:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    resp = requests.post(
        LLM_CONFIG["api_url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        verify=False,
        json={
            "model": LLM_CONFIG["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.3,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _format_section(text: str) -> str:
    import mistune
    html = mistune.markdown(text)
    return (
        f'<h3>AI 市场分析</h3>'
        f'<div style="background:#f0f7ff;border:1px solid #cce5ff;'
        f'border-radius:6px;padding:12px;font-size:12px;line-height:1.6;color:#333;">'
        f'{html}</div>'
    )


def _placeholder_section(msg="待接入大模型分析...") -> str:
    return (
        f'<h3>AI 市场分析</h3>'
        f'<div style="background:#f9f9f9;border:1px solid #eee;'
        f'border-radius:6px;padding:12px;font-size:12px;color:#999;text-align:center;">'
        f'{msg}</div>'
    )
