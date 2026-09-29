"""A self-contained, read-only report for local factor evaluation artifacts."""

import json
from pathlib import Path


def write_report(path: Path, manifest: dict, summary: list, daily: list) -> None:
    payload = json.dumps({"manifest": manifest, "summary": summary, "daily": daily},
                         ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    path.write_text(_HTML.replace("__REPORT_DATA__", payload), encoding="utf-8")


_HTML = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>因子评价报告</title>
<style>
body{font:15px/1.65 system-ui,sans-serif;background:#f4f6fa;color:#172033;margin:0;padding:28px}
main{max-width:1400px;margin:auto}h1{font-size:28px;margin:0}h2{font-size:19px}p{color:#526078}
.card{background:white;border:1px solid #dfe5ed;border-radius:12px;padding:20px;margin:18px 0}
select{font:inherit;padding:7px;max-width:100%;margin-right:18px}label{display:inline-block;margin:5px 0}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e6eaf0;white-space:nowrap}
th{background:#eef2f8}td:first-child{white-space:normal;min-width:260px;max-width:380px;overflow-wrap:anywhere}small{display:block;color:#68748a}
a{color:#175bb5}svg{width:100%;height:240px}.muted{color:#68748a}.warn{background:#fff7e3;border-color:#ead59c}
</style>
<main><h1>因子评价报告</h1><p id="overview"></p>
<div class="card warn">这是历史数据上的描述性评价，不自动筛选因子、不产生交易指令。留出区间未参与自动选因子，但结果一旦用于反复调参，就不再是独立验证。未核实行情复权口径，收益未扣交易成本。</div>
<div class="card"><h2>收益与样本口径</h2><p id="convention"></p>
<p>周期按本次行情中的交易日序列计数。缺失价格不填充，非正价格和零成交量的买卖端点剔除；窗口预热、无效值、常数截面和股票数量不足均记录在明细中。训练段收益结束时间必须早于留出段起点。</p>
<p>Rank IC 为每日截面的 Spearman 相关；IR 为每日 Rank IC 均值 / 样本标准差，未年化。最高组减最低组是持有期累计收益的平均差，不是策略净值或年化收益。不同因子有效样本可能不同，比较时请同时查看覆盖率及有效天数。</p></div>
<div class="card"><label>区间 <select id="segment"><option value="holdout">后段留出</option><option value="train">前段训练</option><option value="all">全部（描述性）</option></select></label>
<label>持有周期 <select id="period"></select></label>
<div class="scroll"><table><thead><tr><th>因子 / 公式</th><th>状态</th><th>覆盖率</th><th>IC 天数</th><th>平均 Rank IC</th><th>Rank IC IR</th><th>最高组 − 最低组</th><th>分组覆盖率</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<div class="card"><h2>每日 Rank IC</h2><select id="factor"></select><p id="chartNote"></p><svg id="chart" viewBox="0 0 1000 240" role="img" aria-label="每日 Rank IC 折线图"></svg></div>
<div class="card"><h2>数据文件</h2><a href="summary.csv">评价汇总 CSV</a> · <a href="daily_rank_ic.csv">每日 Rank IC</a> · <a href="quantile_returns.csv">分组收益</a> · <a href="turnover.csv">分组换手率</a> · <a href="manifest.json">配置和数据口径</a>
<p>完整因子值、原行情和带买卖日期的收益标签分别保存在 factors.parquet、prices.parquet、labels.parquet。状态为“跳过”的结果不能理解为 IC = 0。</p></div></main>
<script type="application/json" id="data">__REPORT_DATA__</script>
<script>
const data=JSON.parse(document.querySelector('#data').textContent),m=data.manifest;
const byId=id=>document.getElementById(id),num=x=>x==null?'—':Number(x).toFixed(4),pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
byId('overview').textContent=`${m.symbols} 只股票 · ${m.sessions} 个交易日 · ${m.alphas.length} 个因子 · 留出起点 ${m.holdout_start.slice(0,10)}`;
byId('convention').textContent=`T 日收盘后得到因子，T+${m.config.entry_offset} 日收盘进入，T+${m.config.entry_offset}+持有周期日收盘退出。收益 = 退出收盘价 / 进入收盘价 − 1。`;
for(const p of m.config.periods){const o=new Option(p+' 个交易周期',p);byId('period').add(o)}
for(const a of m.alphas){byId('factor').add(new Option(a.formula||a.name,a.name))}
const reasons={no_valid_pairs:'无有效收益样本',insufficient_or_constant_cross_sections:'截面恒定或样本不足'};
function draw(){
 const segment=byId('segment').value,period=Number(byId('period').value),tbody=byId('rows');tbody.replaceChildren();
 for(const r of data.summary.filter(r=>r.segment===segment&&r.period===period)){
  const tr=document.createElement('tr'),name=document.createElement('td');name.textContent=r.formula||r.factor;
  const detail=document.createElement('small');detail.textContent=r.factor;name.append(detail);tr.append(name);
  const status=r.status==='ok'?'已计算':('跳过：'+(reasons[r.reason]||r.reason));
  for(const v of [status,pct(r.factor_coverage),r.ic_days,num(r.rank_ic),num(r.rank_ic_ir),pct(r.top_minus_bottom),pct(r.quantile_coverage)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}
  tbody.append(tr);
 }
 const points=data.daily.filter(r=>r.segment===segment&&r.period===period&&r.factor===byId('factor').value);
 const svg=byId('chart');svg.replaceChildren();
 function element(tag,attributes,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attributes))e.setAttribute(k,v);if(text!=null)e.textContent=text;svg.append(e);return e}
 for(const v of [-1,0,1]){const y=110-v*90;element('line',{x1:45,x2:975,y1:y,y2:y,stroke:'#dfe5ed'});element('text',{x:8,y:y+5,fill:'#68748a','font-size':13},v)}
 byId('chartNote').textContent=points.length?`有效 ${points.length} 天；仅连接有有效 IC 的日期，样本很短时均值可能不稳定。`:'当前因子、区间和周期没有可计算的 IC。';
 if(points.length){
  const coordinates=points.map((p,i)=>`${45+i*930/Math.max(1,points.length-1)},${110-p.rank_ic*90}`).join(' ');
  element('polyline',{points:coordinates,fill:'none',stroke:'#1d66b8','stroke-width':2});
  points.forEach((p,i)=>{const circle=element('circle',{cx:45+i*930/Math.max(1,points.length-1),cy:110-p.rank_ic*90,r:3,fill:'#1d66b8'});const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=`${p.date.slice(0,10)}: ${num(p.rank_ic)} (${p.assets}只)`;circle.append(title)});
  element('text',{x:45,y:230,fill:'#68748a','font-size':13},points[0].date.slice(0,10));element('text',{x:975,y:230,'text-anchor':'end',fill:'#68748a','font-size':13},points.at(-1).date.slice(0,10));
 }
}
for(const id of ['segment','period','factor'])byId(id).addEventListener('change',draw);draw();
</script></html>'''
