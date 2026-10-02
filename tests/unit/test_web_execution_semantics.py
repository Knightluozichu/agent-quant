"""HTML-only execution contracts; run inline JS with a fake DOM and no network."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parents[2] / "scripts/web/index.html"


def run_js(scenario: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for isolated HTML contracts")
    html = HTML.read_text()
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    script = script.split("// 启动: 检查 cookie")[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');
const html = INPUT_HTML;
const els = {};
for (const tag of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
  els[tag[1]] = {value: (tag[0].match(/\bvalue="([^"]*)"/)||[])[1]||'',
    checked: /\schecked(?:\s|>)/.test(tag[0]), disabled: false, innerHTML:'', textContent:'',
    className:'', classList:{add(){}, remove(){}}, max:''};
}
const context = {assert, crypto:webcrypto, Date, Number, String, Math, JSON, console,
  document:{getElementById(id){assert.ok(els[id], 'Missing element: '+id); return els[id];}},
  window:{}, setTimeout(){}, confirm(){return true;},
  fetch(){throw Error('No network permitted');}};
vm.createContext(context);
vm.runInContext(SOURCE + `
const el = id => document.getElementById(id);
const requests = [];
let failRequest = false;
let refreshes = 0;
api = async (path, opts) => { requests.push({path, payload:JSON.parse(opts.body)});
  if(failRequest) throw Error('fixture request failed'); return {}; };
loadAll = async () => { refreshes++; };
function pending(){ return {status:'pending', order_id:'fixture-order', date:'2026-09-16',
  confirmable:true, expected_state_version:1,
  sell:{code:'159985', name:'测试豆粕', shares:400, price:2},
  buy:{code:'518880', name:'测试黄金', shares:100, price:8}}; }
function manual(){
  el('mAction').value='buy'; el('mCode').value='518880';
  el('mShares').value='100'; el('mPrice').value='2.5'; el('mFees').value='0';
  el('mDate').value='2026-01-01'; el('mEvidence').checked=true;
}
function confirmFields(){
  for(const side of ['sell','buy']){
    el(side+'On').checked=true; el(side+'Shares').value='100';
    el(side+'Price').value='2.5'; el(side+'Fees').value='0';
  }
  el('pendingEvidence').checked=true;
}
(async()=>{ SCENARIO })()
`, context).catch(e => {console.error(e); process.exitCode=1;});
"""
    harness = harness.replace("INPUT_HTML", json.dumps(html)).replace("SOURCE", json.dumps(script))
    harness = harness.replace("SCENARIO", scenario)
    result = subprocess.run(  # noqa: S603 — local HTML/test code only, no network
        [node, "-e", harness], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pending_never_prefills_execution_or_checks_legs():
    run_js("""
renderPending(pending());
for(const side of ['sell','buy']){
  assert.equal(el(side+'On').checked, false);
  for(const field of ['Shares','Price','Fees']) assert.equal(el(side+field).value,'');
}
assert.equal(el('pendingEvidence').checked,false);
assert.match(el('pendingReason').textContent, /只.*事实|不.*下单/);
""")


@pytest.mark.parametrize(
    "mutation",
    [
        "el('buyOn').checked=false;",
        "el('pendingEvidence').checked=false;",
        "el('sellFees').value='';",
        "el('buyFees').value='-1';",
        "el('buyShares').value='1.5';",
        "el('sellPrice').value='Infinity';",
        "curPending.confirmable=false;",
    ],
)
def test_confirm_rejects_incomplete_or_invalid_facts(mutation):
    run_js(
        "renderPending(pending()); confirmFields();"
        + mutation
        + """
await doConfirm(); assert.equal(requests.length,0);
"""
    )


def test_confirm_fees_retry_and_no_partial_leg_submission():
    run_js("""
renderPending(pending()); confirmFields(); failRequest=true;
await doConfirm(); const key=requests[0].payload.idempotency_key;
assert.ok(key); assert.equal(requests[0].payload.sell.fees,0);
assert.equal(requests[0].payload.buy.fees,0);
renderPending(pending());
assert.equal(el('buyShares').value,'100');
await doConfirm(); assert.equal(requests[1].payload.idempotency_key,key);
failRequest=false; await doConfirm();
assert.equal(el('buyShares').value,''); assert.equal(el('pendingEvidence').checked,false);
assert.equal(el('confirmButton').disabled,true);
""")


@pytest.mark.parametrize(
    "mutation",
    [
        "el('mFees').value='';",
        "el('mFees').value='-0.01';",
        "el('mDate').value='';",
        "el('mDate').value='2999-01-01';",
        "el('mDate').value='2026-02-30';",
        "el('mEvidence').checked=false;",
        "el('mShares').value='0';",
        "el('mPrice').value='NaN';",
    ],
)
def test_manual_requires_explicit_valid_facts(mutation):
    run_js("manual();" + mutation + "await doManual(); assert.equal(requests.length,0);")


def test_manual_retry_key_and_historical_opt_in_reset():
    run_js("""
manual(); assert.equal(el('mHistorical').checked,false);
failRequest=true; await doManual();
const first=requests[0].payload; assert.equal(first.fees,0);
assert.equal(first.historical_fill,false); assert.equal(first.evidence_confirmed,true);
assert.ok(first.idempotency_key);
await doManual(); assert.equal(requests[1].payload.idempotency_key,first.idempotency_key);
assert.equal(el('mShares').value,'100');
failRequest=false; await doManual();
assert.equal(el('mShares').value,''); assert.equal(el('mFees').value,'');
assert.equal(el('mDate').value,''); assert.equal(el('mEvidence').checked,false);
manual(); el('mHistorical').checked=true; await doManual();
assert.equal(requests[3].payload.historical_fill,true);
assert.notEqual(requests[3].payload.idempotency_key,first.idempotency_key);
assert.equal(el('mHistorical').checked,false);
""")


def test_historical_catalogue_only_opt_in_and_no_default_security():
    run_js("""
ETFS=[{code:'518880',name:'测试黄金'}];
RECORDABLE_ETFS=[...ETFS,{code:'501018',name:'测试原油'},{code:'511880',name:'测试防御'}];
el('mAction').value='buy'; fillEtfSelects();
assert.doesNotMatch(el('mCode').innerHTML,/501018/); assert.equal(el('mCode').value,'');
el('mHistorical').checked=true; fillEtfSelects();
assert.match(el('mCode').innerHTML,/501018/);
assert.doesNotMatch(el('buyCode').innerHTML,/501018/);
assert.equal(el('mCode').value,'');
""")


def test_account_notice_blocking_and_amount_formula_contract():
    run_js("""
renderAccountProfile({broker:'广发证券',app:'易淘金',risk_grade:'C5',risk_label:'进取型',
  source:'user_reported',confirmed_on:'2026-09-16'});
assert.match(el('accountProfile').textContent,/C5.*进取型/);
assert.match(el('accountProfile').textContent,/用户自报/);
renderSignal({actionable:false,board:[{code:'518880',name:'测试黄金',score:1,eligible:true}],
  target:{code:'518880',name:'测试黄金'}});
assert.match(el('executionBlock').textContent,/不可.*执行|不.*买入/);
assert.match(el('board').innerHTML,/准入未核实/);
el('scenarioAmount').value='10000'; el('scenarioDrop').value='10'; renderScenario();
assert.match(el('scenarioResult').textContent,/1,000.00/);
assert.match(el('scenarioResult').textContent,/不.*风险预算/);
""")


def test_inflight_manual_double_click_sends_only_once():
    run_js("""
manual(); let release;
api=async(path, opts)=>{
  requests.push({path,payload:JSON.parse(opts.body)});
  return new Promise(resolve=>{release=resolve;});
};
const first=doManual(); await doManual();
assert.equal(requests.length,1); assert.equal(el('manualButton').disabled,true);
release({}); await first; assert.equal(el('manualButton').disabled,false);
""")


def test_success_followed_by_refresh_failure_does_not_retain_completed_form():
    run_js("""
manual(); loadAll=async()=>{throw Error('fixture refresh failed');};
await doManual(); assert.equal(requests.length,1);
assert.equal(el('mShares').value,''); assert.equal(el('mEvidence').checked,false);
assert.match(el('toast').textContent,/已登记.*刷新失败/);
await doManual(); assert.equal(requests.length,1);
""")


def test_new_pending_clears_facts_but_failed_request_does_not():
    run_js("""
renderPending(pending()); confirmFields(); failRequest=true; await doConfirm();
const oldKey=requests[0].payload.idempotency_key;
renderPending({...pending(),order_id:'fixture-next'});
assert.equal(el('buyShares').value,''); assert.equal(el('pendingEvidence').checked,false);
confirmFields(); await doConfirm();
assert.notEqual(requests[1].payload.idempotency_key,oldKey);
""")


def test_historical_buy_does_not_become_regular_buy_when_unchecked():
    run_js("""
ETFS=[{code:'518880',name:'测试黄金'}]; RECORDABLE_ETFS=[{code:'501018',name:'测试原油'},...ETFS];
el('mAction').value='buy'; el('mHistorical').checked=true; fillEtfSelects();
el('mCode').value='501018'; el('mHistorical').checked=false; fillEtfSelects();
assert.equal(el('mCode').value,''); assert.doesNotMatch(el('mCode').innerHTML,/501018/);
""")


def test_history_shows_actual_fees_and_unknown_status():
    run_js("""
const base={date:'2026-09-16',action:'buy',code:'518880',name:'黄金',
  shares:100,price:5,amount:502,gross_amount:500};
renderHistory([{...base,fees:2,fee_status:'reported'}]);
assert.ok(el('historyBox').innerHTML.includes('实际费用 ¥2.00'));
renderHistory([{...base,fees:null,fee_status:'unreported'}]);
assert.ok(el('historyBox').innerHTML.includes('费用未核实'));
""")


def test_unreconciled_history_never_claims_real_equity_curve():
    run_js("""
renderEquity({curve:[{date:'2026-09-15',value:97036.51}],initial_capital:100000,
  needs_reconciliation:true,cash_gap:647.94,shares_gap:0});
assert.ok(el('chart').innerHTML.includes('历史账本待对账'));
assert.ok(el('chart').innerHTML.includes('647.94'));
assert.ok(!el('chart').innerHTML.includes('<svg'));
""")
