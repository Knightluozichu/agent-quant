# 免费 A 股/ETF 分钟数据源调研（2026-11-02）

- 目标：2020–2026、5 分钟（或 1 分钟）、8 个 ETF/LOF、前复权、免费
- 结论：**没有已验证可用的“免费 + 6 年 + ETF/LOF + 稳定”组合**
- 最值得试：BaoStock、mootdx/TDX

## 1. 候选源

| 源 | 分钟频率 | 历史深度 | 免费 | ETF/LOF | 备注 |
| --- | --- | --- | --- | --- | --- |
| **BaoStock** | 5/15/30/60m | 较长（按代码/日期拉） | ✅ | ⚠️ 需实测；参考文档只明确“股票支持” | 官方/社区称免费稳定；有 QPS 限制 |
| **AkShare（新浪分钟）** | 1/5/15/30/60m | **最近约 2 个月** | ✅ | ✅ 实测 8 个 ETF/LOF 都有 | 无法覆盖 2020–2026 |
| **AkShare（东财分钟）** | 1/5/15/30/60m | 最近几天 | ✅ | ✅ | 本机连接被拒；且深度不够 |
| **mootdx / pytdx（通达信）** | 1/5/15/30/60m | 服务器相关 | ✅ | ⚠️ 需实测 | 本机返回空；非官方，稳定性/深度看服务器 |
| **Tushare Pro** | 分钟线 | 有 | ❌ 需积分/权限 | ✅ | 免费额度不够 6 年分钟 |
| **AlphaFeed** | 1/5/15/30/60m | 套餐相关 | 部分免费额度 | ✅ | 长历史需付费套餐 |
| **Yahoo Finance** | 1/5m | 最近约 60 天 | ✅ | ❌ A 股 ETF 支持有限 | 不够 |
| **QMT/PTrade** | 分钟 | 券商本地/托管 | 需权限 | ✅ | 官方通道，但需券商开通 |
| **Wind/Choice/RQData/JQData** | 分钟 | 长历史 | ❌ 付费 | ✅ | 最可靠，但要钱 |

## 2. 本地实测

| 源 | 结果 |
| --- | --- |
| BaoStock `login()` | ❌ `10001011 黑名单用户`（当前出口 IP 被拉黑） |
| BaoStock 5m 查询 | ❌ 未登录，无法验证 ETF |
| AkShare 新浪 5m | ✅ 8 个 ETF/LOF 都有，但只有 2026-08-27→2026-11-02 |
| AkShare 东财分钟 | ❌ 连接被拒 |
| mootdx `bars()` | ❌ 返回空 DataFrame（TDX 服务器/网络问题） |
| 腾讯分钟接口 | ❌ 当前 DNS 不通 |

## 3. 关键限制

1. **BaoStock 官方参考只明确“股票支持分钟”，指数无分钟**；ETF 是否支持需要实测。
2. 即使 BaoStock 支持 ETF，当前 IP 被黑名单，需要换网络/联系官方。
3. mootdx 是通达信协议，非官方；服务器可用性、历史深度不稳定。
4. AkShare 免费分钟只有最近约 2 个月，不能做 2020–2026 全量。
5. 分钟线通常**不复权**；跨除权日要自己复权，否则价格跳空。

## 4. 建议的下一步

### 方案 A：换网络试 BaoStock（首选免费）

在非黑名单网络/服务器上：

```python
import baostock as bs, pandas as pd
bs.login()
rs = bs.query_history_k_data_plus(
    "sh.510300",
    "date,time,open,high,low,close,volume",
    start_date="2020-01-02", end_date="2026-10-30",
    frequency="5", adjustflag="2",
)
rows=[]
while rs.next(): rows.append(rs.get_row_data())
df=pd.DataFrame(rows, columns=rs.fields)
```

先只测 `sh.510300` 2020 年 1 月：
- 有数据 → 再测 501018/161226/159985/511220/511880/518880/513100/159915；
- 无数据 → BaoStock 不支持该 ETF，换源。

### 方案 B：配置 mootdx/TDX

```python
from mootdx.quotes import Quotes
client = Quotes.factory(market='std')
df = client.bars(symbol='510300', frequency=0, offset=800)
```

`frequency=0` 通常对应 5 分钟；但服务器/版本差异大，需要先测一只。

### 方案 C：付费/券商

- QMT/PTrade：券商开通后，分钟数据最稳；
- RQData/JQData/Wind：付费，但可直接拿 6 年 5 分钟 + 复权。

## 5. 结论

- **免费全量 6 年 ETF 分钟数据：目前没有已验证可用源。**
- 最优先试 **BaoStock（换 IP）**；其次 **mootdx/TDX**；都不行就付费/券商。
- 在拿到数据前，严格 14:50 全量回测只能停在“2 个月试点”。

## 参考

- BaoStock API 参考（5/15/30/60 分钟、分钟仅股票支持的说明）：
  https://raw.githubusercontent.com/lzwme/finance-quant-skills/main/skills/baostock/references/api.md
- 腾讯云社区：A 股历史分钟线来源对照（AkShare/BaoStock/AlphaFeed）：
  https://cloud.tencent.cn/developer/article/2749069
- mootdx：
  https://pypi.org/project/mootdx/0.6.8/
- Tushare 积分与频次权限：
  https://tushare.pro/document/1?doc_id=290
