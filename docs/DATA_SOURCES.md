# Data sources and legal operating modes

Provider facts were verified from primary provider/exchange documentation on 20 September 2026. Users must re-check terms before redistribution or commercial use.

## Selected historical source: Yahoo Finance via `yfinance`

Used for reproducible personal/academic **end-of-day research ingestion** because it covers NSE `.NS`, BSE indices, corporate actions and multiple years without a key. Yahoo documents exchange suffixes and delays in its [market coverage table](https://help.yahoo.com/kb/SLN2310.html). `yfinance` is an unofficial client, so this adapter is explicitly named `yahoo_research`, has retries and validation, and is never presented as an exchange-authorized real-time feed.

- Authentication: none.
- Coverage: daily OHLCV for the configured universe; availability can vary by symbol.
- Freshness: treat as end-of-day/delayed and provider-dependent.
- Limits: no contractual free API SLA or published stable request quota; use incremental updates and local persistence.
- Licensing: personal/research use only under the source terms; do not redistribute raw feeds without permission.
- Fallback: replace the `MarketDataProvider` adapter with a licensed vendor export/API. Twelve Data is the implemented credentialed quote option; broker APIs may be added under their agreements.

## Optional quotes: Twelve Data

Twelve Data documents API-key authentication, endpoint credit weights, and plan-dependent access in its [API documentation](https://twelvedata.com/docs). Its [credit policy](https://support.twelvedata.com/en/articles/5615854-credits) states the Basic daily quota and 429 behavior; WebSocket access is plan-dependent per its [service introduction](https://support.twelvedata.com/en/articles/5609168-introduction-to-twelve-data).

- Environment: `TWELVE_DATA_API_KEY`, `QUOTE_PROVIDER=twelve_data`, and the correct `QUOTE_DELAY_MINUTES` for the subscribed entitlement.
- Free-tier limitations: symbol/exchange entitlement and daily credits must be checked for the current plan; WebSockets may require a paid plan.
- The platform does not claim `LIVE` merely because a request succeeded. Status depends on configured entitlement, observation age and Indian market hours.

## Optional news: NewsData.io

NewsData.io is selected for a practical India/business API with published quotas. Its provider documentation states that free access is delayed and excludes full content; see [pricing/limits](https://newsdata.io/blog/pricing-plan-in-newsdata-io/) and [credit consumption](https://newsdata.io/blog/newsdata-credit-consumption/).

- Environment: `NEWSDATA_API_KEY`, `NEWS_PROVIDER=newsdata`.
- Free plan: provider currently documents 200 credits/day, 10 articles/credit and a 12-hour delay; re-check before use.
- Stored fields: headline, description where supplied, link, source, publication timestamp and derived associations/sentiment. Full copyrighted article bodies are not copied.
- Fallback: licensed GDELT/MediaStack/provider export adapter, subject to terms and validation.

## Official NSE data and why it is not the free default

NSE states that real-time, snapshot, delayed, end-of-day and historical market data are governed by subscription/usage agreements in its [Data Sharing & Usage Policy](https://www.nseindia.com/static/market-data/nse-data-policy). Its [market-data product page](https://www.nseindia.com/static/market-data/real-time-data-subscription) describes licensed real-time levels and 15-minute delayed snapshot products. The project therefore does not scrape NSE pages or imply exchange-grade live access.

## Market calendar and freshness

Normal cash-market display logic uses Asia/Kolkata weekdays and an approximate 09:15–15:30 IST window. Exchange holidays and special sessions require an exchange calendar update; outside the session the UI says `MARKET CLOSED`. `DELAYED`, `STALE`, and `UNAVAILABLE` are preferred to a misleading `LIVE` label.

