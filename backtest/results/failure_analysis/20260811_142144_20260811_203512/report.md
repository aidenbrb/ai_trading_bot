# stock_only v1 failure analysis

**POST-HOC EXPLORATORY ANALYSIS.** This report examines the same dataset
that already produced the qualification verdict for `stock_trend_momentum_v1`
(run `20260811_142144`, rejected). Any pattern below - regime clustering, MFE/MAE
shape, cost drag, anything else - is a **hypothesis** for a separately
versioned `stock_trend_momentum_v2`, not a confirmed finding. v2's actual
rules must be predefined before looking at new data and evaluated through
walk-forward or forward-paper evidence. Nothing here justifies a rule by
re-pointing back at this same 2022-2026 dataset.

Automatic stock entries remain disabled regardless of anything in this
report. See `manifest.json` for exact source-file and cache hashes.

## 1. Performance by period, regime, symbol, sector

Realized (exit-time) net P&L by quarter reconciles to summary.json (total $-9,144.48).

| period | count | net_pnl | expectancy | win_rate |
| --- | --- | --- | --- | --- |
| 2022Q1 | 105 | -5,581.0996 | -53.1533 | 0.3429 |
| 2022Q2 | 76 | -3,957.4435 | -52.0716 | 0.2500 |
| 2022Q3 | 83 | 5,149.1057 | 62.0374 | 0.3976 |
| 2022Q4 | 91 | -3,934.2677 | -43.2337 | 0.3077 |
| 2023Q1 | 94 | 401.0668 | 4.2667 | 0.3936 |
| 2023Q2 | 85 | -1,231.1424 | -14.4840 | 0.4353 |
| 2023Q3 | 34 | -406.4291 | -11.9538 | 0.3235 |
| 2023Q4 | 68 | 945.1410 | 13.8991 | 0.3529 |
| 2024Q1 | 50 | -8.0830 | -0.1617 | 0.4000 |
| 2024Q2 | 47 | 16.7793 | 0.3570 | 0.5106 |
| 2024Q3 | 45 | -4.0262 | -0.0895 | 0.3333 |
| 2024Q4 | 40 | -6.2406 | -0.1560 | 0.2500 |
| 2025Q1 | 34 | -8.9999 | -0.2647 | 0.3235 |
| 2025Q2 | 75 | -107.3558 | -1.4314 | 0.2400 |
| 2025Q3 | 79 | 4.4854 | 0.0568 | 0.3797 |
| 2025Q4 | 85 | -33.6148 | -0.3955 | 0.3765 |
| 2026Q1 | 63 | -42.9143 | -0.6812 | 0.3333 |
| 2026Q2 | 48 | -346.6030 | -7.2209 | 0.2917 |
| 2026Q3 | 25 | 7.1578 | 0.2863 | 0.4400 |

**Entry-cohort (decision-time) view - diagnostic, not authoritative:**

| period | count | net_pnl | expectancy | win_rate |
| --- | --- | --- | --- | --- |
| 2022Q1 | 108 | -5,972.9931 | -55.3055 | 0.3333 |
| 2022Q2 | 74 | -3,243.6025 | -43.8325 | 0.2703 |
| 2022Q3 | 82 | 4,827.1582 | 58.8678 | 0.3902 |
| 2022Q4 | 93 | -3,573.5743 | -38.4255 | 0.3226 |
| 2023Q1 | 96 | 585.9144 | 6.1033 | 0.3958 |
| 2023Q2 | 84 | -1,858.8208 | -22.1288 | 0.4048 |
| 2023Q3 | 32 | -359.3891 | -11.2309 | 0.3438 |
| 2023Q4 | 69 | 945.8139 | 13.7074 | 0.3623 |
| 2024Q1 | 51 | -9.3646 | -0.1836 | 0.3922 |
| 2024Q2 | 46 | 18.0609 | 0.3926 | 0.5217 |
| 2024Q3 | 46 | -1.7040 | -0.0370 | 0.3478 |
| 2024Q4 | 39 | -8.5628 | -0.2196 | 0.2308 |
| 2025Q1 | 35 | -9.2733 | -0.2650 | 0.3143 |
| 2025Q2 | 74 | -92.0136 | -1.2434 | 0.2432 |
| 2025Q3 | 78 | -11.2563 | -0.1443 | 0.3718 |
| 2025Q4 | 87 | -33.6971 | -0.3873 | 0.3793 |
| 2026Q1 | 62 | -41.8483 | -0.6750 | 0.3387 |
| 2026Q2 | 46 | -312.4893 | -6.7932 | 0.2826 |
| 2026Q3 | 25 | 7.1578 | 0.2863 | 0.4400 |

**Market regime (SPY, decision-time):**

| regime | count | net_pnl | win_rate |
| --- | --- | --- | --- |
| DOWNTREND | 104 | -4,472.9518 | 0.3077 |
| SIDEWAYS | 785 | -2,318.0499 | 0.3580 |
| UPTREND | 338 | -2,353.4821 | 0.3491 |

**By symbol/sector:**

| symbol | count | net_pnl | expectancy | win_rate | sector |
| --- | --- | --- | --- | --- | --- |
| EOG | 8 | -1,592.7578 | -199.0947 | 0.1250 | Energy |
| AMD | 18 | -1,419.7768 | -78.8765 | 0.2778 | Technology |
| AMZN | 11 | -1,362.7233 | -123.8839 | 0.1818 | Consumer Discretionary |
| OXY | 23 | -1,307.1792 | -56.8339 | 0.2609 | Energy |
| SLB | 15 | -1,287.4436 | -85.8296 | 0.0667 | Energy |
| NFLX | 16 | -1,255.9647 | -78.4978 | 0.3750 | Communication |
| ABBV | 11 | -1,222.8230 | -111.1657 | 0.1818 | Healthcare |
| MRK | 12 | -1,026.3537 | -85.5295 | 0.2500 | Healthcare |
| AMAT | 14 | -915.5084 | -65.3935 | 0.3571 | Technology |
| CRM | 12 | -914.9433 | -76.2453 | 0.2500 | Technology |
| CRWD | 7 | -859.4161 | -122.7737 | 0.1429 | Technology |
| MPC | 6 | -771.7537 | -128.6256 | 0.1667 | Energy |
| KHC | 20 | -716.8902 | -35.8445 | 0.4000 | Consumer Staples |
| GILD | 9 | -693.5121 | -77.0569 | 0.4444 | Healthcare |
| CMCSA | 9 | -651.3053 | -72.3673 | 0.3333 | Communication |
| XOM | 11 | -603.4864 | -54.8624 | 0.2727 | Energy |
| IBB | 5 | -599.0228 | -119.8046 | 0.2000 | ETFs |
| VZ | 17 | -597.9744 | -35.1750 | 0.1765 | Communication |
| VRTX | 3 | -589.6387 | -196.5462 | 0.3333 | Healthcare |
| CVS | 12 | -554.8891 | -46.2408 | 0.5000 | Healthcare |
| LIN | 3 | -537.3834 | -179.1278 | 0.0000 | Materials/Real Estate |
| INTC | 19 | -524.8451 | -27.6234 | 0.3158 | Technology |
| JPM | 4 | -504.4374 | -126.1093 | 0.0000 | Financials |
| KO | 17 | -491.4031 | -28.9061 | 0.2941 | Consumer Staples |
| FDX | 6 | -399.0874 | -66.5146 | 0.3333 | Industrials |
| CAT | 6 | -395.0902 | -65.8484 | 0.3333 | Industrials |
| JNJ | 8 | -390.8220 | -48.8527 | 0.3750 | Healthcare |
| C | 17 | -382.3426 | -22.4907 | 0.2353 | Financials |
| BMY | 17 | -378.2416 | -22.2495 | 0.5294 | Healthcare |
| XLRE | 15 | -375.6122 | -25.0408 | 0.2667 | ETFs |
| AAPL | 8 | -374.2462 | -46.7808 | 0.3750 | Technology |
| ZS | 2 | -351.0176 | -175.5088 | 0.0000 | Technology |
| COST | 9 | -329.7161 | -36.6351 | 0.2222 | Consumer Staples |
| MMM | 4 | -327.7763 | -81.9441 | 0.2500 | Industrials |
| T | 25 | -304.1202 | -12.1648 | 0.1600 | Communication |
| V | 5 | -302.1154 | -60.4231 | 0.2000 | Financials |
| PSX | 5 | -291.1355 | -58.2271 | 0.2000 | Energy |
| LMT | 11 | -274.6771 | -24.9706 | 0.1818 | Industrials |
| AMT | 2 | -266.5281 | -133.2641 | 0.0000 | Materials/Real Estate |
| XLB | 13 | -249.2764 | -19.1751 | 0.3077 | ETFs |
| WFC | 8 | -242.7560 | -30.3445 | 0.2500 | Financials |
| ELV | 3 | -233.8761 | -77.9587 | 0.3333 | Healthcare |
| XLI | 5 | -229.8401 | -45.9680 | 0.4000 | ETFs |
| SPG | 2 | -219.2800 | -109.6400 | 0.5000 | Materials/Real Estate |
| UNH | 6 | -218.7000 | -36.4500 | 0.5000 | Healthcare |
| MU | 13 | -212.4200 | -16.3400 | 0.3077 | Technology |
| NVDA | 15 | -209.8713 | -13.9914 | 0.5333 | Technology |
| META | 10 | -206.4563 | -20.6456 | 0.2000 | Communication |
| GLD | 16 | -193.1621 | -12.0726 | 0.3750 | ETFs |
| PFE | 16 | -191.1472 | -11.9467 | 0.1875 | Healthcare |
| NET | 5 | -177.7232 | -35.5446 | 0.4000 | Technology |
| MS | 8 | -175.8345 | -21.9793 | 0.2500 | Financials |
| XLV | 9 | -166.3211 | -18.4801 | 0.1111 | ETFs |
| CL | 3 | -160.2414 | -53.4138 | 0.0000 | Consumer Staples |
| DIA | 7 | -157.1088 | -22.4441 | 0.2857 | ETFs |
| MDT | 2 | -153.4509 | -76.7255 | 0.5000 | Healthcare |
| GOOGL | 17 | -150.4687 | -8.8511 | 0.3529 | Communication |
| NOW | 4 | -140.9291 | -35.2323 | 0.5000 | Technology |
| CSCO | 13 | -140.1367 | -10.7797 | 0.3077 | Technology |
| EQIX | 1 | -134.0208 | -134.0208 | 0.0000 | Materials/Real Estate |
| WMT | 8 | -128.4595 | -16.0574 | 0.5000 | Consumer Staples |
| SOXX | 7 | -102.2621 | -14.6089 | 0.2857 | ETFs |
| ARM | 13 | -98.5016 | -7.5770 | 0.2308 | Technology |
| PLTR | 25 | -85.6931 | -3.4277 | 0.2800 | Technology |
| BLK | 1 | -57.0245 | -57.0245 | 0.0000 | Financials |
| MA | 2 | -54.0107 | -27.0053 | 0.0000 | Financials |
| PEP | 7 | -46.0637 | -6.5805 | 0.2857 | Consumer Staples |
| USB | 4 | -12.9203 | -3.2301 | 0.0000 | Financials |
| PM | 3 | -4.8198 | -1.6066 | 0.3333 | Consumer Staples |
| TJX | 1 | -3.0442 | -3.0442 | 0.0000 | Consumer Discretionary |
| SPY | 2 | -2.2915 | -1.1458 | 0.0000 | ETFs |
| DECK | 3 | -1.8925 | -0.6308 | 0.3333 | Consumer Discretionary |
| NKE | 9 | -1.1132 | -0.1237 | 0.2222 | Consumer Discretionary |
| XLC | 4 | -0.8265 | -0.2066 | 0.2500 | ETFs |
| DXCM | 4 | -0.7867 | -0.1967 | 0.5000 | Healthcare |
| XLY | 7 | 0.2588 | 0.0370 | 0.1429 | ETFs |
| GS | 1 | 0.3521 | 0.3521 | 1.0000 | Financials |
| IWM | 1 | 0.6729 | 0.6729 | 1.0000 | ETFs |
| QQQ | 1 | 0.7463 | 0.7463 | 1.0000 | ETFs |
| ABT | 4 | 1.1416 | 0.2854 | 0.7500 | Healthcare |
| XLK | 8 | 1.8772 | 0.2346 | 0.5000 | ETFs |
| APP | 12 | 9.6583 | 0.8049 | 0.5000 | Technology |
| NEM | 24 | 45.2830 | 1.8868 | 0.3750 | Materials/Real Estate |
| HD | 5 | 57.8928 | 11.5786 | 0.2000 | Consumer Discretionary |
| MO | 21 | 75.1623 | 3.5792 | 0.3810 | Consumer Staples |
| LLY | 2 | 84.6215 | 42.3108 | 0.5000 | Healthcare |
| PG | 3 | 111.3726 | 37.1242 | 0.6667 | Consumer Staples |
| ETN | 2 | 126.0163 | 63.0081 | 0.5000 | Industrials |
| RTX | 8 | 127.3190 | 15.9149 | 0.3750 | Industrials |
| XLF | 8 | 137.7075 | 17.2134 | 0.6250 | ETFs |
| CI | 4 | 143.3404 | 35.8351 | 0.5000 | Healthcare |
| MCD | 5 | 146.5768 | 29.3154 | 0.2000 | Consumer Discretionary |
| MDLZ | 5 | 162.4102 | 32.4820 | 0.6000 | Consumer Staples |
| XLP | 10 | 163.7845 | 16.3784 | 0.2000 | ETFs |
| LOW | 2 | 172.3232 | 86.1616 | 0.5000 | Consumer Discretionary |
| SCHW | 3 | 176.6667 | 58.8889 | 0.3333 | Financials |
| FCX | 21 | 177.1915 | 8.4377 | 0.3333 | Materials/Real Estate |
| MSFT | 6 | 178.5867 | 29.7644 | 0.3333 | Technology |
| DE | 5 | 190.4562 | 38.0912 | 0.4000 | Industrials |
| HON | 2 | 204.8027 | 102.4014 | 0.5000 | Industrials |
| TGT | 4 | 210.6620 | 52.6655 | 0.2500 | Consumer Discretionary |
| SBUX | 7 | 222.4217 | 31.7745 | 0.4286 | Consumer Discretionary |
| GE | 14 | 231.0165 | 16.5012 | 0.4286 | Industrials |
| UPS | 3 | 263.1604 | 87.7201 | 0.6667 | Industrials |
| QCOM | 2 | 284.4103 | 142.2051 | 0.5000 | Technology |
| CVX | 23 | 290.5920 | 12.6344 | 0.4348 | Energy |
| AXP | 3 | 292.3525 | 97.4508 | 0.3333 | Financials |
| ORCL | 8 | 312.3921 | 39.0490 | 0.6250 | Technology |
| LULU | 5 | 312.9877 | 62.5975 | 0.4000 | Consumer Discretionary |
| UBER | 14 | 323.3465 | 23.0962 | 0.4286 | Consumer Discretionary |
| KKR | 3 | 336.7749 | 112.2583 | 0.3333 | Financials |
| EBAY | 6 | 366.4445 | 61.0741 | 0.5000 | Consumer Discretionary |
| CMG | 6 | 374.9164 | 62.4861 | 0.5000 | Consumer Discretionary |
| ADBE | 3 | 391.6798 | 130.5599 | 1.0000 | Technology |
| TXN | 2 | 403.2207 | 201.6103 | 0.5000 | Technology |
| MRVL | 12 | 437.3081 | 36.4423 | 0.5833 | Technology |
| XLU | 14 | 438.9088 | 31.3506 | 0.4286 | ETFs |
| AXON | 3 | 450.2892 | 150.0964 | 0.6667 | Industrials |
| BX | 5 | 452.7667 | 90.5533 | 0.4000 | Financials |
| COF | 2 | 509.5821 | 254.7910 | 0.5000 | Financials |
| COP | 19 | 554.3294 | 29.1752 | 0.3684 | Energy |
| ISRG | 5 | 599.9499 | 119.9900 | 0.8000 | Healthcare |
| F | 34 | 734.5523 | 21.6045 | 0.3529 | Consumer Discretionary |
| BKNG | 3 | 760.6018 | 253.5339 | 0.6667 | Consumer Discretionary |
| AVGO | 12 | 761.6981 | 63.4748 | 0.5000 | Technology |
| GM | 23 | 770.0406 | 33.4800 | 0.3043 | Consumer Discretionary |
| BAC | 27 | 821.1683 | 30.4136 | 0.5185 | Financials |
| TSLA | 10 | 823.7105 | 82.3711 | 0.4000 | Technology |
| MRNA | 20 | 895.7921 | 44.7896 | 0.4000 | Healthcare |
| XLE | 12 | 920.9118 | 76.7427 | 0.5000 | ETFs |
| PANW | 9 | 954.7311 | 106.0812 | 0.5556 | Technology |
| DVN | 23 | 1,032.1537 | 44.8762 | 0.4348 | Energy |
| DIS | 18 | 1,056.0680 | 58.6704 | 0.5556 | Communication |
| BA | 12 | 1,475.0458 | 122.9205 | 0.5000 | Industrials |

## 2. Entry wait time & unfilled-order capital usage

Filled trades: median wait 0 days 00:00:00, p90 0 days 03:16:36, max 1060 days 22:34:00.

Unfilled (censored) orders: 7

- LLY: pending since 2023-05-03 15:16:00, 1193 days open as of run end

- NET: pending since 2023-05-15 15:16:00, 1181 days open as of run end

- CRWD: pending since 2023-05-18 15:16:00, 1178 days open as of run end

- QCOM: pending since 2023-11-02 15:16:00, 1010 days open as of run end

- AVGO: pending since 2026-04-07 15:16:00, 123 days open as of run end

- MRVL: pending since 2026-04-20 15:16:00, 110 days open as of run end

- FCX: pending since 2026-08-07 15:16:00, 1 days open as of run end


Peak reserved capital: $100,884.23 of $100,000 starting equity. Ending reserved: $90,818.06.


## 3. Exit reason & holding duration

| exit_reason | count | net_pnl | avg_pnl_r | win_rate |
| --- | --- | --- | --- | --- |
| monitor_reversal | 37 | -539.5055 | -0.1339 | 0.3514 |
| stop | 950 | -55,382.4234 | -0.5719 | 0.1874 |
| target | 240 | 46,777.4451 | 1.9833 | 1.0000 |


| duration_bucket | count | net_pnl | win_rate |
| --- | --- | --- | --- |
| <1d | 878 | -27,322.2387 | 0.2540 |
| 1-3d | 248 | 10,755.2819 | 0.5927 |
| 3-7d | 101 | 7,422.4729 | 0.6040 |
| 7-30d | 0 | 0.0000 | nan |
| 30d+ | 0 | 0.0000 | nan |

## 4. Winners'/losers' MFE and MAE (bar-based excursions)

Bar-based excursions from minute OHLC - intrabar high/low sequence is not observable, so this is a resolution approximation, not exact tick-level MFE/MAE. Normalized to planned risk (entry - stop). Cache coverage: 100.0% of closed trades had fully cached minute data; the rest are excluded from this table but counted in the coverage figure.

| group | mfe_r_mean | mfe_r_median | mae_r_mean | mae_r_median | count |
| --- | --- | --- | --- | --- | --- |
| loser | 0.4806 | 0.3950 | 0.8874 | 0.9313 | 796 |
| winner | 1.9826 | 2.0104 | 0.3672 | 0.2826 | 431 |

## 5. Cost impact on expectancy

**(a) Path-dependent (what actually happened - trade sets differ per cost tier because cost feeds equity -> sizing -> admission):**

| cost_model | closed_count | net_expectancy | total_net_pnl |
| --- | --- | --- | --- |
| zero | 979 | 1.0878 | 1,064.9785 |
| baseline | 1227 | -7.4527 | -9,144.4838 |
| stressed | 1087 | -21.4819 | -23,350.8407 |

**(b) Fixed-cohort cost isolation (pure cost drag on one identical set of trades):**

| label | cost_bps_per_leg | total_net_pnl | expectancy | count |
| --- | --- | --- | --- | --- |
| zero_0bps | 0.0000 | 884.4600 | 0.7208 | 1227 |
| baseline_5bps | 5.0000 | -9,144.4838 | -7.4527 | 1227 |
| stressed_13bps | 13.0000 | -25,190.7939 | -20.5304 | 1227 |

For reference, `safe_0_25pct`/baseline: 1403 closed, net P&L $-8,304.80, expectancy $-5.92/trade (headline only).


## 6. Loss clustering by regime & volatility

| regime | count | net_pnl | win_rate |
| --- | --- | --- | --- |
| DOWNTREND | 104 | -4,472.9518 | 0.3077 |
| SIDEWAYS | 785 | -2,318.0499 | 0.3580 |
| UPTREND | 338 | -2,353.4821 | 0.3491 |


| volatility_quartile | count | net_pnl | win_rate |
| --- | --- | --- | --- |
| (0.00074, 0.00504] | 307 | -1,569.1195 | 0.3257 |
| (0.00504, 0.007] | 307 | -3,509.0718 | 0.3583 |
| (0.007, 0.00989] | 306 | 3,053.7194 | 0.3627 |
| (0.00989, 0.035] | 307 | -7,120.0120 | 0.3583 |