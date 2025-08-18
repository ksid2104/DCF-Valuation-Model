# DCF Valuation Model

This project implements a Python-based **Discounted Cash Flow (DCF) valuation model** for a US-listed company with relatively predictable and stable revenues. The model estimates the intrinsic equity value and price per share using historical financial data, projected cash flows, and a terminal value.

**Note:** This model is **not intended to provide precise valuation figures**. It relies on simplifying assumptions and serves primarily as a **learning exercise** to practice the DCF methodology and apply financial concepts in Python.
---

## Features

- **Free Cash Flow (FCF) Calculation**: Computes historical FCF from income statement and cash flow data.  
- **FCF Projection**: Projects future FCF over 5–10 years using historical CAGR.  
- **Weighted Average Cost of Capital (WACC)**: Calculates WACC using market data, cost of debt, CAPM, and tax rate.  
- **DCF Valuation**: Discounts projected FCFs to present value using WACC.  
- **Terminal Value**: Computes terminal value using a perpetual growth rate `g`, based on the average of the US risk-free rate (10Y bond) and historical US GDP growth since 1928.  
- **Equity Value & Price per Share**: Calculates equity value by adjusting for net debt and estimates intrinsic price per share.  
- **Sensitivity Analysis**: Generates a table showing valuation sensitivity to changes in WACC and perpetual growth rate.  
- **Visualizations**:
  - Historical and projected FCF
  - DCF vs FCF 
  - DCF breakdown vs terminal value
---

## Methodology key points
- For the market risk premium, we use an exogenous ERP. You can find it by clicking on this link : https://pages.stern.nyu.edu/~adamodar/
- Terminal value uses a perpetual growth rate calculated as the average of long-term US GDP growth and the current risk-free rate.  
- Sensitivity analysis explores valuation across WACC from 1%–20% and growth rate `g` from 1%–10%.

---

## Example of Output

| Date       | FCF               | DCF               |
|-----------|-----------------|-----------------|
| 2034-12-31 | 75,150,485,925 $ | 36,670,200,189 $ |
| 2033-12-31 | 74,223,718,670 $ | 39,223,692,539 $ |
| 2032-12-31 | 73,308,380,450 $ | 41,954,994,751 $ |
| 2031-12-31 | 72,404,330,320 $ | 44,876,488,432 $ |
| ...       | ...             | ...             |

- **WACC:** 0.083  
- **Terminal Value:** 1,708,920,756,673  
- **DCF Valuation:** 1,272,447,189,769  
- **Equity Value:** 1,259,538,189,769  
- **Estimated Price per Share:** 169.43  
- **Current Market Price (Real):** 520.17  
- **DCF vs Terminal Value:** DCF = 34.47%, TV = 65.53%  
- According to this model, MSFT appears **undervalued by 350.74 $**

Sensitivity Table (WACC vs Terminal Growth g)

| WACC \ g | 0.09               | 0.08               | 0.07               |
|-----------|------------------|------------------|------------------|
| 0.199     | 431,928,839,666 $ | 418,364,305,963 $ | 406,902,800,741 $ |
| 0.198     | 435,337,510,805 $ | 421,438,897,053 $ | 409,711,941,700 $ |
| 0.197     | 438,806,921,857 $ | 424,563,676,670 $ | 412,563,462,221 $ |
| ...       | ...              | ...              | ...              |
