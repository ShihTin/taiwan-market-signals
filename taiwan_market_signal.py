import os
import re
import requests
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone


WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DEBUG_FETCH = os.getenv("DEBUG_FETCH", "1") == "1"

TAIEX = "^TWII"

WATCHLIST = {
    "0050": ["0050.TW"],
    "00981A": ["00981A.TW"],
    "2330": ["2330.TW"],
    "2345": ["2345.TW"],
    "6183": ["6183.TW", "6183.TWO"],
    "9911": ["9911.TW"],
    "2812": ["2812.TW"],
}


def debug(message):
    if DEBUG_FETCH:
        print(message)


def safe_float(value):
    if isinstance(value, pd.Series):
        value = value.dropna()
        if value.empty:
            return None
        value = value.iloc[0]

    if value is None or pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.replace(",", "").replace("%", "").strip()
        if not value:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="column",
        )
    except Exception as e:
        debug(f"[yfinance] {ticker} fetch failed: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)

    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["Close"])

    if df.empty:
        return pd.DataFrame()

    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    return df


def download_first_available(tickers, period: str = "2y"):
    for ticker in tickers:
        df = download_data(ticker, period)
        if not df.empty:
            return ticker, df
    return tickers[0], pd.DataFrame()


def latest_close(df: pd.DataFrame):
    if df.empty:
        return None
    return safe_float(df.iloc[-1]["Close"])


def latest_ma(df: pd.DataFrame, column: str):
    if df.empty or column not in df.columns:
        return None
    return safe_float(df.iloc[-1][column])


def drawdown_from_high(df: pd.DataFrame, window: int):
    if df.empty or "Close" not in df.columns:
        return None, None, None

    recent = df.tail(window)
    high = safe_float(recent["Close"].max())
    close = latest_close(df)

    if high is None or close is None or high == 0:
        return close, high, None

    drawdown_pct = (close / high - 1) * 100
    return close, high, drawdown_pct


def ma_position_text(close, ma_value, ma_name):
    if close is None or ma_value is None:
        return f"{ma_name} N/A"

    position = "高於" if close >= ma_value else "低於"
    return f"{position}{ma_name} ({ma_value:.2f})"


def ndc_signal_label(score):
    if score is None:
        return "N/A"
    if 9 <= score <= 16:
        return "藍燈"
    if 17 <= score <= 22:
        return "黃藍燈"
    if 23 <= score <= 31:
        return "綠燈"
    if 32 <= score <= 37:
        return "黃紅燈"
    if 38 <= score <= 45:
        return "紅燈"
    return "燈號區間外"


def get_manual_float(env_name):
    value = os.getenv(env_name)
    return safe_float(value)


def get_ndc_signal():
    manual_score = get_manual_float("NDC_SIGNAL_SCORE")
    if manual_score is not None:
        return f"{manual_score:.0f} ({ndc_signal_label(manual_score)})"

    urls = [
        "https://index.ndc.gov.tw/n/zh_tw/data/eco/indicators",
        "https://index.ndc.gov.tw/n/zh_tw/data/eco#/1",
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            debug(f"[NDC] {url} status={response.status_code} length={len(response.text)}")
            response.raise_for_status()

            text = BeautifulSoup(response.text, "html.parser").get_text("\n")
            patterns = [
                r"景氣對策信號[^\d]{0,20}(\d{1,2})",
                r"綜合判斷分數[^\d]{0,20}(\d{1,2})",
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    score = safe_float(match.group(1))
                    return f"{score:.0f} ({ndc_signal_label(score)})"

        except Exception as e:
            debug(f"[NDC] fetch failed: {e}")

    return "N/A"


def get_margin_maintenance_ratio():
    manual_ratio = get_manual_float("MARGIN_MAINTENANCE_RATIO")
    if manual_ratio is not None:
        label = "相對低點" if manual_ratio < 140 else "正常"
        return manual_ratio, f"{manual_ratio:.1f}% ({label})"

    return None, "N/A"


def get_margin_balance_series(days: int = 45) -> pd.DataFrame:
    start_date = (datetime.now(timezone.utc) - timedelta(days=days * 2)).strftime("%Y-%m-%d")
    datasets = [
        "TaiwanStockTotalMarginPurchaseShortSale",
        "TaiwanStockMarginPurchaseShortSale",
    ]

    for dataset in datasets:
        try:
            response = requests.get(
                "https://api.finmindtrade.com/api/v4/data",
                params={"dataset": dataset, "start_date": start_date},
                timeout=10,
            )
            debug(f"[FinMind] {dataset} status={response.status_code}")
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None

            if not data:
                continue

            df = pd.DataFrame(data)
            if "date" not in df.columns:
                continue

            balance_col = find_margin_balance_column(df)
            if balance_col is None:
                debug(f"[FinMind] balance column not found: {list(df.columns)}")
                continue

            df["date"] = pd.to_datetime(df["date"])
            df["balance"] = pd.to_numeric(df[balance_col], errors="coerce")
            df = df.dropna(subset=["balance"]).sort_values("date")

            if not df.empty:
                return df[["date", "balance"]]

        except Exception as e:
            debug(f"[FinMind] fetch failed: {e}")

    return pd.DataFrame()


def find_margin_balance_column(df: pd.DataFrame):
    preferred = [
        "MarginPurchaseTodayBalance",
        "MarginPurchaseBalance",
        "margin_purchase_balance",
        "FinancingBalance",
        "融資餘額",
    ]

    for column in preferred:
        if column in df.columns:
            return column

    for column in df.columns:
        normalized = str(column).lower()
        if "marginpurchase" in normalized and "balance" in normalized:
            return column
        if "financing" in normalized and "balance" in normalized:
            return column

    return None


def get_margin_balance_change():
    manual_change = get_manual_float("MARGIN_BALANCE_20D_CHANGE")
    if manual_change is not None:
        return manual_change, f"{manual_change:+.1f}%"

    df = get_margin_balance_series()
    if df.empty or len(df) <= 20:
        return None, "N/A"

    start = safe_float(df.iloc[-21]["balance"])
    end = safe_float(df.iloc[-1]["balance"])

    if start is None or end is None or start == 0:
        return None, "N/A"

    change_pct = (end / start - 1) * 100
    return change_pct, f"{change_pct:+.1f}%"


def get_taiwan_manufacturing_pmi():
    manual_pmi = get_manual_float("TAIWAN_MANUFACTURING_PMI")
    if manual_pmi is not None:
        label = "景氣正在收縮" if manual_pmi < 50 else "景氣擴張"
        return manual_pmi, f"{manual_pmi:.1f} ({label})"

    urls = [
        "https://www.cier.edu.tw",
        "https://www.cier.edu.tw/ct.asp?xItem=20000&CtNode=1350",
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            debug(f"[PMI] {url} status={response.status_code} length={len(response.text)}")
            response.raise_for_status()
            text = BeautifulSoup(response.text, "html.parser").get_text("\n")

            match = re.search(r"製造業\s*PMI[^\d]{0,30}(\d{2}\.\d|\d{2})", text, re.I)
            if match:
                pmi = safe_float(match.group(1))
                label = "景氣正在收縮" if pmi < 50 else "景氣擴張"
                return pmi, f"{pmi:.1f} ({label})"

        except Exception as e:
            debug(f"[PMI] fetch failed: {e}")

    return None, "N/A"


def index_signal(half_drawdown, year_drawdown):
    if half_drawdown is None and year_drawdown is None:
        return "N/A"

    if half_drawdown is not None and half_drawdown >= -0.1:
        return "創新高"

    if year_drawdown is not None:
        for threshold in [-40, -35, -30, -25]:
            if year_drawdown <= threshold:
                return f"距一年高點 {threshold}%"

    if half_drawdown is not None:
        for threshold in [-20, -15, -10, -5]:
            if half_drawdown <= threshold:
                return f"距半年高點 {threshold}%"

    return "正常"


def margin_maintenance_signal(ratio):
    if ratio is None:
        return "N/A"
    if ratio < 140:
        return "大盤融資維持率低於 140%，相對低點"
    return "正常"


def margin_balance_signal(change_pct):
    if change_pct is None:
        return "N/A"
    if change_pct <= -15:
        return "融資餘額 20 日減 15%，斷頭潮"
    if change_pct <= -8:
        return "融資餘額 20 日減 8%，開始恐慌殺出"
    return "正常"


def pmi_signal(pmi):
    if pmi is None:
        return "N/A"
    if pmi < 50:
        return "台灣製造業 PMI 低於 50，景氣正在收縮"
    return "正常"


def price_one_year_text(name, df):
    close, _, drawdown = drawdown_from_high(df, 252)
    close_text = "N/A" if close is None else f"{close:.2f}"
    if drawdown is None:
        drawdown_text = "N/A"
    elif drawdown >= -0.1:
        drawdown_text = "創一年新高"
    else:
        drawdown_text = f"距一年高點 {drawdown:+.1f}%"
    return f"{name}目前股價：{close_text}（{drawdown_text}）"


def main():
    taiex_df = download_data(TAIEX, "2y")
    taiex_close, _, taiex_half_drawdown = drawdown_from_high(taiex_df, 126)
    _, _, taiex_year_drawdown = drawdown_from_high(taiex_df, 252)
    taiex_ma50 = latest_ma(taiex_df, "ma50")
    taiex_ma200 = latest_ma(taiex_df, "ma200")

    ndc_signal = get_ndc_signal()
    margin_ratio, margin_ratio_text = get_margin_maintenance_ratio()
    margin_change, margin_change_text = get_margin_balance_change()
    pmi_value, pmi_text = get_taiwan_manufacturing_pmi()

    stock_lines = []
    for name, tickers in WATCHLIST.items():
        _, df = download_first_available(tickers, "2y")
        stock_lines.append(price_one_year_text(name, df))

    taiex_close_text = "N/A" if taiex_close is None else f"{taiex_close:.2f}"
    taiex_drawdown_text = "N/A" if taiex_half_drawdown is None else f"{taiex_half_drawdown:+.1f}%"
    taiex_ma50_text = ma_position_text(taiex_close, taiex_ma50, "50MA")
    taiex_ma200_text = ma_position_text(taiex_close, taiex_ma200, "200MA")

    message = f"""台股每日訊號通知

每日顯示數據：
1. 大盤
加權指數：{taiex_close_text} [距半年高點 {taiex_drawdown_text}、{taiex_ma50_text}、{taiex_ma200_text}]
國發會景氣對策信號：{ndc_signal}
大盤融資維持率：{margin_ratio_text}
融資餘額 20 日增減%：{margin_change_text}
台灣製造業 PMI：{pmi_text}

2. 個股
{chr(10).join(stock_lines)}

觸發訊號顯示：
大盤加權指數：{index_signal(taiex_half_drawdown, taiex_year_drawdown)}
大盤融資維持率：{margin_maintenance_signal(margin_ratio)}
融資餘額：{margin_balance_signal(margin_change)}
台灣製造業 PMI：{pmi_signal(pmi_value)}
"""

    print(message)
    send_webhook(message)


def send_webhook(message: str):
    if not WEBHOOK_URL:
        print("No webhook url set")
        return

    try:
        response = requests.post(
            WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print("Webhook failed:", e)


if __name__ == "__main__":
    main()
