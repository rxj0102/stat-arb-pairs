"""Static GICS sector mapping for a representative S&P 500 universe."""

# ~20 liquid stocks per sector across 5 sectors
SECTOR_TICKERS = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "ACN",
        "CRM", "AMD", "TXN", "QCOM", "INTC", "IBM", "NOW", "AMAT",
        "MU", "ADI", "KLAC", "LRCX",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP",
        "USB", "PNC", "TFC", "COF", "SCHW", "MET", "PRU", "AFL",
        "ALL", "CB", "MMC", "AON",
    ],
    "Health Care": [
        "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
        "BMY", "AMGN", "PFE", "GILD", "ISRG", "SYK", "MDT", "CI",
        "ELV", "HUM", "CVS", "VRTX",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL",
        "MDLZ", "GIS", "K", "HSY", "CPB", "HRL", "SJM", "MKC",
        "CAG", "CHD", "CLX", "KMB",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
        "PXD", "OXY", "HAL", "BKR", "DVN", "HES", "FANG", "APA",
        "MRO", "NOV", "HP", "RRC",
    ],
}
