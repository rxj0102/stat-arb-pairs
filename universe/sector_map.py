"""Sector groups of Ken French 49 industry portfolios (open data universe).

The earlier version used ~100 individual S&P 500 stocks from Yahoo Finance.
There is no free open source for daily single-stock prices, so the universe is
now the Ken French 49 industry portfolios (value weighted, daily), grouped by
broad sector. Pairs are screened within each group. 'Other' is left out.
"""

SECTOR_TICKERS = {
    "Information Technology": ["Hardw", "Softw", "Chips", "LabEq", "BusSv", "Telcm"],
    "Financials": ["Banks", "Insur", "RlEst", "Fin"],
    "Health Care": ["Hlth", "MedEq", "Drugs"],
    "Consumer Staples": ["Agric", "Food", "Soda", "Beer", "Smoke", "Hshld"],
    "Energy and Utilities": ["Oil", "Coal", "Util"],
    "Materials": ["Chems", "Steel", "Mines", "Gold", "Paper", "Boxes", "Rubbr", "Txtls"],
    "Industrials": ["Mach", "ElcEq", "Aero", "Ships", "Guns", "Trans", "Cnstr", "BldMt", "FabPr"],
    "Consumer Discretionary": ["Autos", "Clths", "Toys", "Fun", "Books", "PerSv", "Rtail", "Meals", "Whlsl"],
}
