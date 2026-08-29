"""
Stock universe organised by sector.
The data node seeds the tickers table from this list on first run.
"""

TECHNOLOGY = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "TSLA",
    "ORCL", "AMD", "ADBE", "CRM", "QCOM", "TXN", "NFLX",
    "CSCO", "INTC", "MU", "AMAT", "NOW", "PANW",
    # High-momentum additions
    "PLTR", "ARM", "MRVL", "NET", "CRWD", "ZS", "APP",
]
CONSUMER_DISCRETIONARY = [
    "AMZN", "HD", "MCD", "NKE", "LOW", "BKNG", "CMG",
    "TJX", "SBUX", "TGT", "GM", "F", "EBAY",
    "LULU", "DECK", "UBER",
]
CONSUMER_STAPLES = [
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO",
    "CL", "MDLZ", "KHC",
]
FINANCIALS = [
    "JPM", "V", "MA", "BAC", "GS", "MS", "BLK",
    "WFC", "AXP", "SCHW", "C", "PNC", "COF", "USB",
    "BX", "KKR", "SPGI",
]
HEALTHCARE = [
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT",
    "PFE", "BMY", "MDT", "GILD", "CVS", "CI", "ELV",
    "ISRG", "DXCM", "VRTX", "REGN", "MRNA",
]
ENERGY = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "DVN"]
INDUSTRIALS = [
    "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE",
    "BA", "FDX", "MMM", "NSC", "ETN", "AXON",
]
COMMUNICATION = ["GOOGL", "META", "DIS", "CMCSA", "T", "VZ", "NFLX"]
MATERIALS_REALESTATE = [
    "LIN", "APD", "SHW", "AMT", "PLD", "EQIX",
    "NEM", "FCX", "SPG",
]
ETFS = [
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs (full set)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY",
    "XLC", "XLRE", "XLB", "XLU",
    # Specialty signals
    "SOXX", "IBB", "GLD",
]

SECTOR_MAP: dict[str, list[str]] = {
    "Technology":             TECHNOLOGY,
    "Consumer Discretionary": CONSUMER_DISCRETIONARY,
    "Consumer Staples":       CONSUMER_STAPLES,
    "Financials":             FINANCIALS,
    "Healthcare":             HEALTHCARE,
    "Energy":                 ENERGY,
    "Industrials":            INDUSTRIALS,
    "Communication":          COMMUNICATION,
    "Materials/Real Estate":  MATERIALS_REALESTATE,
    "ETFs":                   ETFS,
}

UNIVERSE: list[str] = sorted(set(
    TECHNOLOGY + CONSUMER_DISCRETIONARY + CONSUMER_STAPLES +
    FINANCIALS + HEALTHCARE + ENERGY + INDUSTRIALS +
    COMMUNICATION + MATERIALS_REALESTATE + ETFS
))

# -- Crypto universe -----------------------------------------------------------
# Database/data symbols use Yahoo/Coinbase's BASE-USD spelling. Execution
# translates them to Alpaca's BASE/USD spelling. Availability is checked against
# Alpaca's live Assets API before any paper order is submitted.
CRYPTO: list[str] = [
    # Large caps
    "BTC-USD",    # Bitcoin
    "ETH-USD",    # Ethereum
    "SOL-USD",    # Solana
    "XRP-USD",    # XRP
    "DOGE-USD",   # Dogecoin
    "AVAX-USD",   # Avalanche
    # DeFi
    "LINK-USD",   # Chainlink
    "UNI-USD",    # Uniswap
    "AAVE-USD",   # Aave
    "INJ-USD",    # Injective
    # Layer 2 / Alt L1
    "POL-USD",    # Polygon (rebranded from MATIC)
    "OP-USD",     # Optimism
    "ARB-USD",    # Arbitrum
    "SUI-USD",    # Sui
    "SEI-USD",    # Sei
    # Meme / High momentum
    "PEPE-USD",   # Pepe
    "WIF-USD",    # dogwifhat
    # AI / Data
    "FET-USD",    # Fetch.ai
    "RENDER-USD", # Render
    "TAO-USD",    # Bittensor
    # Ecosystem
    "TON-USD",    # Toncoin
    "JUP-USD",    # Jupiter
]

CRYPTO_SET: set[str] = set(CRYPTO)
