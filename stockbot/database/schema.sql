-- Stock Analysis Database Schema
-- Stores historical stock research, findings, and analysis results

-- Analysis runs tracking
CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,  -- 'undervalued', 'single_stock', 'top20'
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    preferences TEXT,  -- JSON string of preferences
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed'
    total_candidates INTEGER DEFAULT 0,
    final_selections INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main stock findings table
CREATE TABLE IF NOT EXISTS stock_finds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id INTEGER,
    ticker TEXT NOT NULL,
    company_name TEXT,
    exchange TEXT,  -- 'TSX', 'NYSE', 'NASDAQ', etc.
    sector TEXT,
    industry TEXT,
    discovery_source TEXT,  -- 'reddit', 'screening', 'fundamental', 'manual'
    confidence_score REAL,  -- 1-10 rating
    current_price REAL,
    market_cap REAL,
    pe_ratio REAL,
    price_to_book REAL,
    debt_to_equity REAL,
    current_ratio REAL,
    revenue_growth REAL,
    analyst_rating TEXT,
    investment_thesis TEXT,
    catalysts TEXT,  -- JSON array of catalysts
    risks TEXT,  -- JSON array of risks
    price_target_bull REAL,
    price_target_base REAL,
    price_target_bear REAL,
    discovered_at TIMESTAMP NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',  -- 'active', 'archived', 'sold'
    FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id),
    UNIQUE(ticker, discovered_at)
);

-- Reddit mentions tracking
CREATE TABLE IF NOT EXISTS reddit_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    title TEXT,
    selftext TEXT,
    sentiment TEXT,  -- 'bullish', 'bearish', 'neutral', 'mixed'
    score INTEGER,
    num_comments INTEGER,
    permalink TEXT,
    top_comments TEXT,  -- JSON array of top comments
    mentioned_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial metrics time series
CREATE TABLE IF NOT EXISTS financial_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    metric_date DATE NOT NULL,
    revenue REAL,
    net_income REAL,
    operating_cash_flow REAL,
    free_cash_flow REAL,
    total_debt REAL,
    shareholders_equity REAL,
    shares_outstanding REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, metric_date)
);

-- Stock relationships (similar stocks, comp sets)
CREATE TABLE IF NOT EXISTS stock_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker_a TEXT NOT NULL,
    ticker_b TEXT NOT NULL,
    relationship_type TEXT NOT NULL,  -- 'similar', 'competitor', 'supplier', 'customer'
    similarity_score REAL,  -- 0-1 similarity metric
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker_a, ticker_b, relationship_type)
);

-- Agent research notes
CREATE TABLE IF NOT EXISTS research_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    note_type TEXT NOT NULL,  -- 'catalyst', 'risk', 'update', 'thesis_change'
    content TEXT NOT NULL,
    source TEXT,  -- 'agent', 'reddit', 'news', 'manual'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_finds_ticker ON stock_finds(ticker);
CREATE INDEX IF NOT EXISTS idx_stock_finds_exchange ON stock_finds(exchange);
CREATE INDEX IF NOT EXISTS idx_stock_finds_sector ON stock_finds(sector);
CREATE INDEX IF NOT EXISTS idx_stock_finds_confidence ON stock_finds(confidence_score);
CREATE INDEX IF NOT EXISTS idx_stock_finds_discovered ON stock_finds(discovered_at);
CREATE INDEX IF NOT EXISTS idx_stock_finds_status ON stock_finds(status);

CREATE INDEX IF NOT EXISTS idx_reddit_mentions_ticker ON reddit_mentions(ticker);
CREATE INDEX IF NOT EXISTS idx_reddit_mentions_subreddit ON reddit_mentions(subreddit);
CREATE INDEX IF NOT EXISTS idx_reddit_mentions_mentioned ON reddit_mentions(mentioned_at);

CREATE INDEX IF NOT EXISTS idx_financial_metrics_ticker ON financial_metrics(ticker);
CREATE INDEX IF NOT EXISTS idx_financial_metrics_date ON financial_metrics(metric_date);

CREATE INDEX IF NOT EXISTS idx_research_notes_ticker ON research_notes(ticker);
CREATE INDEX IF NOT EXISTS idx_research_notes_created ON research_notes(created_at);

-- ======================================================================
-- PORTFOLIO TRACKING AND LEARNING TABLES
-- ======================================================================

-- Portfolio holdings tracking
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_find_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TIMESTAMP NOT NULL,
    shares_virtual INTEGER DEFAULT 1000,  -- Virtual shares for tracking percentage gains
    current_price REAL,
    last_price_update TIMESTAMP,
    exit_price REAL,
    exit_date TIMESTAMP,
    position_status TEXT DEFAULT 'active',  -- 'active', 'closed', 'monitoring'
    holding_days INTEGER DEFAULT 0,
    total_return_pct REAL,
    max_drawdown_pct REAL,
    max_gain_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_find_id) REFERENCES stock_finds(id),
    UNIQUE(stock_find_id)
);

-- Performance tracking snapshots
CREATE TABLE IF NOT EXISTS performance_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    holding_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    price_at_check REAL NOT NULL,
    return_pct REAL,
    days_held INTEGER,
    check_date TIMESTAMP NOT NULL,
    volatility_30d REAL,
    catalyst_events TEXT,  -- JSON array of catalyst events detected
    news_sentiment TEXT,  -- 'positive', 'negative', 'neutral'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (holding_id) REFERENCES portfolio_holdings(id)
);

-- Catalyst realization tracking
CREATE TABLE IF NOT EXISTS catalyst_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_find_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    predicted_catalyst TEXT NOT NULL,
    catalyst_type TEXT,  -- 'earnings', 'product_launch', 'regulatory', 'merger', 'management_change'
    realization_status TEXT DEFAULT 'pending',  -- 'pending', 'realized', 'failed', 'delayed'
    realization_date TIMESTAMP,
    impact_on_price REAL,  -- Price change % when catalyst occurred
    validation_notes TEXT,
    confidence_at_prediction REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP,
    FOREIGN KEY (stock_find_id) REFERENCES stock_finds(id)
);

-- Learning insights aggregation
CREATE TABLE IF NOT EXISTS learning_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_type TEXT NOT NULL,  -- 'confidence_calibration', 'source_reliability', 'sector_timing', 'catalyst_accuracy'
    insight_category TEXT,  -- Subcategory (e.g., 'reddit', 'screening', 'technology', 'healthcare')
    metric_name TEXT NOT NULL,
    metric_value REAL,
    sample_size INTEGER,
    time_period_days INTEGER,
    insight_summary TEXT,
    actionable_recommendation TEXT,
    confidence_level TEXT,  -- 'high', 'medium', 'low' based on sample size
    generated_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance tracking tables
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_ticker ON portfolio_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_status ON portfolio_holdings(position_status);
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_entry_date ON portfolio_holdings(entry_date);

CREATE INDEX IF NOT EXISTS idx_performance_tracking_holding ON performance_tracking(holding_id);
CREATE INDEX IF NOT EXISTS idx_performance_tracking_date ON performance_tracking(check_date);
CREATE INDEX IF NOT EXISTS idx_performance_tracking_ticker ON performance_tracking(ticker);

CREATE INDEX IF NOT EXISTS idx_catalyst_tracking_ticker ON catalyst_tracking(ticker);
CREATE INDEX IF NOT EXISTS idx_catalyst_tracking_status ON catalyst_tracking(realization_status);
CREATE INDEX IF NOT EXISTS idx_catalyst_tracking_stock_find ON catalyst_tracking(stock_find_id);

CREATE INDEX IF NOT EXISTS idx_learning_insights_type ON learning_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_learning_insights_generated ON learning_insights(generated_at);
