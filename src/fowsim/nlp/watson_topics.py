from __future__ import annotations

import os
import logging
from typing import Optional
from dataclasses import dataclass

import pandas as pd
import requests


logger = logging.getLogger(__name__)


@dataclass
class WatsonConfig:
    """Configuration for IBM Watson NLU API."""
    api_key: str
    url: str
    version: str = "2022-04-07"
    
    @classmethod
    def from_env(cls) -> Optional["WatsonConfig"]:
        """Load Watson credentials from environment variables."""
        api_key = os.getenv("WATSON_NLU_API_KEY")
        url = os.getenv("WATSON_NLU_URL")
        
        if not api_key or not url:
            logger.warning(
                "Watson NLU credentials not found. "
                "Set WATSON_NLU_API_KEY and WATSON_NLU_URL environment variables."
            )
            return None
        
        return cls(api_key=api_key, url=url)


def analyze_text_with_watson(
    text: str,
    config: WatsonConfig,
    features: Optional[list[str]] = None
) -> dict:
    """Analyze text using IBM Watson Natural Language Understanding.
    
    Args:
        text: Text to analyze
        config: Watson API configuration
        features: Features to extract (sentiment, keywords, concepts, categories, emotion)
    
    Returns:
        Dictionary with analysis results
    """
    if features is None:
        features = ["sentiment", "keywords", "concepts", "categories"]
    
    url = f"{config.url}/v1/analyze"
    params = {"version": config.version}
    
    payload = {
        "text": text,
        "features": {}
    }
    
    # Configure features
    if "sentiment" in features:
        payload["features"]["sentiment"] = {}
    if "keywords" in features:
        payload["features"]["keywords"] = {"limit": 10, "sentiment": True}
    if "concepts" in features:
        payload["features"]["concepts"] = {"limit": 10}
    if "categories" in features:
        payload["features"]["categories"] = {}
    if "emotion" in features:
        payload["features"]["emotion"] = {}
    
    try:
        response = requests.post(
            url,
            json=payload,
            params=params,
            auth=("apikey", config.api_key),
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Watson NLU analysis failed: {e}")
        return {}


def extract_sentiment_score(analysis: dict) -> float:
    """Extract sentiment score from Watson analysis result.
    
    Returns score between -1 (negative) and 1 (positive).
    """
    if not analysis or "sentiment" not in analysis:
        return 0.0
    
    sentiment = analysis["sentiment"].get("document", {})
    return float(sentiment.get("score", 0.0))


def extract_work_future_concepts(analysis: dict) -> list[str]:
    """Extract concepts related to future of work from Watson analysis."""
    if not analysis or "concepts" not in analysis:
        return []
    
    concepts = analysis.get("concepts", [])
    work_related = [
        "automation", "artificial intelligence", "remote work", "digital transformation",
        "machine learning", "workplace", "employment", "labor market", "skills",
        "technology adoption", "productivity", "innovation", "telecommuting"
    ]
    
    extracted = []
    for concept in concepts:
        text = concept.get("text", "").lower()
        if any(term in text for term in work_related):
            extracted.append(concept.get("text"))
    
    return extracted


def build_text_signal_index(text_records: pd.DataFrame, config: Optional[WatsonConfig] = None) -> pd.DataFrame:
    """Build text sentiment and topic index from news/reports about future of work.

    Expected input columns:
      - year (int)
      - text (string)
      - source (optional, string)
    
    Output columns:
      - year
      - text_sentiment_index (float, -1 to 1)
      - automation_mention_count (int)
      - ai_mention_count (int)
      - remote_work_mention_count (int)
      - innovation_index (float)
    """
    if "year" not in text_records.columns or "text" not in text_records.columns:
        raise ValueError("text_records must include 'year' and 'text' columns")
    
    if config is None:
        config = WatsonConfig.from_env()
    
    results = []
    
    for _, row in text_records.iterrows():
        year = row["year"]
        text = str(row["text"])
        
        # Simple keyword-based analysis (fallback if Watson not available)
        automation_count = text.lower().count("automation") + text.lower().count("automate")
        ai_count = text.lower().count("ai ") + text.lower().count("artificial intelligence")
        remote_count = text.lower().count("remote work") + text.lower().count("telecommuting")
        
        sentiment = 0.0
        innovation_score = 0.0
        
        # Use Watson if available
        if config and len(text) > 50:
            try:
                analysis = analyze_text_with_watson(text, config)
                sentiment = extract_sentiment_score(analysis)
                concepts = extract_work_future_concepts(analysis)
                innovation_score = len(concepts) / 10.0  # Normalize
            except Exception as e:
                logger.debug(f"Watson analysis skipped for year {year}: {e}")
        
        results.append({
            "year": year,
            "text_sentiment_index": sentiment,
            "automation_mention_count": automation_count,
            "ai_mention_count": ai_count,
            "remote_work_mention_count": remote_count,
            "innovation_index": innovation_score
        })
    
    # Aggregate by year
    df = pd.DataFrame(results)
    out = df.groupby("year").agg({
        "text_sentiment_index": "mean",
        "automation_mention_count": "sum",
        "ai_mention_count": "sum",
        "remote_work_mention_count": "sum",
        "innovation_index": "mean"
    }).reset_index()
    
    return out


def load_news_data(file_path: str) -> pd.DataFrame:
    """Load news/reports data from CSV.
    
    Expected columns: date, title, text, source
    Converts to year-based format.
    """
    df = pd.read_csv(file_path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
    
    # Combine title and text
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    
    return df[["year", "text"]].dropna()
