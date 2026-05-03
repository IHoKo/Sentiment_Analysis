from __future__ import annotations

from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
SOURCE_PATH = DATA_DIR / "EventManagmentDataSet.csv"
RULE_OUTPUT_PATH = DATA_DIR / "event_sentiment_analysis.csv"
NLP_OUTPUT_PATH = DATA_DIR / "event_nlp_sentiment_analysis.csv"

POSITIVE_EVENT_WORDS = {
    "creative",
    "ideas",
    "webinar",
    "discussion",
    "session",
    "quiz",
    "debate",
    "review",
    "contest",
    "greeting",
    "singing",
}


def as_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_event_data(source_path: Path = SOURCE_PATH) -> pd.DataFrame:
    return pd.read_csv(source_path)


def rule_sentiment_label(score: float) -> str:
    if score > 20:
        return "Positive"
    if score < -20:
        return "Negative"
    return "Neutral"


def analyze_rule_based_sentiment(row: pd.Series) -> pd.Series:
    score = 0
    reasons: list[str] = []

    decision = as_text(row.get("Accept/Reject")).lower()
    availability = as_text(row.get("Room Available/Not Available")).lower()
    priority = as_text(row.get("Type")).lower()
    event_name = as_text(row.get("Event")).lower()
    booking_requested = int(row.get("Request for Booking", 0) or 0)
    cancellation_requested = int(row.get("Request for Cancelling", 0) or 0)
    registers = int(row.get("Registers", 0) or 0)

    if decision == "accept":
        score += 45
        reasons.append("accepted request")
    elif decision == "reject":
        score -= 55
        reasons.append("request was rejected")

    if availability == "available":
        score += 25
        reasons.append("room is available")
    elif availability == "not available":
        score -= 35
        reasons.append("room is not available")

    if priority == "very important":
        score += 20
        reasons.append("event is marked very important")
    elif priority == "important":
        score += 12
        reasons.append("event is marked important")
    elif priority == "normal":
        reasons.append("event priority is normal")

    if booking_requested:
        score += 5
        reasons.append("booking was requested")

    if cancellation_requested:
        score -= 25
        reasons.append("cancellation was requested")

    if registers >= 100:
        score += 15
        reasons.append("very high registration count")
    elif registers >= 50:
        score += 8
        reasons.append("strong registration count")
    elif registers < 20:
        score -= 5
        reasons.append("low registration count")

    matched_words = sorted(word for word in POSITIVE_EVENT_WORDS if word in event_name)
    if matched_words:
        score += min(10, len(matched_words) * 5)
        reasons.append(f"positive event keywords: {', '.join(matched_words)}")

    score = max(-100, min(100, score))
    label = rule_sentiment_label(score)
    reason_text = "; ".join(reasons) if reasons else "no strong rule-based signal"

    return pd.Series(
        {
            "Rule Score": score,
            "Rule Label": label,
            "Rule Reason": f"{label} because {reason_text}",
        }
    )


def add_rule_based_sentiment(event_data: pd.DataFrame) -> pd.DataFrame:
    rule_columns = event_data.apply(analyze_rule_based_sentiment, axis=1)
    return pd.concat([event_data, rule_columns], axis=1)


def build_event_text(row: pd.Series) -> str:
    event = as_text(row.get("Event"))
    priority = as_text(row.get("Type"))
    registrations = int(row.get("Registers", 0) or 0)
    availability = as_text(row.get("Room Available/Not Available"))
    decision = as_text(row.get("Accept/Reject"))
    booking = "booking was requested" if int(row.get("Request for Booking", 0) or 0) else "booking was not requested"
    cancellation = (
        "cancellation was requested"
        if int(row.get("Request for Cancelling", 0) or 0)
        else "cancellation was not requested"
    )

    return (
        f"The event request is for {event}. "
        f"The event priority is {priority}. "
        f"The event has {registrations} registrations. "
        f"The room is {availability}. "
        f"The request decision is {decision}. "
        f"The {booking}. "
        f"The {cancellation}."
    )


def nlp_sentiment_label(score: float) -> str:
    if score >= 20:
        return "Positive"
    if score <= -20:
        return "Negative"
    return "Neutral"


def clause_reason(text: str, label: str, analyzer: SentimentIntensityAnalyzer) -> str:
    clauses = [part.strip() for part in text.split(".") if part.strip()]
    scored_clauses = [(clause, analyzer.polarity_scores(clause)["compound"]) for clause in clauses]

    if label == "Positive":
        selected = [clause for clause, score in scored_clauses if score > 0]
        reason_start = "Positive because VADER found positive language in"
    elif label == "Negative":
        selected = [clause for clause, score in scored_clauses if score < 0]
        reason_start = "Negative because VADER found negative language in"
    else:
        selected = [clause for clause, score in scored_clauses if abs(score) < 0.2]
        reason_start = "Neutral because VADER found mostly balanced or weak sentiment in"

    if not selected and scored_clauses:
        selected = [max(scored_clauses, key=lambda item: abs(item[1]))[0]]

    return f"{reason_start}: " + "; ".join(selected[:3])


def analyze_nlp_sentiment(row: pd.Series, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.Series:
    analyzer = analyzer or SentimentIntensityAnalyzer()
    text = build_event_text(row)
    scores = analyzer.polarity_scores(text)
    score = round(scores["compound"] * 100, 2)
    label = nlp_sentiment_label(score)

    return pd.Series(
        {
            "NLP Text": text,
            "NLP Negative": scores["neg"],
            "NLP Neutral": scores["neu"],
            "NLP Positive": scores["pos"],
            "NLP Compound": scores["compound"],
            "NLP Score": score,
            "NLP Label": label,
            "NLP Reason": clause_reason(text, label, analyzer),
        }
    )


def add_nlp_sentiment(event_data: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    nlp_columns = event_data.apply(lambda row: analyze_nlp_sentiment(row, analyzer), axis=1)
    return pd.concat([event_data, nlp_columns], axis=1)


def compare_methods(row: pd.Series) -> pd.DataFrame:
    rule_result = analyze_rule_based_sentiment(row)
    nlp_result = analyze_nlp_sentiment(row)

    return pd.DataFrame(
        [
            {
                "Method": "Rule-based",
                "Score": rule_result["Rule Score"],
                "Label": rule_result["Rule Label"],
                "Reason": rule_result["Rule Reason"],
            },
            {
                "Method": "VADER NLP",
                "Score": nlp_result["NLP Score"],
                "Label": nlp_result["NLP Label"],
                "Reason": nlp_result["NLP Reason"],
            },
        ]
    )
