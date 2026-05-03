from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_PATH = DATA_DIR / "event_sentiment_analysis.csv"


def load_event_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    source_files = [path for path in csv_files if path.name != OUTPUT_PATH.name]
    if not source_files:
        raise FileNotFoundError(f"No source CSV files found in {data_dir}")

    if len(source_files) > 1:
        print("Multiple CSV files found. Loading the first one:")
        for csv_file in source_files:
            print(f"  - {csv_file.name}")

    csv_path = source_files[0]
    print(f"Loading data from: {csv_path}")
    return pd.read_csv(csv_path)


def _as_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def analyze_event_sentiment(row: pd.Series) -> pd.Series:
    score = 0
    reasons: list[str] = []

    decision = _as_text(row.get("Accept/Reject")).lower()
    availability = _as_text(row.get("Room Available/Not Available")).lower()
    priority = _as_text(row.get("Type")).lower()
    event_name = _as_text(row.get("Event")).lower()
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

    positive_event_words = {
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
    matched_words = sorted(word for word in positive_event_words if word in event_name)
    if matched_words:
        score += min(10, len(matched_words) * 5)
        reasons.append(f"positive event keywords: {', '.join(matched_words)}")

    score = max(-100, min(100, score))

    if score > 20:
        label = "Positive"
    elif score < -20:
        label = "Negative"
    else:
        label = "Neutral"

    if label == "Negative":
        explanation = "Negative because " + "; ".join(reasons)
    elif label == "Positive":
        explanation = "Positive because " + "; ".join(reasons)
    else:
        explanation = "Neutral because " + "; ".join(reasons)

    return pd.Series(
        {
            "Sentiment Score": score,
            "Sentiment Label": label,
            "Sentiment Reason": explanation,
        }
    )


def add_sentiment_analysis(event_data: pd.DataFrame) -> pd.DataFrame:
    sentiment = event_data.apply(analyze_event_sentiment, axis=1)
    return pd.concat([event_data, sentiment], axis=1)


if __name__ == "__main__":
    event_data = load_event_data()
    sentiment_data = add_sentiment_analysis(event_data)

    print(
        sentiment_data[
            ["Event", "Accept/Reject", "Sentiment Score", "Sentiment Label", "Sentiment Reason"]
        ]
    )
    sentiment_data.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved sentiment results to: {OUTPUT_PATH}")
