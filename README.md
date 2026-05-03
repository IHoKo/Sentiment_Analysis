# Event Request Sentiment Analysis

This project analyzes an event management dataset and assigns a sentiment result to each event request. The workflow loads the source CSV, calculates a sentiment score from `-100` to `100`, explains the reason behind that score, exports the enriched results, and provides interactive Plotly visualizations.

## Project Structure

```text
Sentiment_Analysis/
├── Data/
│   ├── EventManagmentDataSet.csv
│   └── event_sentiment_analysis.csv
├── Script/
│   ├── sentiment_analysis.ipynb
│   ├── sentiment_analysis.py
│   └── sentiment_analysis_plots.ipynb
└── README.md
```

## Dataset

The source file is `Data/EventManagmentDataSet.csv`. It contains event booking records with fields such as:

- `Event`
- `Registers`
- `Type`
- `Request for Booking`
- `Request for Cancelling`
- `Room Available/Not Available`
- `Accept/Reject`

The generated output file is `Data/event_sentiment_analysis.csv`.

## What The Analysis Does

The sentiment analysis adds three columns to the original dataset:

- `Sentiment Score`: numeric score between `-100` and `100`
- `Sentiment Label`: `Positive`, `Neutral`, or `Negative`
- `Sentiment Reason`: plain-language explanation for why the row received that score

Example interpretation:

- A rejected event with an unavailable room receives a negative score because the request failed and the required resource was not available.
- An accepted event with an available room and high importance receives a positive score because the request outcome is successful and the event has strong priority signals.

## Scoring Logic

The sentiment score is rule-based and transparent. Each event starts at `0`, then receives positive or negative adjustments based on the event request details.

Positive signals include:

- Request was accepted
- Room was available
- Event was marked `Important` or `Very important`
- Booking was requested
- Registration count was strong or very high
- Event title contained positive activity keywords such as `creative`, `webinar`, `quiz`, `debate`, `review`, or `contest`

Negative signals include:

- Request was rejected
- Room was not available
- Cancellation was requested
- Registration count was low

The final score is capped between `-100` and `100`.

## Current Results

The current dataset contains `15` event records.

Sentiment label summary:

```text
Positive    13
Negative     2
```

Score summary:

```text
Minimum score: -72
Average score: 69.13
Maximum score: 100
```

## Notebooks

### `Script/sentiment_analysis.ipynb`

Main analysis notebook. It loads the event dataset, applies the sentiment scoring logic, displays the enriched table, and writes the final CSV output.

### `Script/sentiment_analysis_plots.ipynb`

Visualization notebook. It reads `Data/event_sentiment_analysis.csv` and shows interactive Plotly charts using a dark theme.

Charts included:

- Sentiment label distribution
- Sentiment score by event
- Distribution of sentiment scores
- Sentiment score vs. registration count

## Python Script

`Script/sentiment_analysis.py` contains the same core sentiment logic in script form. It can be run from the project root:

```bash
python3 Script/sentiment_analysis.py
```

Running the script prints the sentiment results and regenerates:

```text
Data/event_sentiment_analysis.csv
```

## Requirements

The project uses:

- `pandas`
- `plotly`
- `jupyter`
- `nbformat`
- `nbclient`
- `nbconvert`

Install the required packages if needed:

```bash
python3 -m pip install pandas plotly jupyter nbformat nbclient nbconvert
```

## How To View The Results

1. Open `Script/sentiment_analysis.ipynb` to see the data loading and scoring workflow.
2. Open `Script/sentiment_analysis_plots.ipynb` to see the interactive Plotly charts.
3. Open `Data/event_sentiment_analysis.csv` to inspect the final enriched dataset.

If the notebook view in VS Code looks stale, close the notebook tab and reopen the file from the Explorer.

## Notes

This is a deterministic, rule-based sentiment analysis. It is intentionally explainable: every score is tied to visible columns in the dataset and each row receives a reason string. This makes the results easy to audit, adjust, and defend.

For a larger real-world project, the next improvement would be to compare this rule-based score with a trained NLP sentiment model or an LLM-based evaluator, then validate the output against human-labeled examples.
