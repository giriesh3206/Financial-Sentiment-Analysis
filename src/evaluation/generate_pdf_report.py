import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib import colors


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def build_report():
    reports_dir = "./reports"
    figures_dir = "./figures"
    output_path = os.path.join(reports_dir, "financial_sentiment_comparison_report.pdf")

    log_path = os.path.join(reports_dir, "experiment_log.csv")
    log_df = load_csv(log_path)

    if log_df.empty:
        raise FileNotFoundError(
            "No experiment_log.csv found. Train RNN, LSTM, and FinBERT on the "
            "official 9,938/2,486 splits before generating the final report."
        )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            spaceAfter=4,
        )
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )

    story = []
    story.append(Paragraph(
        "Financial News Sentiment Prediction using Deep Learning & BERT",
        styles["Title"],
    ))
    story.append(Paragraph(
        "Two-page model comparison report",
        styles["Heading2"],
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Dataset and Methodology", styles["Heading2"]))
    story.append(Paragraph(
        "The project uses the Twitter Financial News Sentiment dataset with "
        "9,938 training instances and 2,486 validation instances. Tweets are "
        "cleaned for URLs, mentions and whitespace while financial tickers are "
        "preserved. The baseline vocabulary is built only from training data. "
        "Two recurrent baselines, Simple RNN and LSTM, are trained and compared "
        "with a fine-tuned ProsusAI/FinBERT model.",
        styles["Small"],
    ))

    story.append(Paragraph("2. Quantitative Comparison", styles["Heading2"]))
    cols = ["Model", "Accuracy", "Macro F1", "Precision", "Recall"]
    rows = [cols]
    for _, row in log_df.iterrows():
        rows.append([
            str(row["model"]),
            f'{float(row["val_accuracy"]):.2%}',
            f'{float(row["val_macro_f1"]):.2%}',
            f'{float(row["val_precision"]):.2%}',
            f'{float(row["val_recall"]):.2%}',
        ])
    table = Table(rows, repeatRows=1, colWidths=[1.2*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. RNN vs LSTM Interpretation", styles["Heading2"]))
    story.append(Paragraph(
        "The Simple RNN and LSTM use the same overall embedding-to-classifier "
        "pipeline, making their validation comparison meaningful. LSTM adds "
        "gated recurrent memory, which is designed to retain useful information "
        "over longer sequences and reduce problems associated with basic recurrent "
        "state updates. The observed metric difference is reported as an empirical "
        "result of this experiment, rather than as proof of a single causal factor.",
        styles["Small"],
    ))

    story.append(Paragraph("4. FinBERT Comparison", styles["Heading2"]))
    story.append(Paragraph(
        "FinBERT starts from a pretrained transformer representation and is "
        "fine-tuned for the three financial sentiment classes. The comparison "
        "table reports its measured validation performance against the recurrent "
        "baselines using the same official validation split.",
        styles["Small"],
    ))

    comparison_fig = os.path.join(figures_dir, "model_comparison.png")
    if os.path.exists(comparison_fig):
        story.append(Image(comparison_fig, width=6.8*inch, height=2.6*inch))

    story.append(PageBreak())

    story.append(Paragraph("5. Class-wise Performance", styles["Heading2"]))
    for model in ["RNN", "LSTM", "FINBERT"]:
        path = os.path.join(
            reports_dir,
            f"classification_report_{model.lower()}.csv",
        )
        df = load_csv(path)
        if not df.empty:
            story.append(Paragraph(model, styles["Heading3"]))
            data = [["Class", "Precision", "Recall", "F1", "Support"]]
            for _, row in df.iterrows():
                data.append([
                    row["class"],
                    f'{float(row["precision"]):.2%}',
                    f'{float(row["recall"]):.2%}',
                    f'{float(row["f1"]):.2%}',
                    str(int(row["support"])),
                ])
            t = Table(data, repeatRows=1, colWidths=[1.3*inch, 1.0*inch, 1.0*inch, 1.0*inch, 0.8*inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]))
            story.append(t)
            story.append(Spacer(1, 4))

    story.append(Paragraph("6. Error Analysis", styles["Heading2"]))
    story.append(Paragraph(
        "The training scripts save representative misclassified validation "
        "examples. These files contain the original tweet, actual label and "
        "predicted label, allowing qualitative discussion of ambiguous financial "
        "language, neutral-versus-directional wording, ticker-heavy text, and "
        "other recurring failure patterns observed in the validation predictions.",
        styles["Small"],
    ))

    error_path = os.path.join(reports_dir, "error_analysis_finbert.csv")
    errors = load_csv(error_path)
    if not errors.empty:
        story.append(Paragraph("Representative FinBERT errors:", styles["Heading3"]))
        for _, row in errors.head(4).iterrows():
            story.append(Paragraph(
                f'<b>Actual:</b> {row["actual"]} &nbsp; '
                f'<b>Predicted:</b> {row["predicted"]}<br/>'
                f'{str(row["text"])[:260]}',
                styles["Small"],
            ))

    story.append(Paragraph("7. Streamlit Application", styles["Heading2"]))
    story.append(Paragraph(
        "The Streamlit dashboard supports RNN, LSTM and FinBERT inference. "
        "Users can enter a financial tweet or headline and receive the predicted "
        "sentiment, confidence, class probabilities and probability bar chart. "
        "The benchmark tab also displays validation class distribution, model "
        "accuracy and macro-F1 comparisons, and confusion matrices.",
        styles["Small"],
    ))

    doc.build(story)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    build_report()
