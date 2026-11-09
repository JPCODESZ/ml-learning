"""Draft the note a business owner reads. Jon still has to edit the marked lines."""

from pathlib import Path


def _model_name(name):
    labels = {
        "seasonal_naive": "seasonal naive forecast",
        "last_value": "last-period forecast",
        "residual_boosting": "gradient boosting",
        "holt_winters_add": "Holt-Winters",
        "holt_winters_mul": "Holt-Winters",
        "sarima_lite": "SARIMA",
        "logistic_regression": "logistic regression",
        "gradient_boosting": "gradient boosting",
    }
    return labels.get(name, str(name).replace("_", " "))


def _percent(value):
    return f"{value:.1f}%"


def _number(value):
    return f"{value:,.2f}"


def _forecast_report(result):
    beats = result["beats_baseline"]
    if beats:
        verdict = (
            f"{_model_name(result['selected_model'])} beat the {_model_name(result['baseline_name'])} "
            f"on the held-out {result['frequency']} periods. Average miss "
            f"{_number(result['model_mae'])} versus {_number(result['baseline_mae'])} "
            f"({result['model_mape']:.2f}% versus {result['baseline_mape']:.2f}%)."
        )
        action = (
            f"Use {_model_name(result['recommended_name'])} for the next period "
            f"({result['next_period']}: {_number(result['next_value'])}). "
            "Keep the baseline beside it for one more cycle before you retire the simple rule."
        )
    else:
        verdict = (
            f"The selected method ({_model_name(result['selected_model'])}) did not beat the "
            f"{_model_name(result['baseline_name'])} on the held-out periods. "
            f"Baseline average miss {_number(result['baseline_mae'])} ({result['baseline_mape']:.2f}%). "
            f"Selected method average miss {_number(result['model_mae'])} ({result['model_mape']:.2f}%). "
            "Do not tell the client the fancier model is more accurate."
        )
        action = (
            f"Use the {_model_name(result['baseline_name'])} number for {result['next_period']}: "
            f"{_number(result['next_value'])}."
        )
    lines = [
        f"# Forecast for {result['target']}",
        "",
        "> REVIEW: Confirm this is the decision the client is paying for, and that the date and target columns are right.",
        "",
        "## What we found",
        "",
        f"The file is a {result['frequency']} series. We held out {result['holdout_start']} to {result['holdout_end']}. "
        f"The method was chosen on {result['validation_start']} to {result['validation_end']}, which the holdout never entered.",
        "",
        "## How accurate",
        "",
        verdict,
        "",
        "## What to do next",
        "",
        "> REVIEW: Edit this recommendation in the client's words before you send it.",
        "",
        action,
        "",
        "## What changed in the file",
        "",
    ]
    for change in result["changes"]:
        lines.append(f"- {change['action']}: {change['detail']}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "A one-step check can miss a turn that shows up only when you forecast many periods ahead. "
            "Spikes were kept. This is a practice delivery draft, not a promise about next year.",
            "",
        ]
    )
    return "\n".join(lines)


def _classify_report(result):
    beats = result["beats_baseline"]
    auc = result["test_auc"]
    threshold = result["threshold"]
    holdout = result["holdout_at_threshold"]
    if beats:
        verdict = (
            f"{result['selected_model'].replace('_', ' ')} ranked the held-out rows better than the baselines. "
            f"ROC-AUC {auc[result['selected_model']]:.3f}, versus {auc['simple_rule']:.3f} for the simple rule "
            f"({result['simple_rule']}) and 0.500 for a coin-flip ranking."
        )
        precision = holdout["precision"]
        precision_text = "n/a" if precision is None else f"{precision:.1%}"
        action = (
            f"Act on rows scored at or above {threshold['threshold']}. "
            f"On the holdout that was {holdout['calls']} rows, precision {precision_text}, "
            f"recall {holdout['recall']:.1%}. "
            f"The event rate in the file is {result['event_rate']:.1%}."
        )
    else:
        verdict = (
            f"{result['selected_model'].replace('_', ' ')} did not beat the baseline on the holdout. "
            f"Its ROC-AUC was {auc[result['selected_model']]:.3f}. "
            f"The simple rule ({result['simple_rule']}) scored {auc['simple_rule']:.3f}, "
            "and a ranking with no skill scores 0.500. Do not tell the client the model is better."
        )
        action = "Keep using the simple rule until a model beats it on a fresh holdout."
    driver_lines = [f"- {item['feature']}" for item in result["top_drivers"][:5]] or ["- none stood out"]
    lines = [
        "# Who to act on",
        "",
        "> REVIEW: Confirm the target is the event the client cares about, and that held-out columns are really leakage.",
        "",
        "## What we found",
        "",
        f"{result['rows']} rows. {result['event_rate']:.1%} are the event.",
        "",
        "Cross-validation on the training rows (ROC-AUC, higher is better):",
        "",
        f"- logistic regression: {result['logistic_cv_auc']:.3f}",
        f"- gradient boosting: {result['boosting_cv_auc']:.3f}",
        "",
        f"The validation pick was {result['selected_model'].replace('_', ' ')}.",
        "",
        "## How accurate",
        "",
        verdict,
        "",
        "## What moves the score",
        "",
        *driver_lines,
        "",
        "> REVIEW: Say which of these the client can actually change.",
        "",
        "## What to do next",
        "",
        "> REVIEW: Edit the action so it matches the cost of a wasted contact versus a missed one.",
        "",
        action,
        "",
        "## What changed in the file",
        "",
    ]
    for change in result["changes"]:
        lines.append(f"- {change['action']}: {change['detail']}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "The cutoff was chosen on the training rows. The holdout numbers are the ones to quote. "
            "A score is not a reason to contact someone if the business rule says otherwise.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if result["task"] == "forecast":
        text = _forecast_report(result)
    else:
        text = _classify_report(result)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8")
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Job kit report</title>",
        "<style>body{font-family:Georgia,serif;max-width:40rem;margin:2rem auto;line-height:1.45;color:#1f2933}blockquote{background:#fff7e6;padding:0.8rem 1rem}</style>",
        "</head><body>",
    ]
    for line in text.splitlines():
        if line.startswith("# "):
            html.append(f"<h1>{_escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{_escape(line[3:])}</h2>")
        elif line.startswith("> "):
            html.append(f"<blockquote>{_escape(line[2:])}</blockquote>")
        elif line.startswith("- "):
            html.append(f"<li>{_escape(line[2:])}</li>")
        elif line.strip():
            html.append(f"<p>{_escape(line)}</p>")
    html.append("</body></html>")
    html_path = output_dir / "REPORT.html"
    html_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


def _escape(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
