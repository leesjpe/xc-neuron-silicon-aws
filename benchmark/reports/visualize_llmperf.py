#!/usr/bin/env python3

import sys
import os
import json
import glob
import re
import pandas as pd
import plotly.express as px
from jinja2 import Environment, FileSystemLoader

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 2rem; background-color: #f8f9fa; }
        .container { max-width: 1600px; }
        .plot { margin-bottom: 3rem; background-color: #fff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 0 10px rgba(0,0,0,0.05); }
        h1, h2 { border-bottom: 1px solid #dee2e6; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }
        .table-responsive { max-height: 500px; }
        .status-SUCCESS { color: green; font-weight: bold; }
        .status-FAILED { color: red; font-weight: bold; }
        .status-SKIPPED { color: orange; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">{{ report_title }}</h1>
        
        <h2 id="summary">Summary Table</h2>
        <div class="table-responsive">
            {{ summary_table | safe }}
        </div>

        <h2 id="visuals" class="mt-5">Visualizations</h2>
        <div class="plot">
            {{ throughput_fig | safe }}
        </div>
        <div class="plot">
            {{ ttft_fig | safe }}
        </div>
        <div class="plot">
            {{ tpot_fig | safe }}
        </div>
    </div>
</body>
</html>
"""

def parse_results(results_dir, expected_concurrencies):
    """
    Parses all llmperf summary.json files in a directory, returning a pandas DataFrame.
    It also identifies and marks FAILED and SKIPPED tests.
    """
    data = []
    model_config_dirs = glob.glob(os.path.join(results_dir, "*-tp*-bs*-ctx*"))

    for model_dir in model_config_dirs:
        dir_name = os.path.basename(model_dir)
        
        tp_match = re.search(r'tp(\d+)', dir_name)
        bs_match = re.search(r'bs(\d+)', dir_name)
        ctx_match = re.search(r'ctx(\d+)', dir_name)
        
        if not (tp_match and bs_match and ctx_match):
            continue
            
        tp = int(tp_match.group(1))
        bs = int(bs_match.group(1))
        ctx = int(ctx_match.group(1))

        for conc in expected_concurrencies:
            record = {
                "model_config": dir_name,
                "tp": tp,
                "bs": bs,
                "ctx": ctx,
                "concurrency": conc,
                "throughput_per_s": None,
                "ttft_p50_ms": None,
                "tpot_p50_ms": None,
                "status": "N/A"
            }

            if conc > bs:
                record["status"] = "SKIPPED"
                data.append(record)
                continue

            test_dir = os.path.join(model_dir, f"llmperf_conc{conc}_in1024_out256")
            
            # Look for any summary file, to be robust against llmperf's naming scheme.
            # It could be 'summary.json' (if renamed) or the original messy name.
            summary_files = glob.glob(os.path.join(test_dir, "*_summary.json"))

            if summary_files:
                summary_file = summary_files[0] # Take the first one found
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    record.update({
                        "throughput_per_s": summary_data.get("results_mean_output_throughput_token_per_s"),
                        "ttft_p50_ms": summary_data.get("results_ttft_s_quantiles_p50", 0) * 1000,
                        "tpot_p50_ms": summary_data.get("results_inter_token_latency_s_quantiles_p50", 0) * 1000,
                        "status": "SUCCESS"
                    })
                except (json.JSONDecodeError, KeyError):
                    record["status"] = "FAILED"
            else:
                record["status"] = "FAILED"
            
            data.append(record)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df.sort_values(by=["tp", "bs", "concurrency"]).reset_index(drop=True)

def create_plots(df):
    """
    Generates plotly figures from the benchmark data.
    """
    if df.empty:
        return {}

    plots = {}
    df_sorted = df.sort_values(by='concurrency')
    df_success = df_sorted[df_sorted['status'] == 'SUCCESS']

    if not df_success.empty:
        # Plot 1: Throughput vs. Concurrency
        fig_throughput = px.line(
            df_success, x='concurrency', y='throughput_per_s', color='model_config', markers=True,
            title='LLMPerf: Throughput vs. Concurrency',
            labels={"concurrency": "Concurrency", "throughput_per_s": "Throughput (Output Tokens/sec)", "model_config": "Model Config"}
        )
        fig_throughput.update_layout(xaxis_type='log')
        plots['throughput_fig'] = fig_throughput.to_html(full_html=False, include_plotlyjs='cdn')

        # Plot 2: P50 TTFT vs. Concurrency
        fig_ttft = px.line(
            df_success, x='concurrency', y='ttft_p50_ms', color='model_config', markers=True,
            title='LLMPerf: Median Time to First Token (TTFT) vs. Concurrency',
            labels={"concurrency": "Concurrency", "ttft_p50_ms": "Median TTFT (ms)", "model_config": "Model Config"}
        )
        fig_ttft.update_layout(xaxis_type='log')
        plots['ttft_fig'] = fig_ttft.to_html(full_html=False, include_plotlyjs='cdn')

        # Plot 3: P50 TPOT vs. Concurrency
        fig_tpot = px.line(
            df_success, x='concurrency', y='tpot_p50_ms', color='model_config', markers=True,
            title='LLMPerf: Median Time Per Output Token (TPOT) vs. Concurrency',
            labels={"concurrency": "Concurrency", "tpot_p50_ms": "Median TPOT (ms)", "model_config": "Model Config"}
        )
        fig_tpot.update_layout(xaxis_type='log')
        plots['tpot_fig'] = fig_tpot.to_html(full_html=False, include_plotlyjs='cdn')

    return plots

def format_status(val):
    """Applies CSS class based on status for the HTML table."""
    return f'<span class="status-{val}">{val}</span>'

def generate_html_report(df, plots, output_dir):
    """
    Generates an HTML report from the dataframe and plots.
    """
    # Use absolute path for template loader for robustness
    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.from_string(HTML_TEMPLATE)

    if not df.empty:
        df_display = df.copy()
        df_display['status'] = df_display['status'].apply(format_status)
        
        # Round numeric columns for better readability
        for col in ['throughput_per_s', 'ttft_p50_ms', 'tpot_p50_ms']:
            if col in df_display.columns:
                df_display[col] = df_display[col].map('{:.2f}'.format, na_action='ignore')

        summary_table = df_display.to_html(classes='table table-striped table-hover', index=False, na_rep='N/A', escape=False)
    else:
        summary_table = "<p>No valid llmperf results found to display.</p>"

    report_title = f"LLMPerf Benchmark Report ({os.path.basename(os.path.dirname(output_dir))})"
    
    rendered_html = template.render(
        report_title=report_title,
        summary_table=summary_table,
        **plots
    )

    report_path = os.path.join(output_dir, "llmperf_visual_report.html")
    with open(report_path, 'w') as f:
        f.write(rendered_html)
    
    print(f"✅ HTML report generated at: {report_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <path_to_llmperf_results_dir>")
        print(f"Example: python3 {sys.argv[0]} /home/ubuntu/benchmark_result/llmperf/qwen3-8b/20260225_1024")
        sys.exit(1)

    results_directory = sys.argv[1]
    if not os.path.isdir(results_directory):
        print(f"❌ Error: Directory not found at '{results_directory}'")
        sys.exit(1)

    # This should be read from the config file in a more advanced version
    EXPECTED_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"🔍 Parsing results from: {results_directory}")
    results_df = parse_results(results_directory, EXPECTED_CONCURRENCIES)
    
    if results_df.empty:
        print("⚠️ No results found to visualize.")
    else:
        print("📊 Generating plots...")
        plots_html = create_plots(results_df)
        
        print("📄 Generating HTML report...")
        generate_html_report(results_df, plots_html, results_directory)
