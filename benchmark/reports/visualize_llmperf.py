#!/usr/bin/env python3

import sys
import os
import json
import glob
import re
import pandas as pd
import plotly.express as px
from jinja2 import Environment, FileSystemLoader

# AWS Instance Hourly Cost (On-Demand) - 수정 가능
# 예: trn2.48xlarge 의 대략적인 시간당 비용 ($44.70)
INSTANCE_HOURLY_COST = 44.70

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
        .plot-container { display: flex; flex-wrap: wrap; gap: 2rem; margin-bottom: 2rem; }
        .plot { flex: 1 1 calc(50% - 1rem); min-width: 600px; background-color: #fff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 0 10px rgba(0,0,0,0.05); }
        .plot-full { width: 100%; background-color: #fff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 0 10px rgba(0,0,0,0.05); margin-bottom: 2rem; }
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
        <div class="table-responsive mb-5">
            {{ summary_table | safe }}
        </div>

        <h2 id="visuals" class="mt-5">Executive & Engineering Visualizations</h2>
        
        <div class="plot-container">
            <div class="plot">
                {{ cost_fig | safe }}
            </div>
            <div class="plot">
                {{ e2e_fig | safe }}
            </div>
        </div>

        <div class="plot-full">
            {{ throughput_fig | safe }}
        </div>
        <div class="plot-container">
            <div class="plot">
                {{ ttft_fig | safe }}
            </div>
            <div class="plot">
                {{ tpot_fig | safe }}
            </div>
        </div>
    </div>
</body>
</html>
"""

def parse_results(results_dir, expected_concurrencies):
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
                "e2e_p50_s": None,
                "cost_per_1k_tokens": None,
                "status": "N/A"
            }

            if conc > bs:
                record["status"] = "SKIPPED"
                data.append(record)
                continue

            test_dir = os.path.join(model_dir, f"llmperf_conc{conc}_in1024_out256")
            summary_files = glob.glob(os.path.join(test_dir, "*_summary.json"))

            if summary_files:
                summary_file = summary_files[0]
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    throughput = summary_data.get("results_mean_output_throughput_token_per_s")
                    
                    # 1K 토큰당 비용 계산 = (초당 서버비용) / (초당 1K 토큰 처리량)
                    cost_per_1k = None
                    if throughput and throughput > 0:
                        cost_per_sec = INSTANCE_HOURLY_COST / 3600
                        cost_per_1k = cost_per_sec / (throughput / 1000)

                    record.update({
                        "throughput_per_s": throughput,
                        "ttft_p50_ms": summary_data.get("results_ttft_s_quantiles_p50", 0) * 1000,
                        "tpot_p50_ms": summary_data.get("results_inter_token_latency_s_quantiles_p50", 0) * 1000,
                        "e2e_p50_s": summary_data.get("results_end_to_end_latency_s_quantiles_p50", 0),
                        "cost_per_1k_tokens": cost_per_1k,
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
    if df.empty:
        return {}

    plots = {}
    df_sorted = df.sort_values(by='concurrency')
    df_success = df_sorted[df_sorted['status'] == 'SUCCESS']

    if not df_success.empty:
        
        # 1. Cost per 1K Tokens (Lower is Better)
        fig_cost = px.line(
            df_success, x='concurrency', y='cost_per_1k_tokens', color='model_config', markers=True,
            title='Cost per 1K Output Tokens ($) (Lower is Better)',
            labels={"concurrency": "Concurrent Requests", "cost_per_1k_tokens": "Cost ($)", "model_config": "Config"}
        )
        plots['cost_fig'] = fig_cost.to_html(full_html=False, include_plotlyjs='cdn')

        # 2. End-to-End Latency (Lower is Better)
        fig_e2e = px.line(
            df_success, x='concurrency', y='e2e_p50_s', color='model_config', markers=True,
            title='End-to-End Latency (Lower is Better)',
            labels={"concurrency": "Concurrent Requests", "e2e_p50_s": "Latency (seconds)", "model_config": "Config"}
        )
        plots['e2e_fig'] = fig_e2e.to_html(full_html=False, include_plotlyjs='cdn')

        # 3. Throughput
        fig_throughput = px.line(
            df_success, x='concurrency', y='throughput_per_s', color='model_config', markers=True,
            title='Throughput (Higher is Better)',
            labels={"concurrency": "Concurrent Requests", "throughput_per_s": "Tokens per Second", "model_config": "Config"}
        )
        plots['throughput_fig'] = fig_throughput.to_html(full_html=False, include_plotlyjs='cdn')

        # 4. TTFT
        fig_ttft = px.line(
            df_success, x='concurrency', y='ttft_p50_ms', color='model_config', markers=True,
            title='Time To First Token (TTFT) (Lower is Better)',
            labels={"concurrency": "Concurrent Requests", "ttft_p50_ms": "Latency (ms)", "model_config": "Config"}
        )
        plots['ttft_fig'] = fig_ttft.to_html(full_html=False, include_plotlyjs='cdn')

        # 5. TPOT
        fig_tpot = px.line(
            df_success, x='concurrency', y='tpot_p50_ms', color='model_config', markers=True,
            title='Inter-Token Latency (ITL/TPOT) (Lower is Better)',
            labels={"concurrency": "Concurrent Requests", "tpot_p50_ms": "Latency (ms)", "model_config": "Config"}
        )
        plots['tpot_fig'] = fig_tpot.to_html(full_html=False, include_plotlyjs='cdn')

    return plots

def format_status(val):
    return f'<span class="status-{val}">{val}</span>'

def generate_html_report(df, plots, output_dir):
    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.from_string(HTML_TEMPLATE)

    if not df.empty:
        df_display = df.copy()
        df_display['status'] = df_display['status'].apply(format_status)
        
        # Rounding for display
        for col in ['throughput_per_s', 'ttft_p50_ms', 'tpot_p50_ms', 'e2e_p50_s']:
            if col in df_display.columns:
                df_display[col] = df_display[col].map('{:.2f}'.format, na_action='ignore')
        if 'cost_per_1k_tokens' in df_display.columns:
            df_display['cost_per_1k_tokens'] = df_display['cost_per_1k_tokens'].map('${:.4f}'.format, na_action='ignore')

        summary_table = df_display.to_html(classes='table table-striped table-hover', index=False, na_rep='N/A', escape=False)
    else:
        summary_table = "<p>No valid llmperf results found to display.</p>"

    report_title = f"LLMPerf Executive Report ({os.path.basename(os.path.dirname(output_dir))})"
    
    rendered_html = template.render(
        report_title=report_title,
        summary_table=summary_table,
        **plots
    )

    report_path = os.path.join(output_dir, "llmperf_executive_report.html")
    with open(report_path, 'w') as f:
        f.write(rendered_html)
    
    print(f"✅ HTML report generated at: {report_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <path_to_llmperf_results_dir>")
        sys.exit(1)

    results_directory = sys.argv[1]
    EXPECTED_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]

    results_df = parse_results(results_directory, EXPECTED_CONCURRENCIES)
    if not results_df.empty:
        plots_html = create_plots(results_df)
        generate_html_report(results_df, plots_html, results_directory)