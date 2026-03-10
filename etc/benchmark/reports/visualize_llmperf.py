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
        .table-header { display: flex; justify-content: space-between; align-items: center; }
        .table-responsive { max-height: 600px; }
        th { text-align: center; cursor: pointer; user-select: none; }
        th:hover { background-color: #f2f2f2; }
        td { text-align: center; vertical-align: middle; }
        .table-highlightable tbody tr:hover { background-color: #e9ecef; cursor: pointer; }
        .table-highlightable tbody tr.highlighted { background-color: #cfe2ff; }
        .status-SUCCESS { color: green; font-weight: bold; }
        .status-FAILED { color: red; font-weight: bold; }
        .status-SKIPPED { color: orange; font-weight: bold; }
        .status-COMPILATION_FAILED { color: #808080; font-weight: bold; }
        th.th-sort-asc::after,
        th.th-sort-desc::after {
            content: ' '; display: inline-block; margin-left: 0.5em; border: 5px solid transparent;
        }
        th.th-sort-asc::after { border-bottom-color: #333; }
        th.th-sort-desc::after { border-top-color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">{{ report_title }}</h1>

        <div class="table-header">
            <h2 id="summary">Summary Table</h2>
            <button id="resetSortBtn" class="btn btn-secondary btn-sm">Reset Sort</button>
        </div>
        <div class="table-responsive">
            {{ summary_table | safe }}
        </div>

        <h2 id="visuals" class="mt-5">Visualizations</h2>
        <div id="throughput-plot" class="plot">
            {{ throughput_fig | safe }}
        </div>
        <div id="e2e-latency-plot" class="plot">
            {{ e2e_latency_fig | safe }}
        </div>
        <div id="ttft-plot" class="plot">
            {{ ttft_fig | safe }}
        </div>
        <div id="tpot-plot" class="plot">
            {{ tpot_fig | safe }}
        </div>
        <div id="cost-plot" class="plot">
            {{ cost_fig | safe }}
        </div>
    </div>
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        const table = document.querySelector(".table-sortable");
        if (!table) return;

        const tBody = table.tBodies[0];
        const originalRows = Array.from(tBody.rows);
        let sortCriteria = [];

        function getCellValue(row, colIndex) {
            const cell = row.cells[colIndex];
            if (!cell) return null;
            const text = cell.textContent.trim();
            if (text === 'N/A') return null;
            const num = parseFloat(text.replace(/,/g, ''));
            return isNaN(num) ? text.toLowerCase() : num;
        }

        function sortTable() {
            if (sortCriteria.length === 0) return;

            const sortedRows = Array.from(originalRows).sort((a, b) => {
                for (const criterion of sortCriteria) {
                    const { column, asc } = criterion;
                    const dirModifier = asc ? 1 : -1;
                    const aVal = getCellValue(a, column);
                    const bVal = getCellValue(b, column);

                    if (aVal === null) return 1 * dirModifier;
                    if (bVal === null) return -1 * dirModifier;

                    if (typeof aVal === 'number' && typeof bVal === 'number') {
                        if (aVal < bVal) return -1 * dirModifier;
                        if (aVal > bVal) return 1 * dirModifier;
                    } else {
                        const comparison = String(aVal).localeCompare(String(bVal), undefined, { numeric: true });
                        if (comparison !== 0) return comparison * dirModifier;
                    }
                }
                return 0;
            });

            tBody.append(...sortedRows);
        }

        function updateSortHeaders() {
            table.querySelectorAll("th").forEach(th => th.classList.remove("th-sort-asc", "th-sort-desc"));
            sortCriteria.forEach(criterion => {
                const th = table.querySelector(`th:nth-child(${criterion.column + 1})`);
                if (th) {
                    th.classList.toggle("th-sort-asc", criterion.asc);
                    th.classList.toggle("th-sort-desc", !criterion.asc);
                }
            });
        }

        table.querySelectorAll("th").forEach((headerCell, headerIndex) => {
            headerCell.addEventListener("click", (event) => {
                const existingCriterionIndex = sortCriteria.findIndex(c => c.column === headerIndex);

                if (event.shiftKey) {
                    if (existingCriterionIndex > -1) {
                        sortCriteria[existingCriterionIndex].asc = !sortCriteria[existingCriterionIndex].asc;
                    } else {
                        sortCriteria.push({ column: headerIndex, asc: true });
                    }
                } else {
                    if (existingCriterionIndex > -1) {
                        const currentAsc = sortCriteria[existingCriterionIndex].asc;
                        sortCriteria = [{ column: headerIndex, asc: !currentAsc }];
                    } else {
                        sortCriteria = [{ column: headerIndex, asc: true }];
                    }
                }
                sortTable();
                updateSortHeaders();
            });
        });

        document.getElementById('resetSortBtn').addEventListener('click', () => {
            sortCriteria = [];
            tBody.append(...originalRows);
            table.querySelectorAll("th").forEach(th => th.classList.remove("th-sort-asc", "th-sort-desc"));
        });

        // Initial sort by cost (ascending)
        function performInitialSort() {
            const headers = Array.from(table.querySelectorAll('th'));
            const costColumnIndex = headers.findIndex(th => th.textContent.trim() === 'cost_per_1m_tokens');
            if (costColumnIndex > -1) {
                sortCriteria = [{ column: costColumnIndex, asc: true }];
                sortTable();
                updateSortHeaders();
            }
        }
        performInitialSort();

        // Highlighting functionality
        const plotDivs = document.querySelectorAll('.plot > div');
        let highlightedTraceName = null;

        tBody.addEventListener('click', (event) => {
            const row = event.target.closest('tr');
            if (!row) return;

            const modelConfig = row.cells[0].textContent.trim();
            const isCurrentlyHighlighted = row.classList.contains('highlighted');

            // Clear previous highlights
            tBody.querySelectorAll('tr.highlighted').forEach(r => r.classList.remove('highlighted'));

            if (isCurrentlyHighlighted) {
                // Un-highlight
                highlightedTraceName = null;
                plotDivs.forEach(div => {
                    Plotly.restyle(div, { 'line.width': 2, opacity: 1.0 });
                });
            } else {
                // Highlight new row
                highlightedTraceName = modelConfig;
                row.classList.add('highlighted');
                plotDivs.forEach(div => {
                    const update = {
                        'line.width': Array(div.data.length).fill(0.5),
                        'opacity': Array(div.data.length).fill(0.3)
                    };
                    const traceIndex = div.data.findIndex(trace => trace.name === modelConfig);
                    if (traceIndex !== -1) {
                        update['line.width'][traceIndex] = 4;
                        update['opacity'][traceIndex] = 1.0;
                    }
                    Plotly.restyle(div, update);
                });
            }
        });
    });
    </script>
</body>
</html>
"""

def parse_results(results_dir, expected_concurrencies):
    """
    Parses all llmperf summary.json files in a directory, returning a pandas DataFrame.
    It also identifies and marks FAILED and SKIPPED tests.
    """
    data = []
    model_name = os.path.basename(results_dir)
    compiled_models_base_path = f"/data/compiled_models/{model_name}"

    if not os.path.isdir(compiled_models_base_path):
        print(f"❌ Error: Compiled models directory not found at '{compiled_models_base_path}'.")
        print("   Cannot check for compilation status. Please ensure the path is correct.")
        sys.exit(1)

    # Get all attempted model configurations from the compilation directory
    model_config_dirs = sorted(glob.glob(os.path.join(compiled_models_base_path, "*-tp*-bs*-ctx*")))

    if not model_config_dirs:
        print(f"⚠️ Warning: No compiled model configurations found in '{compiled_models_base_path}'.")
        return pd.DataFrame()

    for compiled_model_path in model_config_dirs:
        dir_name = os.path.basename(compiled_model_path)
        
        tp_match = re.search(r'tp(\d+)', dir_name)
        bs_match = re.search(r'bs(\d+)', dir_name)
        ctx_match = re.search(r'ctx(\d+)', dir_name)
        
        if not (tp_match and bs_match and ctx_match):
            continue

        tp = int(tp_match.group(1))
        bs = int(bs_match.group(1))
        ctx = int(ctx_match.group(1))

        # Check for compilation success
        compile_success_marker = os.path.join(compiled_model_path, ".compile_success")
        is_compiled = os.path.exists(compile_success_marker)

        for conc in expected_concurrencies:
            record = {
                "model_config": dir_name,
                "tp": tp,
                "bs": bs,
                "ctx": ctx,
                "concurrency": conc,
                "throughput_tokens_per_s": None,
                "ttft_p50_ms": None,
                "e2e_latency_p50_ms": None,
                "tpot_p50_ms": None,
                "status": "N/A"
            }

            if not is_compiled:
                record["status"] = "COMPILATION_FAILED"
                data.append(record)
                continue

            if conc > bs:
                record["status"] = "SKIPPED"
                data.append(record)
                continue

            # Path to the actual benchmark results
            model_results_dir = os.path.join(results_dir, dir_name)
            test_dir = os.path.join(model_results_dir, f"llmperf_conc{conc}_in1024_out256")
            
            # Look for any summary file, to be robust against llmperf's naming scheme.
            summary_files = glob.glob(os.path.join(test_dir, "*_summary.json"))

            if summary_files:
                summary_file = summary_files[0] # Take the first one found
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)

                    record.update({
                        "throughput_tokens_per_s": summary_data.get("results_mean_output_throughput_token_per_s"),
                        "ttft_p50_ms": summary_data.get("results_ttft_s_quantiles_p50", 0) * 1000,
                        "e2e_latency_p50_ms": summary_data.get("results_end_to_end_latency_s_quantiles_p50", 0) * 1000,
                        "tpot_p50_ms": summary_data.get("results_inter_token_latency_s_quantiles_p50", 0) * 1000,
                        "status": "SUCCESS"
                    })
                except (json.JSONDecodeError, KeyError):
                    record["status"] = "FAILED"
            else:
                record["status"] = "FAILED" # Test ran but produced no summary.json
            
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
            df_success, x='concurrency', y='throughput_tokens_per_s', color='model_config', markers=True,
            title='LLMPerf: Throughput vs. Concurrency (Higher is Better)',
            labels={"concurrency": "Concurrency", "throughput_tokens_per_s": "Throughput (Output Tokens/sec)", "model_config": "Model Config"}
        )
        fig_throughput.update_xaxes(type='linear', tickvals=df_success['concurrency'].unique())
        plots['throughput_fig'] = fig_throughput.to_html(full_html=False, include_plotlyjs='cdn')

        # Plot 2: P50 End-to-End Latency vs. Concurrency
        fig_e2e = px.line(
            df_success, x='concurrency', y='e2e_latency_p50_ms', color='model_config', markers=True,
            title='LLMPerf: Median End-to-End Latency vs. Concurrency (Lower is Better)',
            labels={"concurrency": "Concurrency", "e2e_latency_p50_ms": "Median E2E Latency (ms)", "model_config": "Model Config"}
        )
        fig_e2e.update_xaxes(type='linear', tickvals=df_success['concurrency'].unique())
        plots['e2e_latency_fig'] = fig_e2e.to_html(full_html=False, include_plotlyjs=False)

        # Plot 3: P50 TTFT vs. Concurrency
        fig_ttft = px.line(
            df_success, x='concurrency', y='ttft_p50_ms', color='model_config', markers=True,
            title='LLMPerf: Median Time to First Token (TTFT) vs. Concurrency (Lower is Better)',
            labels={"concurrency": "Concurrency", "ttft_p50_ms": "Median TTFT (ms)", "model_config": "Model Config"}
        )
        fig_ttft.update_xaxes(type='linear', tickvals=df_success['concurrency'].unique())
        plots['ttft_fig'] = fig_ttft.to_html(full_html=False, include_plotlyjs=False)

        # Plot 4: P50 TPOT vs. Concurrency
        fig_tpot = px.line(
            df_success, x='concurrency', y='tpot_p50_ms', color='model_config', markers=True,
            title='LLMPerf: Median Inter-Token Latency (TPOT) vs. Concurrency (Lower is Better)',
            labels={"concurrency": "Concurrency", "tpot_p50_ms": "Median TPOT (ms)", "model_config": "Model Config"}
        )
        fig_tpot.update_xaxes(type='linear', tickvals=df_success['concurrency'].unique())
        plots['tpot_fig'] = fig_tpot.to_html(full_html=False, include_plotlyjs=False)

        # Plot 5: Cost per 1M Tokens vs. Concurrency
        if 'cost_per_1m_tokens' in df_success.columns:
            fig_cost = px.line(
                df_success, x='concurrency', y='cost_per_1m_tokens', color='model_config', markers=True,
                title='LLMPerf: Cost per 1M Output Tokens vs. Concurrency (Lower is Better)',
                labels={"concurrency": "Concurrency", "cost_per_1m_tokens": "Cost per 1M Tokens ($)", "model_config": "Model Config"}
            )
            fig_cost.update_xaxes(type='linear', tickvals=df_success['concurrency'].unique())
            plots['cost_fig'] = fig_cost.to_html(full_html=False, include_plotlyjs=False)

    return plots

def format_status(val):
    """Applies CSS class based on status for the HTML table."""
    return f'<span class="status-{val}">{val}</span>'

def generate_html_report(df, plots, output_dir, instance_type):
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
        for col in ['throughput_tokens_per_s', 'e2e_latency_p50_ms', 'ttft_p50_ms', 'tpot_p50_ms']:
            if col in df_display.columns:
                df_display[col] = df_display[col].map('{:.2f}'.format, na_action='ignore')
        if 'cost_per_1m_tokens' in df_display.columns:
            df_display['cost_per_1m_tokens'] = df_display['cost_per_1m_tokens'].map('{:.6f}'.format, na_action='ignore')

        summary_table = df_display.to_html(classes='table table-striped table-hover table-sortable table-highlightable', index=False, na_rep='N/A', escape=False)
    else:
        summary_table = "<p>No valid llmperf results found to display.</p>"

    model_name = os.path.basename(output_dir)
    report_title = f"LLMPerf Benchmark Report ({model_name} on {instance_type})"
    
    rendered_html = template.render(
        report_title=report_title,
        summary_table=summary_table,
        **plots
    )

    report_filename = f"llmperf_report_{model_name}_{instance_type}.html"
    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, 'w') as f:
        f.write(rendered_html)
    
    print(f"✅ HTML report generated at: {report_path}")

if __name__ == "__main__":
    # --- Instance Prices (On-Demand, us-east-1, as of Feb 2026) ---
    # This can be expanded with more instance types.
    INSTANCE_PRICES = {
        "trn2.48xlarge": 35.7608,
        "inf2.48xlarge": 12.98,
        # Add other instances here, e.g., "inf2.48xlarge": 12.048
    }

    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <path_to_llmperf_results_dir> <instance_type>")
        print(f"Example: python3 {sys.argv[0]} /home/ubuntu/benchmark_result/llmperf/qwen3-8b trn2.48xlarge")
        print(f"Supported instance types: {', '.join(INSTANCE_PRICES.keys())}")
        sys.exit(1)

    results_directory = sys.argv[1]
    instance_type = sys.argv[2]

    if not os.path.isdir(results_directory):
        print(f"❌ Error: Directory not found at '{results_directory}'")
        sys.exit(1)
    
    if instance_type not in INSTANCE_PRICES:
        print(f"❌ Error: Unsupported instance type '{instance_type}'.")
        print(f"   Supported types are: {', '.join(INSTANCE_PRICES.keys())}")
        sys.exit(1)

    # This should be read from the config file in a more advanced version
    EXPECTED_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]
    hourly_price = INSTANCE_PRICES[instance_type]

    print(f"🔍 Parsing results from: {results_directory}")
    results_df = parse_results(results_directory, EXPECTED_CONCURRENCIES)
    
    if results_df.empty:
        print("⚠️ No results found to visualize.")
    else:
        # --- Calculate Cost per 1M Tokens ---
        if 'throughput_tokens_per_s' in results_df.columns and results_df['throughput_tokens_per_s'].notna().any():
            # Cost per 1M tokens = (price_per_second / tokens_per_second) * 1,000,000
            results_df['cost_per_1m_tokens'] = results_df.apply(
                lambda row: (hourly_price / 3600) / row['throughput_tokens_per_s'] * 1_000_000 if pd.notna(row['throughput_tokens_per_s']) and row['throughput_tokens_per_s'] > 0 else None,
                axis=1
            )

        print("📊 Generating plots...")
        plots_html = create_plots(results_df)
        
        print("📄 Generating HTML report...")
        generate_html_report(results_df, plots_html, results_directory, instance_type)
