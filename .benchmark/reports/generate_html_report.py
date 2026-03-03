#!/usr/bin/env python3
"""
Benchmark HTML Report Generator with Charts
Usage: python3 generate_html_report.py <test_run_directory>
Example: python3 generate_html_report.py benchmark_results/20260211_013045_light
"""

import json
import sys
import os
from datetime import datetime

def format_number(value, decimals=2):
    """Format number with proper decimals"""
    if value is None or value == 'null':
        return 'N/A'
    try:
        return f"{float(value):.{decimals}f}"
    except:
        return 'N/A'

def generate_html_report(test_dir):
    """Generate an HTML report with interactive charts"""
    
    # Load metadata
    metadata_file = os.path.join(test_dir, 'test_metadata.json')
    if not os.path.exists(metadata_file):
        print(f"❌ Error: {metadata_file} not found")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    successful_tests = [t for t in metadata['tests'] if t['status'] == 'SUCCESS']
    failed_tests = [t for t in metadata['tests'] if t['status'] == 'FAILED']
    
    # Calculate duration
    duration_str = "N/A"
    try:
        start = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(metadata['end_time'].replace('Z', '+00:00'))
        duration = end - start
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours}h {minutes}m {seconds}s"
    except:
        pass
    
    # Prepare data for charts
    test_names = []
    throughputs = []
    ttfts = []
    tpots = []
    batch_sizes = []
    concurrencies = []
    
    for test in successful_tests:
        test_names.append(test['test_name'])
        throughputs.append(test['results'].get('throughput_tokens_per_sec', 0))
        ttfts.append(test['results'].get('mean_ttft_ms', 0))
        tpots.append(test['results'].get('mean_tpot_ms', 0))
        batch_sizes.append(test['config']['batch_size'])
        concurrencies.append(test['config']['concurrency'])
    
    # Group by batch size
    by_batch_size = {}
    for test in successful_tests:
        bs = test['config']['batch_size']
        if bs not in by_batch_size:
            by_batch_size[bs] = {'throughput': [], 'ttft': [], 'tpot': []}
        by_batch_size[bs]['throughput'].append(test['results'].get('throughput_tokens_per_sec', 0))
        by_batch_size[bs]['ttft'].append(test['results'].get('mean_ttft_ms', 0))
        by_batch_size[bs]['tpot'].append(test['results'].get('mean_tpot_ms', 0))
    
    batch_size_labels = sorted(by_batch_size.keys())
    batch_size_throughputs = [sum(by_batch_size[bs]['throughput'])/len(by_batch_size[bs]['throughput']) for bs in batch_size_labels]
    batch_size_ttfts = [sum(by_batch_size[bs]['ttft'])/len(by_batch_size[bs]['ttft']) for bs in batch_size_labels]
    batch_size_tpots = [sum(by_batch_size[bs]['tpot'])/len(by_batch_size[bs]['tpot']) for bs in batch_size_labels]
    
    # Group by concurrency
    by_concurrency = {}
    for test in successful_tests:
        conc = test['config']['concurrency']
        if conc not in by_concurrency:
            by_concurrency[conc] = {'throughput': [], 'ttft': [], 'tpot': []}
        by_concurrency[conc]['throughput'].append(test['results'].get('throughput_tokens_per_sec', 0))
        by_concurrency[conc]['ttft'].append(test['results'].get('mean_ttft_ms', 0))
        by_concurrency[conc]['tpot'].append(test['results'].get('mean_tpot_ms', 0))
    
    concurrency_labels = sorted(by_concurrency.keys())
    concurrency_throughputs = [sum(by_concurrency[c]['throughput'])/len(by_concurrency[c]['throughput']) for c in concurrency_labels]
    concurrency_ttfts = [sum(by_concurrency[c]['ttft'])/len(by_concurrency[c]['ttft']) for c in concurrency_labels]
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report - {metadata['test_run_id']}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .info-card label {{
            font-weight: bold;
            color: #7f8c8d;
            display: block;
            margin-bottom: 5px;
            font-size: 0.9em;
        }}
        .info-card value {{
            font-size: 1.2em;
            color: #2c3e50;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .summary-card.failed {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        .summary-card h3 {{
            color: white;
            margin: 0 0 10px 0;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        th {{
            background: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}
        .badge.failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .highlight h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .metric {{
            font-size: 1.1em;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .metric strong {{
            color: #3498db;
        }}
        footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Benchmark Report</h1>
        
        <div class="info-grid">
            <div class="info-card">
                <label>Test Run ID</label>
                <value>{metadata['test_run_id']}</value>
            </div>
            <div class="info-card">
                <label>Test Level</label>
                <value>{metadata['test_level'].upper()}</value>
            </div>
            <div class="info-card">
                <label>Model</label>
                <value>{metadata['model']}</value>
            </div>
            <div class="info-card">
                <label>TP Degree</label>
                <value>{metadata['tp_degree']}</value>
            </div>
            <div class="info-card">
                <label>Start Time</label>
                <value>{metadata['start_time']}</value>
            </div>
            <div class="info-card">
                <label>Duration</label>
                <value>{duration_str}</value>
            </div>
        </div>
        
        <h2>📊 Test Summary</h2>
        <div class="summary-cards">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{metadata['total_tests']}</div>
            </div>
            <div class="summary-card success">
                <h3>✅ Successful</h3>
                <div class="value">{metadata['successful_tests']}</div>
            </div>
            <div class="summary-card failed">
                <h3>❌ Failed</h3>
                <div class="value">{metadata['failed_tests']}</div>
            </div>
            <div class="summary-card">
                <h3>Success Rate</h3>
                <div class="value">{(metadata['successful_tests']/metadata['total_tests']*100) if metadata['total_tests'] > 0 else 0:.1f}%</div>
            </div>
        </div>
"""

    if successful_tests:
        # Key Findings
        best_throughput = max(successful_tests, key=lambda t: t['results'].get('throughput_tokens_per_sec', 0))
        best_latency = min(successful_tests, key=lambda t: t['results'].get('mean_ttft_ms', float('inf')))
        best_tpot = min(successful_tests, key=lambda t: t['results'].get('mean_tpot_ms', float('inf')))
        
        html += f"""
        <h2>🏆 Key Findings</h2>
        <div class="highlight">
            <h3>Best Throughput</h3>
            <div class="metric"><strong>{best_throughput['test_name']}</strong></div>
            <div class="metric">{format_number(best_throughput['results']['throughput_tokens_per_sec'])} tokens/sec</div>
        </div>
        <div class="highlight">
            <h3>⚡ Best Latency (Lowest TTFT)</h3>
            <div class="metric"><strong>{best_latency['test_name']}</strong></div>
            <div class="metric">{format_number(best_latency['results']['mean_ttft_ms'])} ms</div>
        </div>
        <div class="highlight">
            <h3>🎯 Most Efficient (Lowest TPOT)</h3>
            <div class="metric"><strong>{best_tpot['test_name']}</strong></div>
            <div class="metric">{format_number(best_tpot['results']['mean_tpot_ms'])} ms</div>
        </div>
        
        <h2>📈 Performance Charts</h2>
        
        <h3>All Tests Comparison</h3>
        <div class="chart-grid">
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
        
        <h3>By Batch Size</h3>
        <div class="chart-grid">
            <div class="chart-container">
                <canvas id="batchSizeThroughputChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="batchSizeLatencyChart"></canvas>
            </div>
        </div>
        
        <h3>By Concurrency</h3>
        <div class="chart-grid">
            <div class="chart-container">
                <canvas id="concurrencyThroughputChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="concurrencyLatencyChart"></canvas>
            </div>
        </div>
"""

    # Detailed Results Table
    html += """
        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Batch Size</th>
                    <th>Context</th>
                    <th>Input</th>
                    <th>Output</th>
                    <th>Concurrency</th>
                    <th>Throughput<br>(tokens/sec)</th>
                    <th>TTFT<br>(ms)</th>
                    <th>TPOT<br>(ms)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for test in metadata['tests']:
        config = test['config']
        status_badge = '<span class="badge success">✅ SUCCESS</span>' if test['status'] == 'SUCCESS' else '<span class="badge failed">❌ FAILED</span>'
        
        if test['status'] == 'SUCCESS':
            results = test['results']
            html += f"""
                <tr>
                    <td>{test['test_name']}</td>
                    <td>{config['batch_size']}</td>
                    <td>{config['context_length']}</td>
                    <td>{config['input_length']}</td>
                    <td>{config['output_length']}</td>
                    <td>{config['concurrency']}</td>
                    <td>{format_number(results.get('throughput_tokens_per_sec'))}</td>
                    <td>{format_number(results.get('mean_ttft_ms'))}</td>
                    <td>{format_number(results.get('mean_tpot_ms'))}</td>
                    <td>{status_badge}</td>
                </tr>
"""
        else:
            html += f"""
                <tr>
                    <td>{test['test_name']}</td>
                    <td>{config['batch_size']}</td>
                    <td>{config['context_length']}</td>
                    <td>{config['input_length']}</td>
                    <td>{config['output_length']}</td>
                    <td>{config['concurrency']}</td>
                    <td colspan="3" style="color: #721c24;">Error: {test.get('error', 'Unknown')}</td>
                    <td>{status_badge}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <footer>
            <p>Report generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p>Data directory: """ + test_dir + """</p>
        </footer>
    </div>
"""

    if successful_tests:
        # Add Chart.js scripts
        html += f"""
    <script>
        // Chart configuration
        const chartColors = {{
            primary: '#3498db',
            success: '#2ecc71',
            warning: '#f39c12',
            danger: '#e74c3c',
            info: '#9b59b6'
        }};
        
        // All Tests - Throughput
        new Chart(document.getElementById('throughputChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(test_names)},
                datasets: [{{
                    label: 'Throughput (tokens/sec)',
                    data: {json.dumps(throughputs)},
                    backgroundColor: chartColors.primary,
                    borderColor: chartColors.primary,
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Throughput by Test',
                        font: {{ size: 16 }}
                    }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Tokens/sec'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45
                        }}
                    }}
                }}
            }}
        }});
        
        // All Tests - Latency
        new Chart(document.getElementById('latencyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(test_names)},
                datasets: [
                    {{
                        label: 'TTFT (ms)',
                        data: {json.dumps(ttfts)},
                        backgroundColor: chartColors.success,
                        borderColor: chartColors.success,
                        borderWidth: 1
                    }},
                    {{
                        label: 'TPOT (ms)',
                        data: {json.dumps(tpots)},
                        backgroundColor: chartColors.warning,
                        borderColor: chartColors.warning,
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Latency by Test',
                        font: {{ size: 16 }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Milliseconds'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45
                        }}
                    }}
                }}
            }}
        }});
        
        // By Batch Size - Throughput
        new Chart(document.getElementById('batchSizeThroughputChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps([f"BS{bs}" for bs in batch_size_labels])},
                datasets: [{{
                    label: 'Avg Throughput (tokens/sec)',
                    data: {json.dumps(batch_size_throughputs)},
                    backgroundColor: chartColors.primary,
                    borderColor: chartColors.primary,
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Average Throughput by Batch Size',
                        font: {{ size: 16 }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Tokens/sec'
                        }}
                    }}
                }}
            }}
        }});
        
        // By Batch Size - Latency
        new Chart(document.getElementById('batchSizeLatencyChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps([f"BS{bs}" for bs in batch_size_labels])},
                datasets: [
                    {{
                        label: 'Avg TTFT (ms)',
                        data: {json.dumps(batch_size_ttfts)},
                        backgroundColor: chartColors.success,
                        borderColor: chartColors.success,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1
                    }},
                    {{
                        label: 'Avg TPOT (ms)',
                        data: {json.dumps(batch_size_tpots)},
                        backgroundColor: chartColors.warning,
                        borderColor: chartColors.warning,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Average Latency by Batch Size',
                        font: {{ size: 16 }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Milliseconds'
                        }}
                    }}
                }}
            }}
        }});
        
        // By Concurrency - Throughput
        new Chart(document.getElementById('concurrencyThroughputChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps([f"Conc {c}" for c in concurrency_labels])},
                datasets: [{{
                    label: 'Avg Throughput (tokens/sec)',
                    data: {json.dumps(concurrency_throughputs)},
                    backgroundColor: chartColors.info,
                    borderColor: chartColors.info,
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Average Throughput by Concurrency',
                        font: {{ size: 16 }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Tokens/sec'
                        }}
                    }}
                }}
            }}
        }});
        
        // By Concurrency - Latency
        new Chart(document.getElementById('concurrencyLatencyChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps([f"Conc {c}" for c in concurrency_labels])},
                datasets: [{{
                    label: 'Avg TTFT (ms)',
                    data: {json.dumps(concurrency_ttfts)},
                    backgroundColor: chartColors.danger,
                    borderColor: chartColors.danger,
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Average TTFT by Concurrency',
                        font: {{ size: 16 }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Milliseconds'
                        }}
                    }}
                }}
            }}
        }});
    </script>
"""

    html += """
</body>
</html>
"""
    
    # Save HTML file
    output_file = os.path.join(test_dir, 'report.html')
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML report generated: {output_file}")
    print(f"📊 Open in browser to view interactive charts")
    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_html_report.py <test_run_directory>")
        print("Example: python3 generate_html_report.py benchmark_results/20260211_013045_light")
        print()
        
        # Try to find latest test
        if os.path.exists('benchmark_results'):
            dirs = [d for d in os.listdir('benchmark_results') if os.path.isdir(os.path.join('benchmark_results', d))]
            if dirs:
                latest = sorted(dirs)[-1]
                print(f"Latest test found: benchmark_results/{latest}")
                print(f"Run: python3 generate_html_report.py benchmark_results/{latest}")
        sys.exit(1)
    
    test_dir = sys.argv[1]
    
    if not os.path.exists(test_dir):
        print(f"❌ Error: Directory '{test_dir}' not found")
        sys.exit(1)
    
    generate_html_report(test_dir)

if __name__ == '__main__':
    main()
