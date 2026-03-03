#!/usr/bin/env python3
"""
Benchmark Report Generator
Usage: python3 generate_report.py <test_run_directory>
Example: python3 generate_report.py benchmark_results/20260211_013045_light
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

def generate_report(test_dir):
    """Generate a formatted report from test results"""
    
    # Load metadata
    metadata_file = os.path.join(test_dir, 'test_metadata.json')
    if not os.path.exists(metadata_file):
        print(f"❌ Error: {metadata_file} not found")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Report header
    print("=" * 80)
    print("BENCHMARK REPORT".center(80))
    print("=" * 80)
    print()
    
    # Test information
    print("📋 Test Information")
    print("-" * 80)
    print(f"Test Run ID      : {metadata['test_run_id']}")
    print(f"Test Level       : {metadata['test_level'].upper()}")
    print(f"Model            : {metadata['model']}")
    print(f"TP Degree        : {metadata['tp_degree']}")
    print(f"Start Time       : {metadata['start_time']}")
    
    # End time and duration (if available)
    if 'end_time' in metadata:
        print(f"End Time         : {metadata['end_time']}")
        
        # Calculate duration
        try:
            start = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(metadata['end_time'].replace('Z', '+00:00'))
            duration = end - start
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Duration         : {hours}h {minutes}m {seconds}s")
        except:
            pass
    else:
        print(f"End Time         : In Progress...")
        print(f"Duration         : N/A (test still running)")
    
    print()
    
    # Test summary
    print("📊 Test Summary")
    print("-" * 80)
    print(f"Total Tests      : {metadata.get('total_tests', len(metadata['tests']))}")
    print(f"✅ Successful    : {metadata.get('successful_tests', len([t for t in metadata['tests'] if t['status'] == 'SUCCESS']))}")
    print(f"❌ Failed        : {metadata.get('failed_tests', len([t for t in metadata['tests'] if t['status'] == 'FAILED']))}")
    
    total = metadata.get('total_tests', len(metadata['tests']))
    successful = metadata.get('successful_tests', len([t for t in metadata['tests'] if t['status'] == 'SUCCESS']))
    success_rate = (successful / total * 100) if total > 0 else 0
    print(f"Success Rate     : {success_rate:.1f}%")
    print()
    
    # Detailed results
    print("=" * 80)
    print("DETAILED RESULTS".center(80))
    print("=" * 80)
    print()
    
    successful_tests = [t for t in metadata['tests'] if t['status'] == 'SUCCESS']
    failed_tests = [t for t in metadata['tests'] if t['status'] == 'FAILED']
    
    if successful_tests:
        print("✅ Successful Tests")
        print("-" * 80)
        print()
        
        for i, test in enumerate(successful_tests, 1):
            config = test['config']
            results = test.get('results', {})
            
            print(f"{i}. {test['test_name']}")
            print(f"   Configuration:")
            print(f"      Batch Size       : {config['batch_size']}")
            print(f"      Context Length   : {config['context_length']}")
            print(f"      Input Length     : {config['input_length']}")
            print(f"      Output Length    : {config['output_length']}")
            print(f"      Concurrency      : {config['concurrency']}")
            print(f"   Performance:")
            print(f"      Throughput       : {format_number(results.get('throughput_tokens_per_sec'))} tokens/sec")
            print(f"      Mean TTFT        : {format_number(results.get('mean_ttft_ms'))} ms")
            print(f"      Mean TPOT        : {format_number(results.get('mean_tpot_ms'))} ms")
            print(f"      Median TTFT      : {format_number(results.get('median_ttft_ms'))} ms")
            print(f"      P99 TTFT         : {format_number(results.get('p99_ttft_ms'))} ms")
            print()
    
    if failed_tests:
        print("❌ Failed Tests")
        print("-" * 80)
        print()
        
        for i, test in enumerate(failed_tests, 1):
            config = test['config']
            print(f"{i}. {test['test_name']}")
            print(f"   Configuration:")
            print(f"      Batch Size       : {config['batch_size']}")
            print(f"      Context Length   : {config['context_length']}")
            print(f"      Input Length     : {config['input_length']}")
            print(f"      Output Length    : {config['output_length']}")
            print(f"      Concurrency      : {config['concurrency']}")
            print(f"   Error: {test.get('error', 'Unknown error')}")
            print()
    
    # Performance comparison by batch size
    if successful_tests:
        print("=" * 80)
        print("PERFORMANCE COMPARISON".center(80))
        print("=" * 80)
        print()
        
        # Group by batch size
        by_batch_size = {}
        for test in successful_tests:
            bs = test['config']['batch_size']
            if bs not in by_batch_size:
                by_batch_size[bs] = []
            by_batch_size[bs].append(test)
        
        print("📊 By Batch Size")
        print("-" * 80)
        print(f"{'Batch':<8} {'Avg Throughput':<18} {'Avg TTFT':<15} {'Avg TPOT':<15}")
        print(f"{'Size':<8} {'(tokens/sec)':<18} {'(ms)':<15} {'(ms)':<15}")
        print("-" * 80)
        
        for bs in sorted(by_batch_size.keys()):
            tests = by_batch_size[bs]
            avg_throughput = sum(t['results']['throughput_tokens_per_sec'] for t in tests if t['results'].get('throughput_tokens_per_sec')) / len(tests)
            avg_ttft = sum(t['results']['mean_ttft_ms'] for t in tests if t['results'].get('mean_ttft_ms')) / len(tests)
            avg_tpot = sum(t['results']['mean_tpot_ms'] for t in tests if t['results'].get('mean_tpot_ms')) / len(tests)
            
            print(f"{bs:<8} {format_number(avg_throughput):<18} {format_number(avg_ttft):<15} {format_number(avg_tpot):<15}")
        
        print()
        
        # Group by concurrency
        by_concurrency = {}
        for test in successful_tests:
            conc = test['config']['concurrency']
            if conc not in by_concurrency:
                by_concurrency[conc] = []
            by_concurrency[conc].append(test)
        
        print("📊 By Concurrency")
        print("-" * 80)
        print(f"{'Concurrency':<12} {'Avg Throughput':<18} {'Avg TTFT':<15} {'Avg TPOT':<15}")
        print(f"{'':<12} {'(tokens/sec)':<18} {'(ms)':<15} {'(ms)':<15}")
        print("-" * 80)
        
        for conc in sorted(by_concurrency.keys()):
            tests = by_concurrency[conc]
            avg_throughput = sum(t['results']['throughput_tokens_per_sec'] for t in tests if t['results'].get('throughput_tokens_per_sec')) / len(tests)
            avg_ttft = sum(t['results']['mean_ttft_ms'] for t in tests if t['results'].get('mean_ttft_ms')) / len(tests)
            avg_tpot = sum(t['results']['mean_tpot_ms'] for t in tests if t['results'].get('mean_tpot_ms')) / len(tests)
            
            print(f"{conc:<12} {format_number(avg_throughput):<18} {format_number(avg_ttft):<15} {format_number(avg_tpot):<15}")
        
        print()
    
    # Key findings
    if successful_tests:
        print("=" * 80)
        print("KEY FINDINGS".center(80))
        print("=" * 80)
        print()
        
        # Best throughput
        best_throughput = max(successful_tests, key=lambda t: t['results'].get('throughput_tokens_per_sec', 0))
        print(f"🏆 Best Throughput:")
        print(f"   {best_throughput['test_name']}")
        print(f"   {format_number(best_throughput['results']['throughput_tokens_per_sec'])} tokens/sec")
        print()
        
        # Best latency (lowest TTFT)
        best_latency = min(successful_tests, key=lambda t: t['results'].get('mean_ttft_ms', float('inf')))
        print(f"⚡ Best Latency (Lowest TTFT):")
        print(f"   {best_latency['test_name']}")
        print(f"   {format_number(best_latency['results']['mean_ttft_ms'])} ms")
        print()
        
        # Most efficient (best TPOT)
        best_tpot = min(successful_tests, key=lambda t: t['results'].get('mean_tpot_ms', float('inf')))
        print(f"🎯 Most Efficient (Lowest TPOT):")
        print(f"   {best_tpot['test_name']}")
        print(f"   {format_number(best_tpot['results']['mean_tpot_ms'])} ms")
        print()
    
    # Footer
    print("=" * 80)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {test_dir}")
    print("=" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_report.py <test_run_directory>")
        print("Example: python3 generate_report.py benchmark_results/20260211_013045_light")
        print()
        
        # Try to find latest test
        if os.path.exists('benchmark_results'):
            dirs = [d for d in os.listdir('benchmark_results') if os.path.isdir(os.path.join('benchmark_results', d))]
            if dirs:
                latest = sorted(dirs)[-1]
                print(f"Latest test found: benchmark_results/{latest}")
                print(f"Run: python3 generate_report.py benchmark_results/{latest}")
        sys.exit(1)
    
    test_dir = sys.argv[1]
    
    if not os.path.exists(test_dir):
        print(f"❌ Error: Directory '{test_dir}' not found")
        sys.exit(1)
    
    generate_report(test_dir)

if __name__ == '__main__':
    main()
