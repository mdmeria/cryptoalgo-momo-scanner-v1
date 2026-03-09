"""
Live Trading System Startup
Coordinates momentum scanning, trade placement, and exit monitoring
"""

import subprocess
import time
import os
import sys
import argparse
from pathlib import Path


def run_live_system(scan_interval: int = 5, monitor_interval: int = 1, max_trades: int = 5):
    """
    Start the complete live trading system with three components:
    1. Momentum scanner (finds new passes)
    2. Trade manager (places trades from passes)
    3. Exit monitor (checks for SL/TP hits)
    
    All run in parallel.
    """
    
    print("=" * 80)
    print("MOMENTUM TRADE SYSTEM - STARTUP")
    print("=" * 80)
    print()
    
    # Setup paths
    script_dir = Path(__file__).parent
    cwd = str(script_dir)
    
    # Configuration
    momo_passes_file = str(script_dir / "final_momo_passes.csv")
    trade_log = str(script_dir / "live_trades.csv")
    
    print(f"Working directory: {cwd}")
    print(f"Momentum passes: {momo_passes_file}")
    print(f"Trade log: {trade_log}")
    print()
    
    # Verify input file exists
    if not Path(momo_passes_file).exists():
        print(f"✗ Error: {momo_passes_file} not found")
        print("Run scanner first to generate momentum passes")
        sys.exit(1)
    
    # Clear previous trade log if starting fresh
    if Path(trade_log).exists():
        response = input(f"Trade log exists: {trade_log}. Append to it? (y/n): ")
        if response.lower() != 'y':
            print("Starting with fresh trade log")
            Path(trade_log).unlink()
    
    print()
    print("Starting live trading system with 3 components...")
    print(f"  1. Trade Manager (scan interval: {scan_interval} min, max {max_trades} trades/scan)")
    print(f"  2. Exit Monitor (check interval: {monitor_interval} min)")
    print()
    print("Press Ctrl+C to stop any component")
    print("=" * 80)
    print()
    
    processes = []
    
    try:
        # Start Trade Manager
        print("[STARTING] Trade Manager...")
        trade_mgr_proc = subprocess.Popen(
            [
                sys.executable, "-u",
                str(script_dir / "live_trade_manager.py"),
                "--input", momo_passes_file,
                "--output", trade_log,
                "--interval-min", str(scan_interval),
                "--max-trades", str(max_trades)
            ],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("Trade Manager", trade_mgr_proc))
        print(f"  PID: {trade_mgr_proc.pid}")
        time.sleep(2)
        
        # Start Exit Monitor
        print("[STARTING] Exit Monitor...")
        monitor_proc = subprocess.Popen(
            [
                sys.executable, "-u",
                str(script_dir / "live_trade_monitor.py"),
                "--trade-log", trade_log
            ],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("Exit Monitor", monitor_proc))
        print(f"  PID: {monitor_proc.pid}")
        time.sleep(2)
        
        print()
        print("=" * 80)
        print("LIVE TRADING SYSTEM RUNNING")
        print("=" * 80)
        print()
        print("Component outputs:")
        print()
        
        # Monitor processes
        last_output = {name: [] for name, _ in processes}
        output_buffers = {name: "" for name, _ in processes}
        
        while True:
            time.sleep(1)
            
            # Check and display output from all processes
            for name, proc in processes:
                if proc.poll() is not None:
                    # Process has terminated
                    print(f"\n[WARNING] {name} has exited (code: {proc.returncode})")
                    # Try to collect remaining output
                    try:
                        remaining, _ = proc.communicate(timeout=1)
                        if remaining:
                            print(remaining)
                    except:
                        pass
                
                # Read available output (non-blocking)
                try:
                    line = proc.stdout.readline()
                    if line:
                        print(f"[{name}] {line.rstrip()}")
                except:
                    pass
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("SHUTTING DOWN LIVE TRADING SYSTEM")
        print("=" * 80)
        
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}")
                    proc.kill()
        
        print("\nSystem stopped")
        print(f"Trade results saved to: {trade_log}")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        for name, proc in processes:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Momentum Trading System Startup")
    parser.add_argument("--scan-interval", type=int, default=5, help="Momentum scan interval (minutes)")
    parser.add_argument("--monitor-interval", type=int, default=1, help="Price monitor interval (minutes)")
    parser.add_argument("--max-trades", type=int, default=5, help="Max new trades per scan")
    
    args = parser.parse_args()
    
    run_live_system(
        scan_interval=args.scan_interval,
        monitor_interval=args.monitor_interval,
        max_trades=args.max_trades
    )
