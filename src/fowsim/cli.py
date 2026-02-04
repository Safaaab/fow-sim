"""Enhanced CLI for Future of Work Simulator."""
from __future__ import annotations

import argparse
import logging
import warnings
import os
from pathlib import Path

# Suppress all warnings for clean CLI output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fowsim.config.settings import Settings
from fowsim.data.pipeline import build_panel_dataset
from fowsim.models.train import train_and_backtest
from fowsim.simulation.simulator import run_simulation


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _cmd_build_data(args: argparse.Namespace) -> None:
    """Build panel dataset from World Bank and other sources."""
    s = Settings()
    logger.info(f"Building dataset: {args.start_year}-{args.end_year}")
    
    build_panel_dataset(
        settings=s,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    
    print(f"\nDataset built successfully!")
    print(f"Location: {s.paths.processed_panel}")
    print(f"Preview your data: streamlit run src/fowsim/ui/streamlit_app.py")


def _cmd_train(args: argparse.Namespace) -> None:
    """Train forecasting models with backtesting."""
    s = Settings()
    horizons = [int(h) for h in args.horizons]
    
    logger.info(f"Training models for horizons: {horizons}")
    
    train_and_backtest(
        settings=s,
        horizons=horizons,
        include_baseline=args.include_baseline,
        include_ensemble=args.include_ensemble
    )
    
    print(f"\nTraining complete!")
    print(f"Metrics: {s.paths.backtest_metrics}")
    print(f"Forecasts: {s.paths.processed_forecasts}")
    print(f"View results: streamlit run src/fowsim/ui/streamlit_app.py")


def _cmd_simulate(args: argparse.Namespace) -> None:
    """Run scenario simulation."""
    s = Settings()
    
    logger.info(f"Running simulation: {args.country} - {args.scenario} - {args.horizon}y")
    
    result_path = run_simulation(
        settings=s,
        country=args.country,
        scenario_name=args.scenario,
        horizon=int(args.horizon),
    )
    
    print(f"\nSimulation complete!")
    print(f"Results: {result_path}")


def _cmd_evaluate_ethics(args: argparse.Namespace) -> None:
    """Evaluate model for bias and fairness."""
    import pandas as pd
    from fowsim.models.ethics import generate_bias_report, print_ethics_report
    
    s = Settings()
    
    logger.info("Evaluating ethics and fairness...")
    
    # Load data
    try:
        forecasts = pd.read_parquet(s.paths.processed_forecasts)
        panel = pd.read_parquet(s.paths.processed_panel)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'build-data' and 'train' commands first.")
        return
    
    # Generate report
    report = generate_bias_report(forecasts, panel)
    print_ethics_report(report)
    
    # Save report
    if args.save:
        import json
        output_path = s.paths.data_processed / "ethics_report.json"
        
        report_dict = {
            "data_bias": report.data_bias,
            "prediction_bias": report.prediction_bias,
            "fairness_metrics": {
                "demographic_parity_diff": report.fairness_metrics.demographic_parity_diff,
                "equal_opportunity_diff": report.fairness_metrics.equal_opportunity_diff,
                "disparate_impact": report.fairness_metrics.disparate_impact,
                "statistical_parity": report.fairness_metrics.statistical_parity,
            },
            "recommendations": report.recommendations
        }
        
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Report saved: {output_path}")


def _cmd_compare_models(args: argparse.Namespace) -> None:
    """Compare model performance across targets and horizons."""
    import pandas as pd
    
    s = Settings()
    
    try:
        metrics = pd.read_csv(s.paths.backtest_metrics)
    except FileNotFoundError:
        print("No metrics found. Run 'train' command first.")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT")
    print("="*80)
    
    # Overall best models
    print("\n🏆 BEST MODELS BY TARGET & HORIZON")
    print("-"*80)
    
    best = metrics.groupby(["target", "horizon"]).apply(
        lambda g: g.loc[g["rmse"].idxmin()]
    ).reset_index(drop=True)
    
    print(best[["target", "horizon", "model", "rmse", "mae"]].to_string(index=False))
    
    # Average performance by model
    print("\nAVERAGE PERFORMANCE BY MODEL")
    print("-"*80)
    
    avg_perf = metrics.groupby("model").agg({
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"]
    }).round(3)
    
    print(avg_perf.to_string())
    
    # Performance by horizon
    print("\nPERFORMANCE BY HORIZON")
    print("-"*80)
    
    by_horizon = metrics.groupby("horizon").agg({
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"]
    }).round(3)
    
    print(by_horizon.to_string())
    
    print("\n" + "="*80 + "\n")


def _cmd_run_dashboard(args: argparse.Namespace) -> None:
    """Launch Streamlit dashboard."""
    import subprocess
    import sys
    
    s = Settings()
    app_path = s.root / "src" / "fowsim" / "ui" / "streamlit_app.py"
    
    print(f"Launching dashboard...")
    print(f"URL: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.headless else "false"
    ])


def _cmd_run_api(args: argparse.Namespace) -> None:
    """Launch FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("FastAPI/Uvicorn not installed. Install with: pip install -e '.[api]'")
        return
    
    print(f"Launching API server...")
    print(f"URL: http://localhost:{args.port}")
    print(f"Docs: http://localhost:{args.port}/docs")
    print(f"Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "fowsim.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def _cmd_full_pipeline(args: argparse.Namespace) -> None:
    """Run complete pipeline: data -> train -> evaluate."""
    print("\n" + "="*80)
    print("RUNNING FULL PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Build data
    print("Step 1/3: Building dataset...")
    _cmd_build_data(args)
    
    # Step 2: Train models
    print("\nStep 2/3: Training models...")
    _cmd_train(args)
    
    # Step 3: Evaluate ethics
    print("\nStep 3/3: Evaluating ethics...")
    _cmd_evaluate_ethics(args)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  • View dashboard: fowsim run-dashboard")
    print("  • Start API: fowsim run-api")
    print("  • Compare models: fowsim compare-models")


def build_parser() -> argparse.ArgumentParser:
    """Build enhanced CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="fowsim",
        description="Future of Work Simulator - AI-powered workforce prediction"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-data command
    p_data = sub.add_parser("build-data", help="Build panel dataset from data sources")
    p_data.add_argument("--start-year", type=int, default=2000, help="Start year for data collection")
    p_data.add_argument("--end-year", type=int, default=2024, help="End year for data collection")
    p_data.set_defaults(func=_cmd_build_data)

    # train command
    p_train = sub.add_parser("train", help="Train forecasting models with backtesting")
    p_train.add_argument("--horizons", nargs="+", default=["5", "10", "20"], help="Forecast horizons (years)")
    p_train.add_argument("--include-baseline", action="store_true", default=True, help="Include baseline models (ARIMA/ETS)")
    p_train.add_argument("--include-ensemble", action="store_true", default=False, help="Create ensemble models")
    p_train.set_defaults(func=_cmd_train)

    # simulate command
    p_sim = sub.add_parser("simulate", help="Run scenario simulation")
    p_sim.add_argument("--country", required=True, help="ISO3 country code (e.g., PAK, USA)")
    p_sim.add_argument("--scenario", required=True, help="Scenario name (e.g., rapid_ai, baseline)")
    p_sim.add_argument("--horizon", required=True, help="Forecast horizon (5, 10, or 20)")
    p_sim.set_defaults(func=_cmd_simulate)

    # evaluate-ethics command
    p_ethics = sub.add_parser("evaluate-ethics", help="Evaluate model for bias and fairness")
    p_ethics.add_argument("--save", action="store_true", help="Save report to JSON")
    p_ethics.set_defaults(func=_cmd_evaluate_ethics)

    # compare-models command
    p_compare = sub.add_parser("compare-models", help="Compare model performance")
    p_compare.set_defaults(func=_cmd_compare_models)

    # run-dashboard command
    p_dashboard = sub.add_parser("run-dashboard", help="Launch Streamlit dashboard")
    p_dashboard.add_argument("--port", type=int, default=8501, help="Port number")
    p_dashboard.add_argument("--headless", action="store_true", help="Run in headless mode")
    p_dashboard.set_defaults(func=_cmd_run_dashboard)

    # run-api command
    p_api = sub.add_parser("run-api", help="Launch FastAPI server")
    p_api.add_argument("--host", default="0.0.0.0", help="Host address")
    p_api.add_argument("--port", type=int, default=8000, help="Port number")
    p_api.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_api.set_defaults(func=_cmd_run_api)

    # full-pipeline command
    p_pipeline = sub.add_parser("full-pipeline", help="Run complete pipeline (data + train + evaluate)")
    p_pipeline.add_argument("--start-year", type=int, default=2000)
    p_pipeline.add_argument("--end-year", type=int, default=2024)
    p_pipeline.add_argument("--horizons", nargs="+", default=["5", "10", "20"])
    p_pipeline.add_argument("--include-baseline", action="store_true", default=True)
    p_pipeline.add_argument("--include-ensemble", action="store_true", default=False)
    p_pipeline.add_argument("--save", action="store_true", default=True)
    p_pipeline.set_defaults(func=_cmd_full_pipeline)

    return p


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
