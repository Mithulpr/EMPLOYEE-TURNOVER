#!/usr/bin/env python
"""
Unified Terminal Reporting and Visualization CLI for Employee Turnover Prediction.
===================================================================================
A research-grade Python CLI entry point that presents all project analytical results,
model comparisons, explainability outputs, and publication-quality figures directly
through the terminal with zero dependency on the web interface.

Usage Examples:
    python cli.py --all                 # Complete analytical workflow + all 10 figures + summary export
    python cli.py                       # Interactive terminal menu mode
    python cli.py --metrics             # Core performance metrics (Accuracy, F1, ROC-AUC, etc.)
    python cli.py --confusion-matrix    # Confusion matrix breakdown (TN, FP, FN, TP)
    python cli.py --curves              # ROC and Precision-Recall curves analysis
    python cli.py --risk                # Turnover risk score distribution and risk tiers
    python cli.py --champions           # Historical champions progression (Top 5 models)
    python cli.py --comparison          # Base paper comparison (Talebi et al. 2025 vs Step 32)
    python cli.py --shap                # Global SHAP feature importance ranking
    python cli.py --ablation            # Feature ablation study (Baseline vs Enriched vs Screened)
    python cli.py --fairness            # Subgroup fairness audit and Wilson confidence intervals
    python cli.py --calibration         # Probability calibration (Reliability diagram)
    python cli.py --predict EMP_HD_001  # Inspect single employee prediction and SHAP waterfall
    python cli.py --export              # Export Markdown summary to reports/summaries/
    python cli.py --interactive         # Display figures interactively in addition to saving
"""

import os
import sys
import argparse

# Ensure UTF-8 console output on Windows platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root and parent workspace to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from reporting.terminal_reports import (
    Colors,
    show_summary_metrics_report,
    show_confusion_matrix_report,
    show_risk_distribution_report,
    show_global_shap_report,
    show_top_champions_report,
    show_base_paper_comparison_report,
    show_ablation_report,
    show_fairness_report,
    show_single_prediction_report
)

from reporting.visualizer import (
    FIGURES_DIR,
    plot_performance_metrics,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_risk_distribution,
    plot_top_five_champions,
    plot_base_paper_comparison,
    plot_global_shap_importance,
    plot_feature_ablation,
    plot_fairness_subgroups,
    plot_calibration_curve,
    generate_all_figures
)

from reporting.summary_exporter import (
    SUMMARIES_DIR,
    export_executive_summary
)

from reporting.individual_explainer import (
    run_individual_audit,
    interactive_employee_review_loop
)


def run_all_analyses(interactive: bool = False):
    """Executes complete reporting suite across all analytical sections."""
    print("\n" + Colors.CYAN + "═" * 78 + Colors.ENDC)
    print(Colors.BOLD + Colors.CYAN + "  EMPLOYEE TURNOVER PREDICTION SYSTEM — COMPLETE ANALYTICAL AUDIT  ".center(78) + Colors.ENDC)
    print(Colors.CYAN + "═" * 78 + Colors.ENDC)

    # 1. Summary Metrics
    show_summary_metrics_report()
    plot_performance_metrics(interactive=interactive)

    # 2. Confusion Matrix
    show_confusion_matrix_report()
    plot_confusion_matrix(interactive=interactive)

    # 3. ROC & PR Curves
    plot_roc_and_pr_curves(interactive=interactive)

    # 4. Risk Distribution
    show_risk_distribution_report()
    plot_risk_distribution(interactive=interactive)

    # 5. Top Champions Progression
    show_top_champions_report()
    plot_top_five_champions(interactive=interactive)

    # 6. Base Paper Comparison
    show_base_paper_comparison_report()
    plot_base_paper_comparison(interactive=interactive)

    # 7. Global SHAP Feature Importance
    show_global_shap_report()
    plot_global_shap_importance(interactive=interactive)

    # 8. Feature Ablation Study
    show_ablation_report()
    plot_feature_ablation(interactive=interactive)

    # 9. Fairness & Subgroup Audit
    show_fairness_report()
    plot_fairness_subgroups(interactive=interactive)

    # 10. Probability Calibration
    plot_calibration_curve(interactive=interactive)

    # Export comprehensive executive summary
    print("\n" + Colors.BOLD + Colors.BLUE + "► EXPORTING EXECUTIVE SUMMARY REPORT" + Colors.ENDC)
    summary_path = export_executive_summary()

    print("\n" + Colors.GREEN + "═" * 78 + Colors.ENDC)
    print(Colors.BOLD + Colors.GREEN + "  AUDIT COMPLETE: ALL ANALYTICAL MODULES & FIGURES VERIFIED  ".center(78) + Colors.ENDC)
    print(Colors.GREEN + "═" * 78 + Colors.ENDC)
    print(f"  • Figures Directory:   {FIGURES_DIR}")
    print(f"  • Summaries Directory: {SUMMARIES_DIR}")
    print(f"  • Executive Report:    {summary_path}\n")


def interactive_menu():
    """Interactive console menu when script is invoked without flags."""
    while True:
        print("\n" + Colors.CYAN + "╔" + "═" * 74 + "╗" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.BOLD + "   EMPLOYEE TURNOVER PREDICTION SYSTEM — TERMINAL REPORTING CONSOLE   " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.DIM + "   Strictly Inference-Only • Frozen Step 32 Champion • N = 594 Holdout    " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "╠" + "═" * 74 + "╣" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [1] Core Performance Metrics (Accuracy, F1, ROC-AUC, etc.) & Fig 1     " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [2] Actual Outcomes & Confusion Matrix (TN, FP, FN, TP) & Fig 2        " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [3] ROC & Precision-Recall Curves (Operating Threshold τ) & Fig 3       " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [4] Turnover Risk Score Distribution & Risk Tiers & Fig 4               " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [5] Top 5 Historical Champions Milestone Error Progression & Fig 5      " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [6] Base Paper Benchmark Comparison (Talebi et al. 2025) & Fig 6        " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [7] Global SHAP Feature Importance Ranking (Top 10 Drivers) & Fig 7     " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [8] Feature Ablation Cross-Validation Study (Arms A, B, C) & Fig 8      " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [9] Subgroup Fairness Audit & 95% Wilson Confidence Intervals & Fig 9   " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [10] Probability Calibration & Reliability Diagram (Brier Score) & Fig10" + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [11] Inspect Individual Employee Case & Local SHAP Waterfall            " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [A] Run Complete End-to-End Analysis (All 10 Figures + Executive Summary)" + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "║" + Colors.ENDC + "  [Q] Exit Console                                                        " + Colors.CYAN + "║" + Colors.ENDC)
        print(Colors.CYAN + "╚" + "═" * 74 + "╝" + Colors.ENDC)

        choice = input(Colors.BOLD + "\nEnter selection (1-11, A, or Q): " + Colors.ENDC).strip().upper()

        if choice == "1":
            show_summary_metrics_report()
            path = plot_performance_metrics()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "2":
            show_confusion_matrix_report()
            path = plot_confusion_matrix()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "3":
            path = plot_roc_and_pr_curves()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "4":
            show_risk_distribution_report()
            path = plot_risk_distribution()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "5":
            show_top_champions_report()
            path = plot_top_five_champions()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "6":
            show_base_paper_comparison_report()
            path = plot_base_paper_comparison()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "7":
            show_global_shap_report()
            path = plot_global_shap_importance()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "8":
            show_ablation_report()
            path = plot_feature_ablation()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "9":
            show_fairness_report()
            path = plot_fairness_subgroups()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "10":
            path = plot_calibration_curve()
            print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {path}")
        elif choice == "11":
            interactive_employee_review_loop()
        elif choice == "A":
            run_all_analyses()
        elif choice == "Q":
            print("\nExiting Terminal Reporting Console. Goodbye.")
            sys.exit(0)
        else:
            print(f"{Colors.FAIL}Invalid selection. Please choose an option from the menu.{Colors.ENDC}")

        input(Colors.DIM + "\nPress Enter to return to main menu..." + Colors.ENDC)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Terminal Reporting and Visualization CLI for Employee Turnover Prediction System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --all               Run complete suite, generate all 10 figures, and export summary
  python cli.py --metrics           Display performance metrics & save Fig 1
  python cli.py --confusion-matrix  Display confusion matrix & save Fig 2
  python cli.py --curves            Display ROC and PR curves & save Fig 3
  python cli.py --risk              Display risk distribution & save Fig 4
  python cli.py --champions         Display top 5 historical champions & save Fig 5
  python cli.py --comparison        Display base paper benchmark comparison & save Fig 6
  python cli.py --shap              Display global SHAP feature importance & save Fig 7
  python cli.py --ablation          Display feature ablation CV findings & save Fig 8
  python cli.py --fairness          Display demographic subgroup fairness audit & save Fig 9
  python cli.py --calibration       Display calibration reliability diagram & save Fig 10
  python cli.py --predict EMP_HD_001 Inspect individual prediction and local SHAP waterfall
  python cli.py --export            Export executive summary to reports/summaries/
  python cli.py                     Interactive menu mode
        """
    )

    parser.add_argument("--all", action="store_true", help="Execute complete analytical workflow, generate all 10 figures and export reports.")
    parser.add_argument("--metrics", action="store_true", help="Display core Step 32 metrics & generate Figure 1.")
    parser.add_argument("--confusion-matrix", action="store_true", help="Display confusion matrix & generate Figure 2.")
    parser.add_argument("--curves", action="store_true", help="Display ROC and PR curves & generate Figure 3.")
    parser.add_argument("--risk", action="store_true", help="Display risk distribution & generate Figure 4.")
    parser.add_argument("--champions", action="store_true", help="Display top 5 historical champions & generate Figure 5.")
    parser.add_argument("--comparison", action="store_true", help="Display Base Paper comparison & generate Figure 6.")
    parser.add_argument("--shap", action="store_true", help="Display global SHAP feature importance & generate Figure 7.")
    parser.add_argument("--ablation", action="store_true", help="Display feature ablation study & generate Figure 8.")
    parser.add_argument("--fairness", action="store_true", help="Display subgroup fairness audit & generate Figure 9.")
    parser.add_argument("--calibration", action="store_true", help="Display probability calibration & generate Figure 10.")
    parser.add_argument("--predict", type=str, metavar="ID_OR_INDEX", help="Inspect single employee case (e.g. EMP_HD_001 or 0-593) and local SHAP waterfall.")
    parser.add_argument("--inspect", action="store_true", help="Launch interactive multi-employee review loop to inspect several cases continuously.")
    parser.add_argument("--export", action="store_true", help="Export Markdown executive analytical summary to reports/summaries/.")
    parser.add_argument("--interactive", action="store_true", help="Show plots interactively in GUI window in addition to saving to disk.")

    args = parser.parse_args()

    # Check if any argument was passed
    has_flag = any([
        args.all, args.metrics, args.confusion_matrix, args.curves,
        args.risk, args.champions, args.comparison, args.shap,
        args.ablation, args.fairness, args.calibration, args.predict, args.inspect, args.export
    ])

    if not has_flag:
        # Launch interactive menu
        interactive_menu()
        return

    if args.all:
        run_all_analyses(interactive=args.interactive)
        return

    if args.inspect:
        interactive_employee_review_loop()
        return

    if args.metrics:
        show_summary_metrics_report()
        p = plot_performance_metrics(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.confusion_matrix:
        show_confusion_matrix_report()
        p = plot_confusion_matrix(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.curves:
        p = plot_roc_and_pr_curves(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.risk:
        show_risk_distribution_report()
        p = plot_risk_distribution(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.champions:
        show_top_champions_report()
        p = plot_top_five_champions(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.comparison:
        show_base_paper_comparison_report()
        p = plot_base_paper_comparison(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.shap:
        show_global_shap_report()
        p = plot_global_shap_importance(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.ablation:
        show_ablation_report()
        p = plot_feature_ablation(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.fairness:
        show_fairness_report()
        p = plot_fairness_subgroups(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.calibration:
        p = plot_calibration_curve(interactive=args.interactive)
        print(f"\n{Colors.GREEN}[SAVED FIGURE]{Colors.ENDC} {p}")

    if args.predict:
        run_individual_audit(args.predict, interactive=args.interactive)

    if args.export:
        p = export_executive_summary()
        print(f"\n{Colors.GREEN}[EXPORTED SUMMARY]{Colors.ENDC} {p}")


if __name__ == "__main__":
    main()
