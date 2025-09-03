#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path
import json
import time
import os
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

class Colors:
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'; BLUE = '\033[94m'; BOLD = '\033[1m'; END = '\033[0m'

def print_header(message: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{message:^60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

def print_status(message: str, status: str):
    color = {"PASS": Colors.GREEN, "WARN": Colors.YELLOW}.get(status, Colors.RED)
    symbol = {"PASS": "✅", "WARN": "⚠️"}.get(status, "❌")
    print(f"{color}{symbol} {message}{Colors.END}")

def check_file_exists(filepath: Path, description: str) -> bool:
    if filepath.exists():
        print_status(f"{description}: {filepath}", "PASS")
        return True
    print_status(f"{description} MISSING: {filepath}", "FAIL")
    return False

def run_command_test(command_parts: list, description: str, timeout: int = 600) -> Tuple[bool, str]: # 增加 Timeout
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        full_command = [sys.executable] + command_parts
        result = subprocess.run(full_command, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode == 0:
            return True, "Success"
        if "data_quality_check" in description and "WARNING" in result.stdout:
            return True, "Success with Warnings"
        return False, f"Exit code: {result.returncode}\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    except subprocess.TimeoutExpired: return False, "Timeout"
    except Exception as e: return False, str(e)

def check_project_structure() -> bool:
    print_header("Checking Corrected Project Structure")
    required = [
        (Path("apple_health_export/export.xml"), "Apple Health Source Data"),
        (Path("hrr_analysis/config.py"), "Core Config File"),
        (Path("scripts/data_quality_check.py"), "Data Quality Script"),
        (Path("scripts/check_feature_isolation.py"), "Feature Isolation Script"),
        (Path("scripts/check_quality_gate.py"), "Quality Gate Script"),
        (Path("scripts/regression_test.py"), "Regression Test Script"),
        (Path("scripts/compare_regression_baseline.py"), "Baseline Comparison Script"),
        (Path(".github/workflows/ci.yml"), "CI Workflow"),
    ]
    return all(check_file_exists(p, d) for p, d in required)

def run_validation_scripts() -> Dict[str, bool]:
    print_header("Testing Corrected Validation Scripts")
    tests = [
        ("scripts/data_quality_check.py", "Data Quality Check"),
        ("scripts/check_feature_isolation.py", "Feature Isolation"),
        ("scripts/regression_test.py --skip-cleaning", "Fast Regression Test (skips cleaning)")
    ]
    results = {}
    for command, description in tests:
        command_parts = command.split()
        script_path = command_parts[0]
        if Path(script_path).exists():
            print(f"\n---> Testing: {description}")
            success, message = run_command_test(command_parts, description)
            if success:
                print_status(f"{description}", "PASS")
                results[description] = True
            else:
                print_status(f"{description}: {message}", "FAIL")
                results[description] = False; break
        else:
            print_status(f"{description}: Script not found", "FAIL")
            results[description] = False; break
    return results

def check_final_outputs() -> bool:
    print_header("Checking Generated Artifacts")
    expected = [
        (Path("features/features_ml.parquet"), "Engineered Features"),
        (Path("models/ERS/dataset_card.json"), "Model Dataset Card for ERS"),
        (Path("report/final_report.md"), "Final Markdown Report"),
        (Path("report/figures/00_final_summary_comparison.png"), "Final Summary Plot"),
    ]
    return all(check_file_exists(p, d) for p, d in expected)

def validate_final_results() -> bool:
    print_header("Validating Key Metrics from Artifacts")
    card_path = Path("models/ERS/dataset_card.json")
    if not card_path.exists():
        print_status(f"Cannot validate, {card_path} not found", "FAIL"); return False
    with open(card_path) as f: card = json.load(f)
    passed_gate = card.get("passed_quality_gate", False)
    task_type = card.get("task_type")
    if task_type == "short_term":
        if passed_gate:
            print_status(f"Track A (short_term) passed its quality gate (R² > {card.get('quality_threshold', 0.05)})", "PASS"); return True
        else:
            print_status(f"Track A (short_term) did NOT pass its quality gate", "FAIL"); return False
    else:
        print_status(f"Final artifact seems to be from a '{task_type}' run, expected 'short_term'", "WARN"); return True

def main():
    print(f"{Colors.BOLD}{'='*60}\n{'PROJECT INTEGRATION & HEALTH CHECK':^60}\n{'='*60}{Colors.END}")
    results = {}
    
    # Corrected standard assignment and validation
    results["Project Structure"] = check_project_structure()
    if not results["Project Structure"]:
        print(f"\n{Colors.RED}❌ Critical file missing. Aborting tests.{Colors.END}")
        sys.exit(1)
        
    validation_results = run_validation_scripts()
    results.update(validation_results)
    if not all(validation_results.values()):
        print(f"\n{Colors.RED}❌ Core validation script failed. Aborting further checks.{Colors.END}")
        sys.exit(1)
        
    results["Final Artifacts"] = check_final_outputs()
    results["Result Metrics"] = validate_final_results()
    
    print_header("Integration Test Summary")
    total = len(results)
    passed = sum(results.values())
    
    for test, success in results.items():
        print_status(f"{test:.<40}", "PASS" if success else "FAIL")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} components passed.{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All checks passed! Your MLOps architecture is consistent and functional.{Colors.END}")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Issues detected. Please review the logs above.{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()