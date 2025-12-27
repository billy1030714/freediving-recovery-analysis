"""
validate_ers_final.py - ERS Final Comprehensive Validation
===========================================================
整合所有必要的 ERS 驗證分析：
1. 標準關聯效度（ERS vs HRR60）
2. Train/Test 性能對比與 Overfitting 檢測
3. 完整學習曲線（訓練+驗證雙曲線，使用固定 split）
4. 穩定性比較（變異係數）
5. 計算邏輯驗證
6. 數據完整度分析
7. 殘差診斷分析

Usage:
    python validate_ers_final.py

Output:
    - Terminal: 完整驗證報告
    - JSON: validation_reports/ers_final_validation_{timestamp}.json
    - Plots: residual_analysis_train.png, residual_analysis_test.png, 
             learning_curve_complete.png
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

from paths import DIR_FEATURES, DIR_MODELS

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ERSFinalValidator:
    """ERS 最終完整驗證器"""
    
    def __init__(self):
        self.target = "ERS"
        self.model_dir = DIR_MODELS / self.target
        self.feature_path = DIR_FEATURES / "features_ml.parquet"
        
        # 載入設定
        self.dataset_card = self._load_json(self.model_dir / "dataset_card.json")
        self.feature_schema = self._load_json(self.model_dir / "feature_schema.json")
        
        logging.info("Initialized ERSFinalValidator")
    
    def _load_json(self, path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_data_and_model(self) -> Tuple:
        df = pd.read_parquet(self.feature_path)
        df_valid = df[df[self.target].notna()].copy()
        model = joblib.load(self.model_dir / "best_model.joblib")
        feature_cols = self.feature_schema['features']
        
        logging.info(f"Loaded {len(df_valid)} samples, model: {self.dataset_card['best_model_name']}")
        return df_valid, model, feature_cols
    
    def recreate_split(self, df: pd.DataFrame, feature_cols: list) -> Dict:
        """重現 70/30 時間序列分割"""
        df_sorted = df.sort_values("end_apnea_time")
        split_idx = int(len(df_sorted) * 0.7)
        
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        return {
            'X_train': train_df[feature_cols],
            'y_train': train_df[self.target],
            'X_test': test_df[feature_cols],
            'y_test': test_df[self.target],
            'n_train': len(train_df),
            'n_test': len(test_df)
        }
    
    def validate_train_test(self, model, split: Dict) -> Dict:
        """Train/Test 性能驗證"""
        logging.info("\n" + "="*60)
        logging.info("TRAIN/TEST PERFORMANCE")
        logging.info("="*60)
        
        # 訓練集
        y_pred_train = model.predict(split['X_train'])
        train_r2 = r2_score(split['y_train'], y_pred_train)
        train_mae = mean_absolute_error(split['y_train'], y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(split['y_train'], y_pred_train))
        
        # 測試集
        y_pred_test = model.predict(split['X_test'])
        test_r2 = r2_score(split['y_test'], y_pred_test)
        test_mae = mean_absolute_error(split['y_test'], y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(split['y_test'], y_pred_test))
        
        r2_gap = train_r2 - test_r2
        severity = ("SEVERE" if r2_gap > 0.15 else "MODERATE" if r2_gap > 0.10 
                   else "MILD" if r2_gap > 0.05 else "HEALTHY")
        
        results = {
            "train": {"r2": float(train_r2), "mae": float(train_mae), "rmse": float(train_rmse)},
            "test": {"r2": float(test_r2), "mae": float(test_mae), "rmse": float(test_rmse)},
            "gap": float(r2_gap),
            "severity": severity,
            "residuals_train": (split['y_train'].values - y_pred_train),
            "residuals_test": (split['y_test'].values - y_pred_test),
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test
        }
        
        logging.info(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Gap: {r2_gap:.4f} ({severity})")
        return results
    
    def generate_learning_curve(self, model, split: Dict) -> Dict:
        """
        生成完整學習曲線（使用固定 train/val split，逐步增加訓練樣本）
        
        這個方法正確追蹤單一模型架構在不同訓練集大小下的學習動態
        """
        logging.info("\n" + "="*60)
        logging.info("LEARNING CURVE ANALYSIS (Train + Validation)")
        logging.info("="*60)
        
        X_train = split['X_train']
        y_train = split['y_train']
        X_val = split['X_test']  # 使用固定的測試集作為驗證集
        y_val = split['y_test']
        
        # 設定訓練樣本數範圍（從 20% 到 100%）
        train_sizes_pct = np.linspace(0.2, 1.0, 10)
        train_sizes_abs = [int(len(X_train) * pct) for pct in train_sizes_pct]
        
        train_scores = []
        val_scores = []
        
        logging.info("Computing learning curve...")
        for n_samples in train_sizes_abs:
            # 使用前 n_samples 個訓練樣本
            X_subset = X_train.iloc[:n_samples]
            y_subset = y_train.iloc[:n_samples]
            
            # 克隆模型以確保每次都是新的訓練
            model_clone = clone(model)
            model_clone.fit(X_subset, y_subset)
            
            # 評估訓練集和驗證集
            train_score = r2_score(y_subset, model_clone.predict(X_subset))
            val_score = r2_score(y_val, model_clone.predict(X_val))
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)
        
        # 【修正】找到收斂點：連續兩次增幅都 <0.01
        convergence_idx = None
        for i in range(2, len(val_scores)):
            if (val_scores[i] - val_scores[i-1] < 0.01 and 
                val_scores[i-1] - val_scores[i-2] < 0.01):
                convergence_idx = i - 1
                break
        
        convergence_point = train_sizes_abs[convergence_idx] if convergence_idx else None
        
        # 視覺化
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 繪製訓練集曲線
        ax.plot(train_sizes_abs, train_scores, 'o-', color='#2E86AB', 
                linewidth=2, markersize=8, label='Training Score', alpha=0.9)
        
        # 繪製驗證集曲線
        ax.plot(train_sizes_abs, val_scores, 'o-', color='#A23B72',
                linewidth=2, markersize=8, label='Validation Score', alpha=0.9)
        
        # 標記收斂點
        if convergence_point:
            ax.axvline(convergence_point, color='green', linestyle='--', 
                    linewidth=1.5, alpha=0.7, label=f'Convergence (~{convergence_point} samples)')
        
        # 標記最終 Gap
        final_gap = train_scores[-1] - val_scores[-1]
        ax.text(train_sizes_abs[-1] * 0.95, train_scores[-1] - final_gap/2,
                f'Final Gap\n{final_gap:.3f}', 
                fontsize=10, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
        
        # 標記最終分數
        ax.annotate(f'Final R²: {val_scores[-1]:.4f}',
                xy=(train_sizes_abs[-1], val_scores[-1]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Number of Training Samples', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Complete Learning Curve: Training vs Validation\n(Fixed 70/30 Split)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.5, 1.05])
        
        plt.tight_layout()
        plt.savefig('learning_curve_complete.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Learning curve saved: learning_curve_complete.png")
        logging.info(f"Convergence point: ~{convergence_point} samples")
        logging.info(f"Final train score: {train_scores[-1]:.4f}")
        logging.info(f"Final val score: {val_scores[-1]:.4f}")
        logging.info(f"Final gap: {final_gap:.4f}")
        
        return {
            "train_sizes": [int(x) for x in train_sizes_abs],
            "train_scores": train_scores.tolist(),
            "val_scores": val_scores.tolist(),
            "convergence_point": convergence_point,
            "final_gap": float(final_gap),
            "final_train_score": float(train_scores[-1]),
            "final_val_score": float(val_scores[-1])
        }
    
    def validate_criterion(self, df: pd.DataFrame) -> Dict:
        """標準關聯效度（ERS vs HRR60）"""
        logging.info("\n" + "="*60)
        logging.info("CRITERION VALIDITY (ERS vs HRR60)")
        logging.info("="*60)
        
        valid = df[['ERS', 'hrr60']].dropna()
        
        pearson_r, pearson_p = stats.pearsonr(valid['ERS'], valid['hrr60'])
        spearman_r, spearman_p = stats.spearmanr(valid['ERS'], valid['hrr60'])
        
        ers_cv = valid['ERS'].std() / valid['ERS'].mean()
        hrr_cv = valid['hrr60'].std() / valid['hrr60'].mean()
        
        results = {
            "correlation": {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "interpretation": self._interpret_r(pearson_r)
            },
            "stability": {
                "ers_cv": float(ers_cv),
                "hrr60_cv": float(hrr_cv),
                "improvement_pct": float((hrr_cv - ers_cv) / hrr_cv * 100)
            }
        }
        
        logging.info(f"Correlation: r={pearson_r:.3f} (p={pearson_p:.3e}) - {results['correlation']['interpretation']}")
        logging.info(f"Stability: ERS CV={ers_cv:.3f} vs HRR60 CV={hrr_cv:.3f} ({results['stability']['improvement_pct']:.1f}% improvement)")
        return results
    
    def _interpret_r(self, r: float) -> str:
        abs_r = abs(r)
        return ("Strong" if abs_r > 0.7 else "Moderate" if abs_r > 0.5 
                else "Weak" if abs_r > 0.3 else "Very weak")
    
    def validate_calculation(self, df: pd.DataFrame) -> Dict:
        """計算邏輯驗證"""
        logging.info("\n" + "="*60)
        logging.info("CALCULATION LOGIC")
        logging.info("="*60)
        
        df_calc = df[['ERS', 'recovery_ratio_60s', 'recovery_ratio_90s', 'normalized_slope']].copy()
        
        def calc_ers(row):
            valid = [v for v in [row['recovery_ratio_60s'], row['recovery_ratio_90s'], 
                                 row['normalized_slope']] if pd.notna(v)]
            return np.mean(valid) if valid else None
        
        ers_replica = df_calc.apply(calc_ers, axis=1)
        valid_pairs = pd.DataFrame({'actual': df_calc['ERS'], 'replica': ers_replica}).dropna()
        
        max_diff = (valid_pairs['actual'] - valid_pairs['replica']).abs().max()
        
        results = {
            "verified": bool(max_diff < 1e-10),
            "max_diff": float(max_diff),
            "r2": float(r2_score(valid_pairs['actual'], valid_pairs['replica']))
        }
        
        logging.info(f"Verified: {'✓ YES' if results['verified'] else '✗ NO'} (max_diff={max_diff:.2e})")
        return results
    
    def analyze_completeness(self, df: pd.DataFrame) -> Dict:
        """數據完整度"""
        logging.info("\n" + "="*60)
        logging.info("DATA COMPLETENESS")
        logging.info("="*60)
        
        df_valid = df[df['ERS'].notna()].copy()
        comps = ['recovery_ratio_60s', 'recovery_ratio_90s', 'normalized_slope']
        n_comps = df_valid[comps].notna().sum(axis=1)
        
        results = {
            "total": len(df_valid),
            "complete_3": int((n_comps == 3).sum()),
            "partial_2": int((n_comps == 2).sum()),
            "partial_1": int((n_comps == 1).sum())
        }
        
        logging.info(f"Total: {results['total']}, Complete: {results['complete_3']} ({results['complete_3']/results['total']*100:.1f}%)")
        return results
    
    def residual_analysis(self, train_test_results: Dict):
        """殘差診斷分析"""
        logging.info("\n" + "="*60)
        logging.info("RESIDUAL DIAGNOSTICS")
        logging.info("="*60)
        
        for name in ['train', 'test']:
            res = train_test_results[f'residuals_{name}']
            y_pred = train_test_results[f'y_pred_{name}']
            
            # 殘差統計
            res_mean = np.mean(res)
            res_std = np.std(res)
            
            # Shapiro-Wilk 正態性檢驗
            if len(res) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(res)
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            logging.info(f"{name.capitalize()}: mean={res_mean:.4f}, std={res_std:.4f}, Shapiro-Wilk p={shapiro_p:.3f}")
            
            # 視覺化
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 殘差 vs 預測值
            axes[0].scatter(y_pred, res, alpha=0.6)
            axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0].set_xlabel('Predicted ERS')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title(f'Residual vs Predicted ({name})')
            axes[0].grid(True, alpha=0.3)
            
            # 殘差分布
            axes[1].hist(res, bins=20, density=True, alpha=0.6, edgecolor='black')
            axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Density')
            axes[1].set_title(f'Residual Distribution ({name})')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'residual_analysis_{name}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        logging.info("Residual plots saved: residual_analysis_train.png, residual_analysis_test.png")
    
    def generate_assessment(self, train_test: Dict, criterion: Dict, 
                          calculation: Dict, completeness: Dict, 
                          learning_curve_results: Dict) -> Dict:
        """整合評估"""
        checks = {
            "calculation_verified": calculation['verified'],
            "correlation_adequate": abs(criterion['correlation']['pearson_r']) > 0.4,
            "overfitting_acceptable": train_test['severity'] in ['HEALTHY', 'MILD'],
            "test_performance_good": train_test['test']['r2'] > 0.85,
            "data_quality_good": completeness['complete_3'] / completeness['total'] > 0.7,
            "learning_curve_converged": learning_curve_results['convergence_point'] is not None
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        if passed == total:
            status = "EXCELLENT"
        elif passed >= total - 1:
            status = "GOOD"
        elif passed >= total - 2:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS_IMPROVEMENT"
        
        return {
            "checks": checks,
            "status": status,
            "key_findings": {
                "ers_hrr60_correlation": criterion['correlation']['pearson_r'],
                "stability_improvement": criterion['stability']['improvement_pct'],
                "overfitting_gap": train_test['gap'],
                "test_r2": train_test['test']['r2'],
                "learning_convergence_samples": learning_curve_results['convergence_point'],
                "learning_final_gap": learning_curve_results['final_gap']
            }
        }
    
    def print_summary(self, train_test: Dict, criterion: Dict, calculation: Dict,
                     completeness: Dict, learning_curve_results: Dict, assessment: Dict):
        """列印摘要"""
        print("\n" + "="*80)
        print("ERS FINAL VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Model: {self.dataset_card['best_model_name']}")
        print("="*80)
        
        print("\n[1] DATA COMPLETENESS")
        print("-"*80)
        print(f"Total: {completeness['total']}, Complete (3 components): {completeness['complete_3']} ({completeness['complete_3']/completeness['total']*100:.1f}%)")
        
        print("\n[2] CALCULATION VERIFICATION")
        print("-"*80)
        print(f"{'✓ VERIFIED' if calculation['verified'] else '✗ FAILED'}: Max error = {calculation['max_diff']:.2e}")
        
        print("\n[3] CRITERION VALIDITY (ERS vs HRR60)")
        print("-"*80)
        print(f"Correlation: r = {criterion['correlation']['pearson_r']:.3f} ({criterion['correlation']['interpretation']})")
        print(f"Stability: ERS {criterion['stability']['improvement_pct']:.1f}% more stable than HRR60")
        
        print("\n[4] TRAIN/TEST PERFORMANCE")
        print("-"*80)
        print(f"Train R²: {train_test['train']['r2']:.4f}")
        print(f"Test R²:  {train_test['test']['r2']:.4f}")
        print(f"Gap:      {train_test['gap']:.4f} ({train_test['severity']})")
        
        print("\n[5] LEARNING CURVE ANALYSIS")
        print("-"*80)
        print(f"Convergence: ~{learning_curve_results['convergence_point']} samples")
        print(f"Final Train Score: {learning_curve_results['final_train_score']:.4f}")
        print(f"Final Val Score:   {learning_curve_results['final_val_score']:.4f}")
        print(f"Final Gap:         {learning_curve_results['final_gap']:.4f}")
        
        print("\n[6] OVERALL ASSESSMENT")
        print("-"*80)
        print(f"Status: {assessment['status']}")
        print(f"Key Findings:")
        for k, v in assessment['key_findings'].items():
            # 修正：分別處理 float 和其他類型
            if isinstance(v, float):
                print(f"  • {k}: {v:.3f}")
            else:
                print(f"  • {k}: {v}")
        
        print("\n" + "="*80)
        conclusion = ("ERS 通過所有驗證，可作為描述性恢復指標" if assessment['status'] == "EXCELLENT"
                     else "ERS 通過主要驗證，在明確範圍內可用" if assessment['status'] in ["GOOD", "ACCEPTABLE"]
                     else "ERS 存在驗證問題，需改進")
        print(f"CONCLUSION: {conclusion}")
        print("="*80 + "\n")
    
    def save_report(self, train_test: Dict, criterion: Dict, calculation: Dict,
                   completeness: Dict, learning_curve_results: Dict, assessment: Dict):
        """儲存報告"""
        output_dir = Path("validation_reports")
        output_dir.mkdir(exist_ok=True)
        
        # 移除 numpy 類型以便 JSON 序列化
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() 
                       if k not in ['residuals_train', 'residuals_test', 'y_pred_train', 'y_pred_test']}
            elif isinstance(d, (np.integer, np.floating)):
                return float(d)
            elif isinstance(d, np.bool_):
                return bool(d)
            elif isinstance(d, np.ndarray):
                return d.tolist()
            return d
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.dataset_card['best_model_name']
            },
            "data_completeness": completeness,
            "calculation_verification": calculation,
            "criterion_validity": criterion,
            "train_test_performance": clean_dict(train_test),
            "learning_curve_analysis": learning_curve_results,
            "assessment": assessment
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"ers_final_validation_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Report saved: {output_path}")

def main():
    logging.info("Starting ERS Final Validation...")
    
    try:
        validator = ERSFinalValidator()
        
        df, model, feature_cols = validator.load_data_and_model()
        split = validator.recreate_split(df, feature_cols)
        
        train_test = validator.validate_train_test(model, split)
        learning_curve_results = validator.generate_learning_curve(model, split)
        criterion = validator.validate_criterion(df)
        calculation = validator.validate_calculation(df)
        completeness = validator.analyze_completeness(df)
        
        validator.residual_analysis(train_test)
        
        assessment = validator.generate_assessment(
            train_test, criterion, calculation, completeness, learning_curve_results
        )
        
        validator.print_summary(
            train_test, criterion, calculation, completeness, 
            learning_curve_results, assessment
        )
        validator.save_report(
            train_test, criterion, calculation, completeness, 
            learning_curve_results, assessment
        )
        
        logging.info("Validation complete!")
        logging.info("\nGenerated outputs:")
        logging.info("  - learning_curve_complete.png")
        logging.info("  - residual_analysis_train.png")
        logging.info("  - residual_analysis_test.png")
        logging.info("  - validation_reports/ers_final_validation_*.json")
        
    except Exception as e:
        logging.error(f"Validation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()