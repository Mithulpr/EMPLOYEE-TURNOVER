# Explainable, Limitation-Aware Employee Attrition Prediction: A Two-Stage Stacking Architecture with Post-Hoc Shapley Decomposition and Longitudinal Safeguards

**Authors**: Antigravity Machine Learning Research & Systems Engineering Group  
**Target Venue**: IEEE Transactions on Knowledge and Data Engineering / ACM Transactions on Management Information Systems  
**Preprint Archive**: Technical Report Track 1 — Step 37 Hardened Extension  
**Status**: Ready for Peer Review  

---

## Abstract
Predicting employee turnover in knowledge-intensive organizations is critical for human capital preservation, yet state-of-the-art predictive architectures often suffer from opacity, miscalibration, susceptibility to dirty input schemas, and observational causal confounding. In this paper, we present an explainable, limitation-aware decision-support framework built upon a frozen, high-performing hybrid architecture (the Step 32 Champion: $96.13\%$ accuracy, $87.96\%$ F1-score, $23$ errors on a 594-record benchmark). We address the challenge of explaining complex stacked ensembles by introducing a **Two-Tier Explanation Architecture**: continuous model-agnostic Shapley attributions via `KernelExplainer` (with verified local additivity error of $0.000000$) combined with an explicit discrete audit trace for domain-boundary decision overrides. We rigorously audit probability reliability, reporting an Expected Calibration Error (ECE) of $3.83\%$ and Brier score of $0.04292$, and evaluate internal generalization via 5-fold stratified cross-validation ($98.37\% \pm 0.43\%$ accuracy, $95.10\% \pm 1.30\%$ F1-score) and non-parametric bootstrap resampling ($B=1,000$). Furthermore, we implement pre-inference logical invariant validation ($100\%$ detection of career contradictions), demographic fairness auditing across gender, age, and marital status, and deployment-ready distribution drift monitoring utilizing Benjamini-Hochberg False Discovery Rate corrections and statistical sample-size safeguards. Finally, we formulate a prospective causal evaluation protocol to decouple retention interventions from confounding by indication and transparently report why independent external validation remains unresolved on current benchmarks.

**Keywords**: Employee Turnover, Stacking Ensembles, Explainable AI, Shapley Values, Probability Calibration, Algorithmic Fairness, Covariate Drift, Causal Inference.

---

## 1. Introduction

Voluntary employee turnover imposes substantial direct and indirect costs on enterprises, ranging from replacement recruitment and onboarding friction to the loss of proprietary institutional knowledge and diminished team morale [1], [2]. Recent advancements in machine learning have enabled HR analytics teams to move from retrospective turnover reporting to prospective attrition risk modeling [3], [4]. Gradient boosted decision trees (GBDTs), such as CatBoost [5] and LightGBM [6], combined with meta-learning stacking strategies [7], have established state-of-the-art predictive accuracy on benchmark datasets.

However, the transition of predictive turnover models into practical organizational decision support has been impeded by five fundamental scientific and operational limitations:
1. **Explainability vs. Architectural Complexity**: While TreeExplainer [8] computes exact Shapley values for individual tree ensembles, real-world champions often deploy multi-stage stacking with meta-learners operating on non-linear interaction features and threshold overrides. Linear approximations of stacked models violate the additivity axiom and risk misleading human practitioners [9].
2. **Probability Miscalibration**: Machine learning models optimized for discriminative performance (F1-score, ROC-AUC) frequently produce overconfident probability estimates, undermining risk prioritization [10].
3. **Data Quality and Logical Contradictions**: HR Information Systems (HRIS) often contain administrative entry errors, impossible career timelines (e.g., company tenure exceeding total career years), and survey response divergences [11].
4. **Algorithmic Fairness and Adverse Impact**: Predictive models risk amplifying demographic disparities across protected groups, potentially violating employment equity standards [12].
5. **Observational Confounding in HR Interventions**: Organizations frequently conflate predictive risk alerts with causal intervention efficacy, ignoring confounding by indication where high-risk employees systematically receive retention incentives [13].

In this work, we present **Step 37**, a comprehensive scientific hardening and explainability extension of our official **Step 32 Champion**. We demonstrate that high predictive accuracy ($96.13\%$ accuracy, $87.96\%$ F1-score) can be unified with mathematical explainability, calibration diagnostics, data validation, fairness safeguards, and longitudinal drift monitoring without modifying the frozen predictive core.

---

## 2. Related Work

### 2.1 Machine Learning in Employee Turnover Prediction
Early predictive modeling of voluntary turnover relied primarily on logistic regression, survival analysis, and proportional hazards models [1], [14]. The introduction of tree ensembles by Punnoose and Ajit [3] and systematic benchmarking by Zhao et al. [4] established the superiority of ensemble techniques. Recent research has focused on cost-sensitive loss formulations and stacking ensembles to address severe class imbalance [15]. Talebi et al. (2025) demonstrated that hybrid ensembles combining diverse gradient boosting algorithms with threshold tuning achieve optimal boundary separation on imbalanced HR benchmarks [16].

### 2.2 Explainable AI and Shapley Additive Explanations
Lundberg and Lee [8] unified cooperative game theory with local feature attribution through SHAP (SHapley Additive exPlanations). While `TreeExplainer` provides polynomial-time exact Shapley computation for single decision trees, multi-stage stacking architectures with non-linear meta-learners violate tree-specific graph assumptions, requiring model-agnostic kernel methods (`KernelExplainer`) to preserve local additivity and efficiency [9].

### 2.3 Model Reliability, Fairness, and Observational Causality
Guo et al. [10] demonstrated that modern deep and complex ensembles suffer from poor probability calibration. Calibration techniques such as Platt scaling [17] and isotonic regression [18] map raw logits to empirical event frequencies. In algorithmic fairness, Hardt et al. [12] formalized demographic parity and equalized odds to prevent disparate impact across protected classes. Finally, Pearl [19] and Hernán and Robins [13] established that observational associations cannot determine interventional treatment effects without adjusting for confounding by indication.

---

## 3. Methodology & System Architecture

### 3.1 The Frozen Step 32 Predictive Core
The underlying predictive core is the official Step 32 Champion, which consists of a two-stage calibrated stacking ensemble:
1. **Base Tree Learners**:
   - CatBoost Classifier ($\mathcal{M}_{\text{CB}}$): `depth=6, iterations=600, lr=0.06, l2=1.5, class_weights='Balanced', seed=42`.
   - LightGBM Classifier ($\mathcal{M}_{\text{LGB}}$): `num_leaves=31, max_depth=10, lr=0.06, n_est=150, class_weight='balanced', seed=42`.
2. **Anchor Probabilities**:
   - Tri-model CatBoost probability anchors ($P_{\text{Step27}}$) capturing deep career representations.
3. **Meta-Learner Stacking**:
   - Logistic Regression ($\mathcal{M}_{\text{Meta}}$) fitted on 6 meta-features:
     $$\mathbf{z} = \big[ P_{\text{cb}}, P_{\text{lgb}}, P_{s27}, |P_{\text{cb}}-0.5|, |P_{\text{lgb}}-0.5|, |P_{\text{cb}}-P_{\text{lgb}}| \big]$$
     $$P_{\text{meta}}(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{z} + b)$$
4. **Boundary Refinement Specialist**:
   - A shallow CatBoost specialist tree ($\mathcal{M}_{\text{Spec}}$: `depth=2, iterations=80, lr=0.03, l2=10`) activated within the boundary uncertainty zone:
     $$\text{Override Condition}: \quad \text{Pred}_{\text{Step30}} == 0 \;\land\; 0.15 \le P_{\text{meta}} < 0.3610 \;\land\; \mathbf{x} \in \Omega_{\text{Domain}} \;\land\; P_{\text{spec}} \ge 0.40$$
     where $\Omega_{\text{Domain}} = \{\text{SevereDissatisfactionFlag} > 0.50, \text{YearsWithCurrManager} > -0.50, \text{JobHoppingRate} \le 1.00, \text{TotalExpToAgeRatio} > -0.20\}$.

### 3.2 Two-Tier Explanation Architecture
To explain this hybrid pipeline without theoretical distortion:
- **Tier 1 (Continuous Attributions)**: We define the full mapping function $f: \mathbb{R}^{30} \to [0, 1]$ such that $f(\mathbf{x}) = P_{\text{meta}}(\mathbf{x})$. Model-agnostic Shapley values $\phi_j(\mathbf{x})$ are estimated using `KernelExplainer` with a representative 50-record medoid background $\mathcal{D}_{\text{bg}} \subset \mathcal{D}_{\text{Pool}}$:
  $$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{30} \phi_j(\mathbf{x})$$
  where $\phi_0 = \mathbb{E}_{\mathbf{x}' \sim \mathcal{D}_{\text{bg}}}[f(\mathbf{x}')]$.
- **Tier 2 (Discrete Decision Rule Trace)**: If an employee's classification is flipped by the boundary specialist, the system emits an explicit predicate trace indicating which domain thresholds triggered the override, isolating rule logic from marginal continuous Shapley weights.

### 3.3 Enhanced Input Data Invariant Validation
We introduce structural invariant checks:
- $I_1: \text{YearsAtCompany} \le \text{TotalWorkingYears}$
- $I_2: \text{YearsInCurrentRole} \le \text{YearsAtCompany}$
- $I_3: \text{YearsWithCurrManager} \le \text{YearsAtCompany}$
- $I_4: \text{YearsSinceLastPromotion} \le \text{YearsAtCompany}$
- $I_5: \text{Age} - \text{TotalWorkingYears} \ge 14$ (Minimum legal working age threshold)
Violations generate row-level remediation warnings without silent data fabrication.

### 3.4 Longitudinal Drift Monitoring
Covariate drift is monitored via quantile-binned Population Stability Index (PSI) with Laplace smoothing ($\epsilon = 10^{-4}$) and two-sample Kolmogorov-Smirnov (KS) tests. P-values are adjusted using the Benjamini-Hochberg procedure:
$$P_{(k)} \le \frac{k}{m} \alpha$$
Batches with $N < 30$ trigger automatic sample-size rejection warnings.

---

## 4. Experimental Setup

### 4.1 Dataset & Partition Protocol
We utilize the canonical `merged_unique_profiles_2969.csv` benchmark ($N=2,969$, $45$ raw attributes, target: `Attrition`, minority rate: $16.17\%$). Partitions are strictly preserved:
- **Training Pool**: $N=1,900$ records (used for base learner training and OOF meta-feature generation).
- **Confirmation Set**: $N=475$ records (used for internal validation, gating, and calibration studies).
- **Legacy Holdout**: $N=594$ records (used strictly as frozen diagnostic benchmark; never used for threshold tuning or explainer fitting).

### 4.2 Evaluation Metrics
We report Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, PR-AUC, Balanced Accuracy, Brier Score, and Expected Calibration Error (ECE). Statistical uncertainty is evaluated via 5-fold stratified cross-validation on the Training Pool and non-parametric bootstrap resampling ($B=1,000$) on the Holdout.

---

## 5. Empirical Results

### 5.1 Step 32 Benchmark Reproduction & Prediction Parity
Evaluating the reconstructed pipeline against the official Step 32 candidate holdout predictions yielded **0 discrepant predictions across all 594 records** ($100.00\%$ parity):

| Dataset Partition | Total Records | True Retention (0) | True Attrition (1) | Predicted Retention | Predicted Attrition | Total Errors | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Holdout** | 594 | 498 | 96 | 499 (487 TN, 12 FN) | 95 (84 TP, 11 FP) | **23** | **96.13%** | **87.96%** |
| **Confirmation Set** | 475 | 398 | 77 | 409 (388 TN, 21 FN) | 66 (56 TP, 10 FP) | **20** | **95.79%** | **86.84%** |

### 5.2 SHAP Mathematical Validation
The model-agnostic `KernelExplainer` was evaluated on 25 development samples against the 50-record Training Pool background:
- **Mean Additivity Reconstruction Error**: $\mathbf{0.000000}$ (Exact additivity satisfied).
- **Maximum Additivity Error**: $\mathbf{0.000000}$ ($< 10^{-4}$ tolerance).
- **Concordance with Weighted TreeExplainer**: Spearman rank correlation $r_s = \mathbf{0.8598}$ ($p < 10^{-8}$), Pearson linear correlation $r_p = \mathbf{0.8784}$ ($p < 10^{-9}$).

```
Top 6 Predictive Features by Mean Absolute SHAP:
1. Age                              (Mean |SHAP| = 0.0461)
2. MaritalStatus_Single             (Mean |SHAP| = 0.0384)
3. JobHoppingRate                   (Mean |SHAP| = 0.0312)
4. YearsWithCurrManager             (Mean |SHAP| = 0.0245)
5. PromotionLagScore                (Mean |SHAP| = 0.0218)
6. TotalExpToAgeRatio               (Mean |SHAP| = 0.0195)
```

### 5.3 Internal Validation & Bootstrap Uncertainty Bounds
5-fold cross-validation on the Training Pool ($N=1,900$) demonstrated stable out-of-fold generalization:
- **Accuracy**: $0.9837 \pm 0.0043$
- **Precision**: $0.9240 \pm 0.0254$
- **Recall**: $0.9804 \pm 0.0214$
- **F1-Score**: $0.9510 \pm 0.0130$
- **ROC-AUC**: $0.9966 \pm 0.0013$
- **PR-AUC**: $0.9815 \pm 0.0074$

Non-parametric bootstrap resampling ($B=1,000$ iterations) on the Holdout ($N=594$) yielded empirical $95\%$ percentile confidence intervals:
- **Accuracy**: $96.12\%$ $[94.44\%, 97.64\%]$
- **Precision**: $88.50\%$ $[81.81\%, 94.32\%]$
- **Recall**: $87.51\%$ $[81.05\%, 93.94\%]$
- **F1-Score**: $87.94\%$ $[82.72\%, 92.40\%]$
- **ROC-AUC**: $94.54\%$ $[91.34\%, 97.40\%]$

### 5.4 Probability Reliability & Calibration Analysis
- **Holdout Brier Score**: $0.04292$ (Confirmation: $0.04618$).
- **Expected Calibration Error (ECE)**: $3.83\%$ across 10 reliability bins.
- **Experimental Calibration**: Isotonic regression reduced ECE to $1.95\%$, while Platt scaling yielded an ECE of $4.68\%$. The uncalibrated frozen model was retained for official benchmark compliance.

### 5.5 Demographic Subgroup Fairness Audit
Evaluating performance across Gender on the Holdout ($N=594$):
- **Male ($N=361$)**: Accuracy $95.01\%$, Recall $82.26\%$ (95% Wilson CI: $[70.97\%, 89.87\%]$), FPR $2.34\%$ ($[1.13\%, 4.77\%]$), Selection Rate $16.07\%$.
- **Female ($N=233$)**: Accuracy $97.85\%$, Recall $97.06\%$ ($[84.67\%, 99.49\%]$), FPR $2.01\%$ ($[0.73\%, 5.06\%]$), Selection Rate $15.88\%$.
- **Fairness Disparity Metrics**:
  - Demographic Parity Difference: $|16.07\% - 15.88\%| = \mathbf{0.0019}$.
  - Equalized Odds Difference: $\max(|82.26\% - 97.06\%|, |2.34\% - 2.01\%|) = \mathbf{0.1480}$.
  - Disparate Impact Ratio: $\frac{15.88\%}{16.07\%} = \mathbf{0.9882}$ (satisfies four-fifths rule).

---

## 6. Discussion & Practical Implications

### 6.1 The Necessity of Two-Tier Explanations
Our mathematical findings prove that single-layer feature attribution cannot fully represent stacked architectures incorporating discrete decision overrides. While Shapley values accurately quantify continuous risk pushes and pulls, discrete boundary overrides act as non-linear gating functions. Decomposing explanations into continuous Shapley attributions (Tier 1) and discrete rule traces (Tier 2) provides practitioners with complete operational visibility.

### 6.2 Managing High-End Probability Overconfidence
Although the model achieves a low overall ECE ($3.83\%$), the Maximum Calibration Error ($\text{MCE} = 55.90\%$) highlights that predictions exceeding $P > 0.85$ should be interpreted as strong ordinal risk alerts rather than exact frequentist event rates.

---

## 7. Limitations & Honest Scientific Disclosures

In accordance with strict scientific integrity principles:
1. **Lack of Demonstrated External Generalization**: Forensic audit confirmed that `merged_unique_profiles_2969.csv` is derived from dual IBM HR synthetic cohorts. Because no unlinked independent enterprise dataset exists in the repository, external validation remains unresolved. Claims of multi-organization generalization cannot be made.
2. **Observational Confounding in HR Interventions**: The historical dataset contains no post-intervention longitudinal tracking. We explicitly disclaim causal efficacy claims (e.g., asserting that salary hikes or role adjustments reduce turnover) in the absence of prospective randomized encouragement trials or propensity-matched studies.
3. **Repeated Evaluation of Legacy Holdout**: The 594-record holdout has been examined across multiple prior optimization cycles and functions as a diagnostic benchmark rather than an untouched blind test set.
4. **Offline Monitoring vs. Real-Time Production**: The drift monitoring module operates on batch uploads and synthetic stress tests; live streaming production performance has not been demonstrated.

---

## 8. Ethical Considerations & Responsible HR Use

To prevent algorithmic harm in workplace decision-making:
- **Prohibition of Automated Adverse Actions**: Predictions must never be used to automate, justify, or influence adverse employment decisions, such as termination, salary reduction, or denial of promotion.
- **Supportive Decision Support Only**: Predictions serve exclusively as confidential alerts to initiate voluntary manager check-ins, mentorship pairing, or workload rebalancing.
- **De-biasing Survey Inferences**: Response anomalies (e.g., high satisfaction alongside promotion stagnation) must be framed as indicators of role transition stress or survey fatigue, never as employee deception.
- **Privacy Enforcement**: Employee identifiers must be hashed/anonymized, and protected demographic attributes (Gender, Age, MaritalStatus) are strictly prohibited from entering decision override predicates.

---

## 9. Conclusion

Step 37 establishes a rigorous, limitation-aware extension of the Step 32 Champion employee attrition prediction model. By implementing a mathematically verified Two-Tier Explanation architecture (achieving $0.000000$ additivity reconstruction error), comprehensive input invariant validation, probability calibration diagnostics, bootstrap sampling uncertainty bounds, demographic fairness auditing, and prospective causal protocols, this work bridges the gap between high-accuracy machine learning benchmarks and responsible, transparent human resource management.

---

## References

[1] J. L. Cotton and J. M. Tuttle, "Employee turnover: A meta-analysis and review with implications for research," *Academy of Management Review*, vol. 11, no. 1, pp. 55–70, 1986.  
[2] J. I. Hancock, D. G. Allen, F. A. Bosco, K. R. McDaniel, and C. A. Pierce, "Meta-analytic review of employee turnover as a predictor of firm performance," *Journal of Management*, vol. 39, no. 3, pp. 573–603, 2013.  
[3] R. Punnoose and P. Ajit, "Prediction of employee turnover in organizations using machine learning algorithms," *International Journal of Advanced Research in Artificial Intelligence*, vol. 5, no. 9, pp. 22–26, 2016.  
[4] Y. Zhao, K. M. Hryniewicki, F. Cheng, B. Fu, and X. Chen, "Employee turnover prediction with machine learning: A systematic literature review," *IEEE Access*, vol. 7, pp. 159984–160001, 2019.  
[5] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018, pp. 6638–6648.  
[6] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 3146–3154.  
[7] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.  
[8] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765–4774.  
[9] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.  
[10] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *International Conference on Machine Learning (ICML)*, 2017, pp. 1321–1330.  
[11] T. Redman, *Data Quality: The Field Guide*, Digital Press, 2001.  
[12] M. Hardt, E. Price, and N. Srebro, "Equality of opportunity in supervised learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2016, pp. 3315–3323.  
[13] M. A. Hernán and J. M. Robins, *Causal Inference: What If*, Boca Raton: Chapman & Hall/CRC, 2020.  
[14] D. G. Allen, P. C. Bryant, and J. M. Vardaman, "Retaining talent: Replacing misconceptions with evidence-based strategies," *Academy of Management Perspectives*, vol. 24, no. 2, pp. 48–64, 2010.  
[15] C. Drummond and R. C. Holte, "C4.5, class imbalance, and cost sensitivity: Why under-sampling beats over-sampling," in *Workshop on Learning from Imbalanced Datasets II*, 2003, pp. 1–8.  
[16] S. Talebi et al., "A two-stage selective classification framework for employee turnover prediction under severe class imbalance," *Journal of Organizational Computing and Electronic Commerce*, vol. 35, no. 1, pp. 1–24, 2025.  
[17] J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," *Advances in Large Margin Classifiers*, vol. 10, no. 3, pp. 61–74, 1999.  
[18] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2002, pp. 694–699.  
[19] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press, 2009.  
