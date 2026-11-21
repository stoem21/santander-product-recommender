# Explainability of the final model (LightGBM multiclass, cap+B, 178 features, 100 rounds)
Sample: 8,000 random test customers (2016-06 recommendations). TreeSHAP via LightGBM `pred_contrib=True`.
SHAP values are in raw-score units per product class. They explain the model, not causes.

## What drives the top-1 recommendation (share of |SHAP|)
- 43% "this product is not held now" - mostly eligibility (non-holders always have lag1 = 0); not an insight.
- Excluding that: history of OTHER products 44%, past history of THIS product 16%, other profile 14%, product counts 10%,
  age 6%, previous-month profile (B) 5%, month_of_year 3.5%, sex/foreigner/residence 0.8%, n_prior_capped 0.4%.

## month_of_year = June (constant in test)
Raises taxes (+0.33) and current_account (+0.11); lowers long_term_deposit (-0.50), securities (-0.25).
Learned from a single June (2015-06) in training.

## Counterfactual checks (change one feature only)
| change | top-1 changes | top-7 set changes |
|---|---|---|
| sex H<->V | 8.4% | 15.5% |
| foreigner N<->S | 6.0% | 18.5% |
| age +10 years | 16.5% | 23.7% |
| n_prior_capped -> 6 | 0.1% | 0.1% |
Small SHAP share but not small effect on rankings, because the top candidates are often close.

## Removing sex / foreigner_index / residence_index (4 folds x 2 seeds)
MAP@7 0.8983 -> 0.8980 (diff -0.0003, without better in 3/8 pairs): within noise.
Removing direct use does not rule out proxies through other features.

Customer examples: customer_cards.md

---
# Update: final model v2 (sex / foreigner_index / residence_index removed, 175 features)
Previous model and submission kept in models/v1_with_sensitive and submission/v1_with_sensitive.

## month_of_year ablation (4 folds x 2 seeds, on v2 features)
Removing it: mean +0.0001 (better in 3/8). Helps Feb (+0.0015 both seeds), hurts Apr (-0.001 both seeds; Apr has a repeated taxes spike).
Kept: neutral on average; captures seasonality where it repeats. Judgment call, not a measured gain.

## Do other models think alike? (same 5,000 test customers, explaining the final model's #1 product)
| | LightGBM s1 vs s2 | LightGBM vs XGBoost |
|---|---|---|
| same #1 recommendation | 84.5% | 82.0% |
| top-7 Jaccard | 0.953 | 0.901 |
| per-customer top-5 feature overlap | 86.0% | 72.2% |
| same #1 feature | 98.9% | 66.7% |
| Spearman of mean abs SHAP per (product, feature) | 0.898 | 0.840 |
Group shares (eligibility excluded) are close across all three models: other-product history ~40%, own history 15-19%,
counts 10-14%, other profile 13-15%, age 7%, profile lags 5-7%, month_of_year 3-4%.
Most #1-feature disagreements: LightGBM credits "this product not held" (eligibility) where XGBoost credits age or ever_held.
Same June direction in all models (taxes up, long_term_deposit down); magnitudes differ.

## Ranking instability
The #1 product changes with the seed alone for 15.5% of customers; for them the gap p(#1)-p(#2) has median 0.019
(vs 0.363 when unchanged). 28% of customers have a gap below 0.05: the model is near-indifferent between its top two.
