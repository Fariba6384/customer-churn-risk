"""
churn_risk.py
=============
Customer Churn Risk Prediction

Uses Logistic Regression and a Random Forest Classifier to predict which
customers are likely to churn, assigns risk tiers, and generates actionable
retention recommendations.

Author : Fariba Seyedjafarrangraz
         Business Transformation Analyst | CBAP | PhD
         Dalhousie University — Registrar's Office
         Halifax, Nova Scotia, Canada
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
DATA_PATH    = Path(__file__).parent.parent / "data" / "customers.csv"
OUTPUT_PATH  = Path(__file__).parent.parent / "outputs" / "churn_predictions.csv"

FEATURES = [
    "Tenure_Months", "Monthly_Charges", "Total_Charges",
    "Num_Products", "Support_Tickets", "Last_Login_Days",
    "Satisfaction_Score", "Contract_Enc", "Payment_Enc",
]

TARGET = "Churned"


# ── 1. DATA ───────────────────────────────────────────────────────────────────

def load_or_generate(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    print("ℹ  No data file found — generating synthetic customer dataset.")
    np.random.seed(RANDOM_STATE)
    n = 600

    tenure          = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n).round(2)
    num_products    = np.random.randint(1, 6, n)
    support_tickets = np.random.poisson(2, n)
    last_login_days = np.random.randint(0, 90, n)
    satisfaction    = np.random.uniform(1.0, 5.0, n).round(1)
    contracts       = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n,
                                        p=[0.50, 0.30, 0.20])
    payments        = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], n)

    # Churn probability driven by real-world signals
    churn_prob = (
        0.25
        - 0.003 * tenure
        + 0.002 * monthly_charges
        - 0.04  * num_products
        + 0.05  * support_tickets
        + 0.004 * last_login_days
        - 0.06  * satisfaction
        + 0.15  * (contracts == "Month-to-Month").astype(float)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churned    = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "Customer_ID":      [f"C{str(i+1).zfill(4)}" for i in range(n)],
        "Tenure_Months":    tenure,
        "Monthly_Charges":  monthly_charges,
        "Total_Charges":    (tenure * monthly_charges).round(2),
        "Num_Products":     num_products,
        "Support_Tickets":  support_tickets,
        "Last_Login_Days":  last_login_days,
        "Satisfaction_Score": satisfaction,
        "Contract_Type":    contracts,
        "Payment_Method":   payments,
        TARGET:             churned,
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✔  Dataset saved → {path}\n")
    return df


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le_contract = LabelEncoder()
    le_payment  = LabelEncoder()
    df["Contract_Enc"] = le_contract.fit_transform(df["Contract_Type"])
    df["Payment_Enc"]  = le_payment.fit_transform(df["Payment_Method"])

    # Composite risk signals
    df["Engagement_Score"] = (
        (5 - df["Satisfaction_Score"])      # low satisfaction → high risk
        + df["Support_Tickets"] * 0.5
        + (df["Last_Login_Days"] / 30)
    ).round(3)

    df["Value_Score"] = (
        df["Monthly_Charges"] * df["Num_Products"]
        / (df["Tenure_Months"] + 1)
    ).round(3)

    return df


# ── 3. MODEL ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> tuple:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Logistic Regression (interpretable baseline)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_tr_sc, y_train)
    lr_preds = lr.predict(X_te_sc)
    lr_probs = lr.predict_proba(X_te_sc)[:, 1]

    # Random Forest (higher accuracy)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "lr": {
            "accuracy": accuracy_score(y_test, lr_preds),
            "auc":      roc_auc_score(y_test, lr_probs),
            "report":   classification_report(y_test, lr_preds, target_names=["Retained", "Churned"]),
            "cm":       confusion_matrix(y_test, lr_preds),
        },
        "rf": {
            "accuracy": accuracy_score(y_test, rf_preds),
            "auc":      roc_auc_score(y_test, rf_probs),
            "report":   classification_report(y_test, rf_preds, target_names=["Retained", "Churned"]),
            "cm":       confusion_matrix(y_test, rf_preds),
            "importances": pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False),
        },
    }

    return rf, scaler, metrics


# ── 4. RISK SCORING ───────────────────────────────────────────────────────────

def _risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "🔴 HIGH"
    elif prob >= 0.40:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def score_customers(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    X = df[FEATURES]
    probs = model.predict_proba(X)[:, 1]
    result = df[["Customer_ID", "Tenure_Months", "Monthly_Charges",
                  "Satisfaction_Score", "Contract_Type", TARGET]].copy()
    result["Churn_Probability"] = probs.round(4)
    result["Risk_Tier"]         = [_risk_tier(p) for p in probs]
    return result.sort_values("Churn_Probability", ascending=False)


# ── 5. REPORT ─────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, metrics: dict, scored: pd.DataFrame) -> None:
    DIV = "─" * 70

    def section(t):
        print(f"\n▌ {t}\n{DIV}")

    print("\n" + "═" * 70)
    print("   CUSTOMER CHURN RISK REPORT")
    print("   Churn Prediction & Retention Intelligence")
    print("═" * 70)

    section("Dataset Overview")
    churn_rate = df[TARGET].mean() * 100
    print(f"  Customers     : {len(df):,}")
    print(f"  Churn Rate    : {churn_rate:.1f}%")
    print(f"  Avg Tenure    : {df['Tenure_Months'].mean():.1f} months")
    print(f"  Avg Monthly $ : ${df['Monthly_Charges'].mean():.2f}")
    print(f"  Avg Satisfaction : {df['Satisfaction_Score'].mean():.2f} / 5.00")

    section("Logistic Regression Performance")
    m = metrics["lr"]
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  ROC-AUC   : {m['auc']:.4f}")
    print(f"\n  Classification Report:\n")
    for line in m["report"].splitlines():
        print(f"    {line}")

    section("Random Forest Performance")
    m = metrics["rf"]
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  ROC-AUC   : {m['auc']:.4f}")
    print(f"\n  Classification Report:\n")
    for line in m["report"].splitlines():
        print(f"    {line}")

    section("Top Predictive Features (Random Forest)")
    for feat, val in m["importances"].head(6).items():
        bar = "█" * int(val * 50)
        print(f"  {feat:<26} {val:.4f}  {bar}")

    section("Risk Tier Distribution")
    tier_counts = scored["Risk_Tier"].value_counts()
    for tier, count in tier_counts.items():
        pct = count / len(scored) * 100
        print(f"  {tier:<12} : {count:>4} customers ({pct:.1f}%)")

    section("Top 10 At-Risk Customers")
    top = scored[scored["Risk_Tier"] == "🔴 HIGH"].head(10)
    if not top.empty:
        print(top[["Customer_ID", "Churn_Probability", "Contract_Type",
                    "Monthly_Charges", "Satisfaction_Score"]].to_string(index=False))
    else:
        print("  No HIGH-risk customers identified.")

    section("Retention Recommendations")
    high_pct  = (scored["Risk_Tier"] == "🔴 HIGH").mean() * 100
    med_pct   = (scored["Risk_Tier"] == "🟡 MEDIUM").mean() * 100
    avg_sat   = df["Satisfaction_Score"].mean()
    avg_login = df["Last_Login_Days"].mean()

    if high_pct > 15:
        print(f"  ⚠  {high_pct:.1f}% of customers at HIGH risk — launch immediate outreach campaign.")
    if avg_sat < 3.5:
        print("  ⚠  Avg satisfaction below 3.5 — prioritise service quality review.")
    if avg_login > 30:
        print(f"  ⚡ Avg last-login {avg_login:.0f} days — re-engagement emails for inactive users.")
    if metrics["rf"]["auc"] > 0.80:
        print("  ✔  Model ROC-AUC >0.80 — predictions reliable for targeting campaigns.")

    mtm_churn = df[df["Contract_Type"] == "Month-to-Month"][TARGET].mean() * 100
    if mtm_churn > 30:
        print(f"  ◉  Month-to-Month churn at {mtm_churn:.1f}% — offer annual plan incentives.")

    print(f"\n{DIV}\n  Report complete.\n{DIV}\n")


# ── 6. SAVE ───────────────────────────────────────────────────────────────────

def save(scored: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_PATH, index=False)
    print(f"✔  Predictions saved → {OUTPUT_PATH}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df      = engineer_features(load_or_generate(DATA_PATH))
    model, scaler, metrics = train(df)
    scored  = score_customers(df, model, scaler)
    print_report(df, metrics, scored)
    save(scored)
