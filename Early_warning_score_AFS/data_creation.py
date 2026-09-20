import numpy as np
import pandas as pd
from datetime import timedelta
import json

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Population design
# ---------------------------------------------------------------------------
# Four sub-segments so the dataset isn't a trivial separate-the-classes toy:
#   good_quiet        - stays healthy throughout, no stress signal at all
#   good_false_alarm  - shows a mild, transient stress signal but recovers
#                       (normal life volatility -> the false-positive case)
#   bad_detectable    - shows a genuine, intensifying stress signal in the
#                       3 months before the observation point, then goes bad
#   bad_shock         - goes bad with NO prior transactional warning at all
#                       (idiosyncratic shock -> the model's detection ceiling)
N_GOOD_QUIET = 570
N_GOOD_FALSE_ALARM = 30
N_BAD_DETECTABLE = 170
N_BAD_SHOCK = 30
N_ACCOUNTS = N_GOOD_QUIET + N_GOOD_FALSE_ALARM + N_BAD_DETECTABLE + N_BAD_SHOCK

segments = (
    ["good_quiet"] * N_GOOD_QUIET
    + ["good_false_alarm"] * N_GOOD_FALSE_ALARM
    + ["bad_detectable"] * N_BAD_DETECTABLE
    + ["bad_shock"] * N_BAD_SHOCK
)
rng.shuffle(segments)

account_ids = [f"AC{100000+i}" for i in range(N_ACCOUNTS)]
bad_flag_map = {"good_quiet": 0, "good_false_alarm": 0, "bad_detectable": 1, "bad_shock": 1}

# 9-month feature/history window: Oct 2025 -> Jun 2026 (observation point = 30 Jun 2026)
# implied 3-month forward outcome window: Jul-Sep 2026 (i.e. "did they go bad by ~now")
month_starts = pd.date_range("2025-10-01", periods=9, freq="MS")

# ---------------------------------------------------------------------------
# Account master
# ---------------------------------------------------------------------------
accounts = []
for aid, seg in zip(account_ids, segments):
    base_income = float(rng.lognormal(mean=np.log(650), sigma=0.35))
    base_income = min(max(base_income, 250), 2000)  # BHD/month, illustrative only
    has_loan = bool(rng.random() < 0.7)
    loan_installment = base_income * rng.uniform(0.18, 0.28) if has_loan else 0.0
    essential_ratio_base = rng.uniform(0.35, 0.50)
    discretionary_ratio_base = rng.uniform(0.15, 0.30)
    initial_balance = base_income * rng.uniform(0.5, 2.0)

    if seg == "bad_detectable":
        severity = rng.uniform(0.7, 1.0)
    elif seg == "good_false_alarm":
        severity = rng.uniform(0.2, 0.45)
    else:
        severity = 0.0

    accounts.append(
        dict(
            account_id=aid,
            true_segment=seg,
            bad_flag=bad_flag_map[seg],
            declared_monthly_income=round(base_income, 2),
            has_loan=has_loan,
            loan_installment=round(loan_installment, 2),
            essential_ratio_base=round(essential_ratio_base, 3),
            discretionary_ratio_base=round(discretionary_ratio_base, 3),
            initial_balance=round(initial_balance, 2),
            severity=round(severity, 3),
        )
    )

accounts_df = pd.DataFrame(accounts)


def stress_for_month(month_idx, severity):
    """Stress ramps over the 3 months immediately before the observation point
    (month indices 6,7,8 = calendar months 7,8,9 of the 9-month window)."""
    if severity <= 0:
        return 0.0
    ramp = {6: 1 / 3, 7: 2 / 3, 8: 1.0}
    return severity * ramp.get(month_idx, 0.0)


# ---------------------------------------------------------------------------
# Transaction generation
# ---------------------------------------------------------------------------
rows = []
for r in accounts_df.itertuples(index=False):
    aid = r.account_id
    income = r.declared_monthly_income
    loan_inst = r.loan_installment
    ess_base = r.essential_ratio_base
    disc_base = r.discretionary_ratio_base
    severity = r.severity

    for m_idx, m_start in enumerate(month_starts):
        stress = stress_for_month(m_idx, severity)
        days_in_month = (m_start + pd.offsets.MonthEnd(1) - m_start).days + 1

        # --- salary credit ---
        income_reduction = rng.uniform(0.1, 0.4) * stress if stress > 0 else 0.0
        pay_amount = income * (1 - income_reduction)
        pay_day = int(min(28, rng.integers(1, 6) + stress * rng.integers(3, 14)))
        rows.append((aid, m_start + timedelta(days=pay_day - 1), "SALARY_CREDIT",
                     round(pay_amount, 2), "SUCCESS"))

        # --- EMI debit (only if the account has a loan) ---
        if loan_inst > 0:
            bounce_prob = 0.005 + stress * 0.30
            bounced = rng.random() < bounce_prob
            status = "BOUNCED" if bounced else "SUCCESS"
            rows.append((aid, m_start + timedelta(days=4), "EMI_DEBIT",
                         round(loan_inst, 2), status))
            if bounced:
                rows.append((aid, m_start + timedelta(days=5), "NSF_FEE", 5.0, "SUCCESS"))

        # --- essential spend (rent, utilities, groceries) ---
        ess_ratio = min(0.95, ess_base + stress * 0.25)
        ess_total = income * ess_ratio
        n_ess = int(rng.integers(3, 7))
        for amt in rng.dirichlet(np.ones(n_ess)) * ess_total:
            d = int(rng.integers(1, days_in_month + 1))
            rows.append((aid, m_start + timedelta(days=d - 1), "ESSENTIAL_SPEND",
                         round(float(amt), 2), "SUCCESS"))

        # --- discretionary spend (dining, shopping, entertainment) ---
        disc_ratio = max(0.02, disc_base * (1 - stress * 0.6))
        disc_total = income * disc_ratio
        n_disc = int(rng.integers(2, 5))
        for amt in rng.dirichlet(np.ones(n_disc)) * disc_total:
            d = int(rng.integers(1, days_in_month + 1))
            rows.append((aid, m_start + timedelta(days=d - 1), "DISCRETIONARY_SPEND",
                         round(float(amt), 2), "SUCCESS"))

        # --- occasional high-risk / cash-advance style spend under stress ---
        if stress > 0.5 and rng.random() < 0.2 * stress:
            d = int(rng.integers(1, days_in_month + 1))
            rows.append((aid, m_start + timedelta(days=d - 1), "HIGH_RISK_SPEND",
                         round(float(rng.uniform(20, 100)), 2), "SUCCESS"))

        # --- cash withdrawals (noise feature, not stress-linked) ---
        for _ in range(int(rng.integers(1, 3))):
            d = int(rng.integers(1, days_in_month + 1))
            rows.append((aid, m_start + timedelta(days=d - 1), "CASH_WITHDRAWAL",
                         round(float(rng.uniform(20, 80)), 2), "SUCCESS"))

txn_df = pd.DataFrame(rows, columns=["account_id", "date", "category", "amount", "status"])
txn_df = txn_df.sort_values(["account_id", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Running balance per account
# ---------------------------------------------------------------------------
init_bal_map = accounts_df.set_index("account_id")["initial_balance"].to_dict()
balances = np.empty(len(txn_df))
bal_state = dict(init_bal_map)
for i, r in enumerate(txn_df.itertuples(index=False)):
    b = bal_state[r.account_id]
    if r.status == "SUCCESS":
        b += r.amount if r.category == "SALARY_CREDIT" else -r.amount
    bal_state[r.account_id] = b
    balances[i] = round(b, 2)
txn_df["balance_after"] = balances

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
out_accounts = accounts_df.drop(columns=["severity"]).copy()
out_accounts.to_csv("/home/claude/accounts.csv", index=False)
txn_df.to_csv("/home/claude/transactions.csv", index=False)

# ---------------------------------------------------------------------------
# Sanity-check aggregates (for validation charts)
# ---------------------------------------------------------------------------
merged = txn_df.merge(accounts_df[["account_id", "true_segment"]], on="account_id")
merged["month_idx"] = merged.groupby("account_id")["date"].rank(method="dense").astype(int)
# recompute month_idx properly from calendar month instead of rank (safer)
month_map = {d: i + 1 for i, d in enumerate(month_starts)}
merged["month_num"] = merged["date"].dt.to_period("M").dt.to_timestamp().map(month_map)

summary = {}
for seg in ["good_quiet", "good_false_alarm", "bad_detectable", "bad_shock"]:
    seg_df = merged[merged.true_segment == seg]
    n_seg_accounts = accounts_df[accounts_df.true_segment == seg].shape[0]
    bounced_by_month = (
        seg_df[seg_df.category == "EMI_DEBIT"]
        .assign(is_bounced=lambda d: (d.status == "BOUNCED").astype(int))
        .groupby("month_num")["is_bounced"].sum() / n_seg_accounts
    ).reindex(range(1, 10), fill_value=0).round(3).tolist()

    ess_ratio_by_month = []
    for mn in range(1, 10):
        m_df = seg_df[seg_df.month_num == mn]
        ess = m_df[m_df.category == "ESSENTIAL_SPEND"].groupby("account_id")["amount"].sum()
        inc = m_df[m_df.category == "SALARY_CREDIT"].groupby("account_id")["amount"].sum()
        ratio = (ess / inc).replace([np.inf, -np.inf], np.nan).dropna().mean()
        ess_ratio_by_month.append(round(float(ratio), 3))

    summary[seg] = dict(n_accounts=int(n_seg_accounts),
                         bounced_emi_rate_by_month=bounced_by_month,
                         avg_essential_ratio_by_month=ess_ratio_by_month)

print(json.dumps({
    "n_accounts": N_ACCOUNTS,
    "n_transactions": len(txn_df),
    "date_range": [str(txn_df.date.min().date()), str(txn_df.date.max().date())],
    "bad_rate": round(accounts_df.bad_flag.mean(), 3),
    "segment_counts": accounts_df.true_segment.value_counts().to_dict(),
    "segment_summary": summary,
}, indent=2, default=str))
