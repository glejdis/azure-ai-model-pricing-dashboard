"""
normalize.py – Field normalization, month range derivation, and enrichment joins.

Responsibilities:
  - Compute the previous calendar month date range.
  - Map raw Cost Details CSV column names to a canonical schema.
  - Filter to Azure OpenAI rows.
  - Compute derived metrics (discount %, effective price per 1M tokens).
  - Optionally join retail / price-sheet prices for discount computation.
"""

from __future__ import annotations

import logging
from calendar import monthrange
from datetime import date, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical column names used throughout the project
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "subscription_id",
    "date",
    "resource_id",
    "service_name",
    "service_tier",
    "product_name",
    "meter_name",
    "token_type",
    "quantity",
    "unit_price",
    "effective_price",
    "payg_price",
    "cost",
    "currency",
    "additional_info",
]

# ---------------------------------------------------------------------------
# Token type classification
# ---------------------------------------------------------------------------
# Azure OpenAI / Foundry bills cached prompt tokens under a dedicated meter
# (whose name contains "cached") at a discounted rate.  These must be separated
# from regular input tokens, so "cached" is checked *before* "input"/"prompt"
# below (cached meters are typically named e.g. "Cached Input Tokens").

TOKEN_TYPE_CACHED = "Cached Input Tokens"
TOKEN_TYPE_INPUT = "Input Tokens"
TOKEN_TYPE_OUTPUT = "Output Tokens"
TOKEN_TYPE_PTU = "PTU"
TOKEN_TYPE_IMAGE = "Image"


def classify_token_type(meter_name: str | None) -> str:
    """Classify a Cost Details meter name into a token / usage category.

    Returns one of: ``"Cached Input Tokens"``, ``"Input Tokens"``,
    ``"Output Tokens"``, ``"PTU"``, ``"Image"``, ``"Other"`` or ``"Unknown"``.

    The ordering is significant: cached-token meters often contain both
    ``"cached"`` and ``"input"`` (e.g. "Cached Input Tokens"), so ``"cached"``
    must be matched first to avoid them being counted as plain input tokens.
    """
    low = (meter_name or "").lower()
    if not low:
        return "Unknown"
    if "cached" in low or "cache" in low:
        return TOKEN_TYPE_CACHED
    if "prompt" in low or "input" in low:
        return TOKEN_TYPE_INPUT
    if "completion" in low or "output" in low:
        return TOKEN_TYPE_OUTPUT
    if "ptu" in low or "provisioned" in low:
        return TOKEN_TYPE_PTU
    if "image" in low or "dall" in low:
        return TOKEN_TYPE_IMAGE
    return "Other"


def add_token_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``token_type`` column derived from ``meter_name`` (in place)."""
    if df.empty:
        return df
    if "meter_name" in df.columns:
        df["token_type"] = df["meter_name"].map(classify_token_type)
    else:
        df["token_type"] = "Unknown"
    return df

# Mapping from common raw column names → canonical names.
# The keys are lower-cased and stripped; values are canonical names.
COLUMN_ALIASES: dict[str, str] = {
    # Subscription
    "subscriptionid": "subscription_id",
    "subscriptionguid": "subscription_id",
    "subscription id": "subscription_id",
    "subscription_id": "subscription_id",
    # Date
    "date": "date",
    "usagedatetime": "date",
    "billingperiodstartdate": "date",
    # Resource
    "resourceid": "resource_id",
    "resource id": "resource_id",
    "instanceid": "resource_id",
    "instance id": "resource_id",
    # Service
    "servicename": "service_name",
    "service name": "service_name",
    "consumedservice": "service_name",
    "consumed service": "service_name",
    "serviceinfo1": "service_name",
    # Service tier
    "servicetier": "service_tier",
    "service tier": "service_tier",
    "metercategory": "service_tier",
    "meter category": "service_tier",
    "metername": "meter_name",
    "meter name": "meter_name",
    # Product
    "productname": "product_name",
    "product name": "product_name",
    "product": "product_name",
    # Quantity / cost
    "quantity": "quantity",
    "usagequantity": "quantity",
    "usage quantity": "quantity",
    "unitprice": "unit_price",
    "unit price": "unit_price",
    "effectiveprice": "effective_price",
    "effective price": "effective_price",
    "payasyougoprice": "payg_price",
    "pay as you go price": "payg_price",
    "paygprice": "payg_price",
    "costinbillingcurrency": "cost",
    "cost in billing currency": "cost",
    "pretaxcost": "cost",
    "pre-tax cost": "cost",
    "cost": "cost",
    "billingcurrencycode": "currency",
    "billing currency code": "currency",
    "billingcurrency": "currency",
    "currency": "currency",
    "currencycode": "currency",
    # Additional info
    "additionalinfo": "additional_info",
    "additional info": "additional_info",
    "additionalinformation": "additional_info",
    "additional information": "additional_info",
}

# Service name / tier patterns that indicate Azure OpenAI or Foundry model usage
OPENAI_SERVICE_PATTERNS = [
    "azure openai",
    "openai",
    "cognitive services",   # OpenAI is under Cognitive Services umbrella
    "cognitiveservices",    # ConsumedService value: "Microsoft.CognitiveServices"
    "foundry models",       # MeterCategory for Azure AI Foundry model deployments
    "azure ai",             # Azure AI Foundry and related Azure AI services
]

OPENAI_TIER_PATTERNS = [
    "azure openai",
    "openai",
    "foundry models",
    "azure ai",
]

OPENAI_PRODUCT_PATTERNS = [
    "azure openai",
    "openai",
    "gpt",
    "dall-e",
    "whisper",
    "text-embedding",
    "ada",
    "davinci",
    "curie",
    "babbage",
]


# ---------------------------------------------------------------------------
# Date range helpers
# ---------------------------------------------------------------------------

def get_previous_month_range(reference: date | None = None) -> tuple[date, date]:
    """Return (start, end) dates for the previous calendar month.

    Parameters
    ----------
    reference:
        The reference date.  Defaults to today (UTC).

    Returns
    -------
    tuple[date, date]
        ``(first_day_of_last_month, last_day_of_last_month)``
    """
    if reference is None:
        reference = date.today()

    first_of_current = reference.replace(day=1)
    last_of_previous = first_of_current - timedelta(days=1)
    first_of_previous = last_of_previous.replace(day=1)

    return first_of_previous, last_of_previous


def parse_month_arg(month_arg: str) -> tuple[date, date]:
    """Parse the ``--month`` CLI argument.

    Parameters
    ----------
    month_arg:
        Either the literal string ``"last"`` or an ISO year-month ``"YYYY-MM"``.

    Returns
    -------
    tuple[date, date]
        ``(start_date, end_date)`` for the specified month.

    Raises
    ------
    ValueError
        If *month_arg* cannot be parsed.
    """
    if month_arg.lower() == "last":
        return get_previous_month_range()

    parts = month_arg.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --month value: '{month_arg}'.  "
            "Expected 'last' or 'YYYY-MM' (e.g. '2024-12')."
        )
    year, month = int(parts[0]), int(parts[1])
    _, last_day = monthrange(year, month)
    return date(year, month, 1), date(year, month, last_day)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Cost Details columns to canonical names.

    Unknown columns are kept as-is.  When multiple raw columns map to the
    same canonical name, only the first occurrence is kept.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = col.lower().strip()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)

    # Drop duplicate canonical columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def filter_openai_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that relate to Azure OpenAI or Azure AI Foundry model usage.

    Matches on ``service_tier``, ``service_name``, ``product_name``, or
    ``meter_name``.  Foundry-based model deployments are identified by
    ``service_tier`` = "Foundry Models" and ``service_name`` =
    "Microsoft.CognitiveServices".
    """
    if df.empty:
        return df

    mask = pd.Series([False] * len(df), index=df.index)

    for col in ("service_tier", "service_name", "product_name", "meter_name"):
        if col not in df.columns:
            continue
        col_lower = df[col].fillna("").str.lower()
        for pattern in (
            OPENAI_SERVICE_PATTERNS
            if col in ("service_name", "service_tier")
            else OPENAI_PRODUCT_PATTERNS
        ):
            mask |= col_lower.str.contains(pattern, regex=False)

    filtered = df[mask].copy()
    logger.debug(
        "AI (OpenAI/Foundry) filter: %d/%d rows retained", len(filtered), len(df)
    )
    return filtered


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float, replacing unparseable values with NaN."""
    numeric_cols = [
        "quantity",
        "unit_price",
        "effective_price",
        "payg_price",
        "cost",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_derived_columns(
    df: pd.DataFrame,
    retail_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add discount and effective-price-per-1M columns.

    Parameters
    ----------
    df:
        Normalised DataFrame (already filtered to OpenAI rows).
    retail_prices:
        Optional mapping of meter name → retail unit price, used when
        ``payg_price`` is absent.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - ``discount_abs``   : absolute discount per unit (payg − effective)
        - ``discount_pct``   : discount percentage
        - ``effective_price_per_1m`` : effective price per 1 000 000 tokens
    """
    if df.empty:
        return df

    # Determine reference price for discount computation
    if "payg_price" in df.columns:
        ref_price = df["payg_price"].fillna(0.0)
    elif retail_prices and "meter_name" in df.columns:
        ref_price = (
            df["meter_name"]
            .fillna("")
            .str.lower()
            .map(retail_prices)
            .fillna(0.0)
        )
    else:
        ref_price = pd.Series([0.0] * len(df), index=df.index)

    eff = df.get("effective_price", pd.Series([0.0] * len(df), index=df.index)).fillna(0.0)

    df["discount_abs"] = (ref_price - eff).clip(lower=0.0)
    df["discount_pct"] = (
        (df["discount_abs"] / ref_price.replace(0, float("nan"))) * 100
    ).fillna(0.0)

    # Price per 1M tokens (quantity assumed to be in tokens)
    qty = df.get("quantity", pd.Series([0.0] * len(df), index=df.index)).fillna(0.0)
    df["effective_price_per_1m"] = (eff * 1_000_000).where(qty > 0, other=0.0)

    return df


def aggregate_by_subscription_meter(df: pd.DataFrame) -> pd.DataFrame:
    """Group and aggregate the normalised data.

    Groups by: subscription_id, meter_name, product_name, date.
    Computes:
      - total_quantity (tokens)
      - total_cost
      - avg_effective_price (quantity-weighted)
      - avg_payg_price (quantity-weighted, if present)
      - discount_abs, discount_pct

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame.
    """
    if df.empty:
        return df

    group_keys = [
        c
        for c in ["subscription_id", "meter_name", "product_name", "date"]
        if c in df.columns
    ]
    if not group_keys:
        logger.warning("No grouping columns found; returning raw data")
        return df

    # Ensure numeric
    df = coerce_numeric(df)

    qty_col = "quantity" if "quantity" in df.columns else None
    cost_col = "cost" if "cost" in df.columns else None
    eff_col = "effective_price" if "effective_price" in df.columns else None
    payg_col = "payg_price" if "payg_price" in df.columns else None

    def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
        w_total = weights.sum()
        if w_total == 0:
            return float("nan")
        return float((values * weights).sum() / w_total)

    records = []
    for keys, grp in df.groupby(group_keys, dropna=False):
        key_dict: dict[str, Any] = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))

        qty = grp[qty_col].fillna(0.0) if qty_col else pd.Series([0.0])
        total_qty = float(qty.sum())
        total_cost = float(grp[cost_col].fillna(0.0).sum()) if cost_col else float("nan")

        avg_eff = weighted_avg(grp[eff_col].fillna(0.0), qty) if eff_col else float("nan")
        avg_payg = weighted_avg(grp[payg_col].fillna(0.0), qty) if payg_col else float("nan")

        discount_abs = max(avg_payg - avg_eff, 0.0) if (not pd.isna(avg_payg) and not pd.isna(avg_eff)) else 0.0
        discount_pct = (
            (discount_abs / avg_payg) * 100.0
            if (not pd.isna(avg_payg) and avg_payg > 0)
            else 0.0
        )

        currency = (
            grp["currency"].dropna().iloc[0]
            if "currency" in grp.columns and not grp["currency"].dropna().empty
            else ""
        )

        records.append(
            {
                **key_dict,
                "total_quantity": total_qty,
                "total_cost": total_cost,
                "avg_effective_price": avg_eff,
                "avg_payg_price": avg_payg,
                "discount_abs": discount_abs,
                "discount_pct": discount_pct,
                "effective_price_per_1m": avg_eff * 1_000_000 if avg_eff else 0.0,
                "currency": currency,
            }
        )

    return pd.DataFrame(records)


def normalize_dataframe(
    df: pd.DataFrame,
    retail_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Full normalization pipeline: rename → filter → coerce → enrich → aggregate.

    Parameters
    ----------
    df:
        Raw DataFrame from Cost Details CSV.
    retail_prices:
        Optional retail prices for discount computation fallback.

    Returns
    -------
    pd.DataFrame
        Clean, aggregated DataFrame ready for parquet storage.
    """
    if df.empty:
        return df

    df = normalize_columns(df)
    df = filter_openai_rows(df)

    if df.empty:
        logger.info("No Azure OpenAI / Foundry model rows found in this dataset")
        return df

    df = coerce_numeric(df)
    df = add_derived_columns(df, retail_prices=retail_prices)
    df = aggregate_by_subscription_meter(df)
    df = add_token_type(df)
    return df
