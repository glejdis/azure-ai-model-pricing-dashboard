"""
test_parsing.py – Tests for CSV normalization and OpenAI row filtering.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.collect.normalize import (
    COLUMN_ALIASES,
    add_derived_columns,
    classify_token_type,
    coerce_numeric,
    filter_openai_rows,
    normalize_columns,
    normalize_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(**kwargs: list) -> pd.DataFrame:
    """Helper to quickly build a DataFrame from keyword-list arguments."""
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# normalize_columns
# ---------------------------------------------------------------------------

class TestNormalizeColumns:
    def test_renames_known_columns(self) -> None:
        df = make_df(
            SubscriptionId=["sub-1"],
            MeterName=["Completion Tokens"],
            EffectivePrice=["0.002"],
            Quantity=["1000"],
            Cost=["2.0"],
        )
        result = normalize_columns(df)
        assert "subscription_id" in result.columns
        assert "meter_name" in result.columns
        assert "effective_price" in result.columns
        assert "quantity" in result.columns
        assert "cost" in result.columns

    def test_leaves_unknown_columns(self) -> None:
        df = make_df(MyCustomCol=["x"])
        result = normalize_columns(df)
        assert "MyCustomCol" in result.columns

    def test_case_insensitive(self) -> None:
        df = make_df(SUBSCRIPTIONID=["sub-2"])
        result = normalize_columns(df)
        assert "subscription_id" in result.columns

    def test_all_defined_aliases_have_correct_canonical(self) -> None:
        """Every alias key must map to a known canonical name."""
        canonical_set = {
            "subscription_id", "date", "resource_id", "service_name",
            "service_tier", "product_name", "meter_name", "quantity",
            "unit_price", "effective_price", "payg_price", "cost",
            "currency", "additional_info",
        }
        for alias, canonical in COLUMN_ALIASES.items():
            assert canonical in canonical_set, (
                f"Alias '{alias}' maps to unknown canonical '{canonical}'"
            )


# ---------------------------------------------------------------------------
# filter_openai_rows
# ---------------------------------------------------------------------------

class TestFilterOpenAIRows:
    def test_keeps_openai_service_tier(self) -> None:
        df = make_df(
            service_tier=["Azure OpenAI", "Storage", "Compute"],
            quantity=["100", "200", "300"],
        )
        result = filter_openai_rows(df)
        assert len(result) == 1
        assert result["service_tier"].iloc[0] == "Azure OpenAI"

    def test_keeps_openai_in_product_name(self) -> None:
        df = make_df(
            product_name=["Azure OpenAI GPT-4o", "Azure Blob Storage"],
            quantity=["10", "20"],
        )
        result = filter_openai_rows(df)
        assert len(result) == 1
        assert "openai" in result["product_name"].iloc[0].lower()

    def test_no_openai_rows_returns_empty(self) -> None:
        df = make_df(
            service_tier=["Storage", "Compute", "Network"],
            quantity=["10", "20", "30"],
        )
        result = filter_openai_rows(df)
        assert result.empty

    def test_empty_input_returns_empty(self) -> None:
        result = filter_openai_rows(pd.DataFrame())
        assert result.empty

    def test_cognitive_services_included(self) -> None:
        df = make_df(
            service_name=["Cognitive Services"],
            product_name=["OpenAI GPT-4"],
            quantity=["50"],
        )
        result = filter_openai_rows(df)
        assert len(result) == 1

    def test_case_insensitive_matching(self) -> None:
        df = make_df(
            service_tier=["AZURE OPENAI", "azure openai", "Azure OpenAI"],
            quantity=["1", "2", "3"],
        )
        result = filter_openai_rows(df)
        assert len(result) == 3

    def test_foundry_models_service_tier_included(self) -> None:
        """Rows with service_tier='Foundry Models' (Azure AI Foundry) must be kept."""
        df = make_df(
            service_tier=["Foundry Models", "Storage"],
            service_name=["Microsoft.CognitiveServices", "Microsoft.Storage"],
            product_name=["Azure OpenAI GPT5", "Blob Storage"],
            quantity=["100", "200"],
        )
        result = filter_openai_rows(df)
        assert len(result) == 1
        assert result["service_tier"].iloc[0] == "Foundry Models"

    def test_cognitiveservices_no_space_included(self) -> None:
        """'Microsoft.CognitiveServices' (no space) in service_name must be kept."""
        df = make_df(
            service_name=["Microsoft.CognitiveServices", "Microsoft.Storage"],
            service_tier=["Foundry Models", "Storage"],
            quantity=["50", "100"],
        )
        result = filter_openai_rows(df)
        assert len(result) == 1
        assert result["service_name"].iloc[0] == "Microsoft.CognitiveServices"

    def test_foundry_row_from_csv_example(self) -> None:
        """End-to-end check using field values from the problem-statement CSV example."""
        df = make_df(
            service_tier=["Foundry Models"],
            service_name=["Microsoft.CognitiveServices"],
            product_name=["Azure OpenAI GPT5 - 5.4 nano Opt Gl - SE Central"],
            meter_name=["5.4 nano Opt Gl 1M Tokens"],
            quantity=["1.25"],
        )
        result = filter_openai_rows(df)
        assert len(result) == 1

    def test_non_ai_rows_excluded(self) -> None:
        """Rows with no AI-related identifiers must remain excluded."""
        df = make_df(
            service_tier=["Storage", "Compute", "Network"],
            service_name=["Microsoft.Storage", "Microsoft.Compute", "Microsoft.Network"],
            product_name=["Blob Storage", "Virtual Machine", "VPN Gateway"],
            quantity=["10", "20", "30"],
        )
        result = filter_openai_rows(df)
        assert result.empty


# ---------------------------------------------------------------------------
# coerce_numeric
# ---------------------------------------------------------------------------

class TestCoerceNumeric:
    def test_parses_float_strings(self) -> None:
        df = make_df(quantity=["100.5"], cost=["2.5"], effective_price=["0.002"])
        result = coerce_numeric(df)
        assert result["quantity"].dtype.kind == "f"
        assert result["cost"].iloc[0] == pytest.approx(2.5)
        assert result["effective_price"].iloc[0] == pytest.approx(0.002)

    def test_invalid_values_become_nan(self) -> None:
        df = make_df(quantity=["not_a_number"], cost=["$2.5"])
        result = coerce_numeric(df)
        assert pd.isna(result["quantity"].iloc[0])
        # "$2.5" cannot be parsed to float
        assert pd.isna(result["cost"].iloc[0])

    def test_missing_columns_ignored(self) -> None:
        df = make_df(some_other_col=["hello"])
        result = coerce_numeric(df)
        assert "some_other_col" in result.columns


# ---------------------------------------------------------------------------
# add_derived_columns
# ---------------------------------------------------------------------------

class TestAddDerivedColumns:
    def _base_df(self) -> pd.DataFrame:
        return make_df(
            effective_price=[0.002, 0.003],
            payg_price=[0.004, 0.004],
            quantity=[1000.0, 500.0],
        )

    def test_discount_abs(self) -> None:
        df = coerce_numeric(self._base_df())
        result = add_derived_columns(df)
        assert result["discount_abs"].iloc[0] == pytest.approx(0.002)
        assert result["discount_abs"].iloc[1] == pytest.approx(0.001)

    def test_discount_pct(self) -> None:
        df = coerce_numeric(self._base_df())
        result = add_derived_columns(df)
        # Row 0: (0.004 - 0.002) / 0.004 = 50%
        assert result["discount_pct"].iloc[0] == pytest.approx(50.0)

    def test_discount_non_negative(self) -> None:
        """Discount should never be negative."""
        df = make_df(
            effective_price=[0.010],   # higher than payg (unusual)
            payg_price=[0.005],
            quantity=[100.0],
        )
        df = coerce_numeric(df)
        result = add_derived_columns(df)
        assert result["discount_abs"].iloc[0] >= 0.0

    def test_effective_price_per_1m(self) -> None:
        df = make_df(
            effective_price=[0.002],
            payg_price=[0.004],
            quantity=[1000.0],
        )
        df = coerce_numeric(df)
        result = add_derived_columns(df)
        assert result["effective_price_per_1m"].iloc[0] == pytest.approx(0.002 * 1_000_000)

    def test_retail_prices_fallback(self) -> None:
        """When payg_price is absent, retail_prices dict is used."""
        df = make_df(
            meter_name=["completion tokens"],
            effective_price=[0.002],
            quantity=[1000.0],
        )
        df = coerce_numeric(df)
        retail = {"completion tokens": 0.006}
        result = add_derived_columns(df, retail_prices=retail)
        assert result["discount_abs"].iloc[0] == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# Full normalize_dataframe pipeline
# ---------------------------------------------------------------------------

class TestNormalizeDataframe:
    def _raw_csv_df(self) -> pd.DataFrame:
        """Simulate a raw Cost Details CSV as a DataFrame."""
        return make_df(
            SubscriptionId=["sub-1", "sub-1", "sub-2"],
            Date=["2024-04-01", "2024-04-02", "2024-04-01"],
            MeterName=[
                "Azure OpenAI Completion Tokens",
                "Azure OpenAI Prompt Tokens",
                "Azure OpenAI Completion Tokens",
            ],
            ServiceTier=["Azure OpenAI", "Azure OpenAI", "Azure OpenAI"],
            ProductName=["Azure OpenAI", "Azure OpenAI", "Azure OpenAI"],
            Quantity=["1000", "500", "2000"],
            EffectivePrice=["0.002", "0.001", "0.002"],
            PayAsYouGoPrice=["0.004", "0.002", "0.004"],
            Cost=["2.0", "0.5", "4.0"],
            BillingCurrencyCode=["USD", "USD", "USD"],
        )

    def test_output_has_expected_columns(self) -> None:
        df = self._raw_csv_df()
        result = normalize_dataframe(df)
        assert not result.empty
        for col in ("total_quantity", "total_cost", "avg_effective_price"):
            assert col in result.columns, f"Missing column: {col}"

    def test_filters_to_openai_only(self) -> None:
        df = self._raw_csv_df()
        # Add a non-OpenAI row
        extra = make_df(
            SubscriptionId=["sub-1"],
            Date=["2024-04-01"],
            MeterName=["Blob Storage LRS"],
            ServiceTier=["Storage"],
            ProductName=["Azure Storage"],
            Quantity=["999"],
            EffectivePrice=["0.01"],
            PayAsYouGoPrice=["0.02"],
            Cost=["9.99"],
            BillingCurrencyCode=["USD"],
        )
        combined = pd.concat([df, extra], ignore_index=True)
        result = normalize_dataframe(combined)
        # Storage row should be excluded
        if "meter_name" in result.columns:
            assert not any(
                "blob" in str(m).lower() for m in result["meter_name"].tolist()
            )

    def test_empty_df_returns_empty(self) -> None:
        result = normalize_dataframe(pd.DataFrame())
        assert result.empty

    def test_discount_computed(self) -> None:
        df = self._raw_csv_df()
        result = normalize_dataframe(df)
        assert "discount_pct" in result.columns
        # All rows have payg_price > effective_price → positive discount
        assert (result["discount_pct"] >= 0).all()


# ---------------------------------------------------------------------------
# Token type classification (cached / input / output)
# ---------------------------------------------------------------------------

class TestClassifyTokenType:
    def test_cached_input_detected(self) -> None:
        # "cached" must win over "input" since the name contains both.
        assert classify_token_type("Cached Input Tokens") == "Cached Input Tokens"
        assert classify_token_type("gpt 4o cached Inp glbl Tokens") == "Cached Input Tokens"
        assert classify_token_type("Input Cached Tokens") == "Cached Input Tokens"

    def test_input_detected(self) -> None:
        assert classify_token_type("Azure OpenAI Prompt Tokens") == "Input Tokens"
        assert classify_token_type("gpt 4o Input global Tokens") == "Input Tokens"

    def test_output_detected(self) -> None:
        assert classify_token_type("Azure OpenAI Completion Tokens") == "Output Tokens"
        assert classify_token_type("gpt 4o Output global Tokens") == "Output Tokens"

    def test_other_categories(self) -> None:
        assert classify_token_type("gpt 4o PTU") == "PTU"
        assert classify_token_type("DALL-E Images") == "Image"
        assert classify_token_type("Some Random Meter") == "Other"

    def test_empty_or_none(self) -> None:
        assert classify_token_type("") == "Unknown"
        assert classify_token_type(None) == "Unknown"


class TestTokenTypeInPipeline:
    def test_normalize_adds_token_type_with_cached(self) -> None:
        df = make_df(
            SubscriptionId=["sub-1", "sub-1", "sub-1"],
            Date=["2024-04-01", "2024-04-01", "2024-04-01"],
            MeterName=[
                "Azure OpenAI Prompt Tokens",
                "Azure OpenAI Completion Tokens",
                "Azure OpenAI Cached Input Tokens",
            ],
            ServiceTier=["Azure OpenAI", "Azure OpenAI", "Azure OpenAI"],
            ProductName=["Azure OpenAI", "Azure OpenAI", "Azure OpenAI"],
            Quantity=["1000", "500", "800"],
            EffectivePrice=["0.001", "0.002", "0.0005"],
            Cost=["1.0", "1.0", "0.4"],
            BillingCurrencyCode=["USD", "USD", "USD"],
        )
        result = normalize_dataframe(df)
        assert "token_type" in result.columns
        types = set(result["token_type"].tolist())
        assert "Cached Input Tokens" in types
        assert "Input Tokens" in types
        assert "Output Tokens" in types
