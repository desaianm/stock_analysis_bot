"""Smoke: instantiate UndervaluedAnalysisFlow, run startup ritual, check state."""

from dotenv import load_dotenv

from stockbot.audit import clear_state, read_state, write_state
from stockbot.flows.undervalued import (
    UndervaluedAnalysisFlow,
    ValueScreeningPreferences,
)

load_dotenv()


def main():
    prefs = ValueScreeningPreferences(
        max_price=50,
        min_price=5,
        min_volume=500_000,
        max_pe=20,
        min_market_cap=300_000_000,
        min_current_ratio=1.5,
        max_debt_equity=1.5,
        price_vs_high=0.4,
    )

    clear_state("undervalued")
    flow = UndervaluedAnalysisFlow(prefs)
    print("Flow init OK")
    print(f"  reasoning_model = {flow.reasoning_model_id}")
    print(f"  summary_model   = {flow.summary_model_id}")
    print(f"  extraction_model= {flow.extraction_model_id}")
    print(
        f"  agents: screening={flow.screening_agent.name}, "
        f"turnaround={flow.turnaround_agent.name}, "
        f"reddit={flow.reddit_sentiment_agent.name}, "
        f"bottleneck={flow.bottleneck_research_agent.name}"
    )
    print(f"  tools wired to screening: {len(flow.screening_agent.tools)}")

    print("\n--- startup ritual ---")
    flow._run_startup_ritual()
    print("--- end ritual ---")

    write_state("undervalued", run_id=999, phase="smoke", note="init test")
    state = read_state("undervalued")
    print(f"\nState after smoke write: {state}")
    clear_state("undervalued")
    print("Smoke complete.")


if __name__ == "__main__":
    main()
