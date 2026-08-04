from pathlib import Path
import re
root = Path(r"D:\Code\0622\optimal_demo\reference_papers\md")

def extract_sections(name, patterns, max_chars=2500):
    d = root/name
    doc = d/"document.md"
    text = doc.read_text(encoding="utf-8", errors="replace")
    print("="*80, name[:60])
    # first 15 lines title
    print("TITLE LINE:", text.splitlines()[0][:120])
    for pat in patterns:
        m = re.search(pat, text, re.I|re.S)
        if m:
            snip = m.group(0)[:max_chars]
            print(f"\n-- match {pat[:50]} --")
            print(snip)
        else:
            print(f"\n-- NO match {pat[:50]}")

# constrained RL CSEE
extract_sections(
"Dynamic_Energy_Dispatch_Strategy_for_Integrated_Energy_System_Based_on_Constrain_md",
[r"## 1\. Introduction.{0,1200}", r"(contribution|main contributions).{0,1200}", r"(constrained|CPO|safety|penalty).{0,800}", r"## 5\..{0,1500}"],
)

extract_sections(
"Multi-agent_deep_reinforcement_learning_for_efficient_multi-timescale_bidding_of_md",
[r"## Abstract.{0,1200}|A B S T R A C T.{0,1200}", r"(contribution|main contributions).{0,1500}", r"(day-ahead|real-time|price-taker|bidding).{0,600}", r"(baseline|compared with|compared to).{0,800}"],
)

extract_sections(
"Multi-agent_hierarchical_reinforcement_learning_for_energy_management_md",
[r"A B S T R A C T.{0,1500}", r"(contribution|contributions).{0,1500}", r"(hierarchical|high-level|low-level).{0,800}", r"(case study|results).{0,800}"],
)

extract_sections(
"Optimal_price-taker_bidding_strategy_of_distributed_energy_storage_systems_in_th_md",
[r"(price-taker|bidding|spot market).{0,900}", r"(contribution|methodology|model).{0,1200}", r"(case|simulation|result).{0,800}"],
)

extract_sections(
"A_market_feedback_framework_for_improved_estimates_of_the_arbitrage_value_of_ene_md",
[r"A B S T R A C T.{0,1500}|Abstract.{0,1500}", r"(price-taker|feedback|limitation).{0,1000}", r"(conclusion).{0,800}"],
)

extract_sections(
"Optimal_dispatch_of_integrated_energy_system_based_on_deep_reinforcement_learnin_md",
[r"A B S T R A C T.{0,1200}|Abstract.{0,1200}", r"(algorithm|TD3|DQN|DDPG|PPO).{0,600}", r"(conclusion).{0,600}"],
)
