from pathlib import Path
import re

root = Path(r"D:\Code\0622\optimal_demo\reference_papers\md")

def head_tail(path, n=80):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    # find contribution / conclusion / method keywords
    keys = []
    for i,l in enumerate(lines):
        if re.search(r'(?i)contribution|conclusion|case study|simulation|baseline|TD3|price-taker|day-ahead|safe exploration|hierarchical|reward|MDP|algorithm', l):
            keys.append((i+1, l[:140]))
    return "\n".join(lines[:n]), keys[:40], len(lines), len(text)

targets = [
"Collaborative_scheduling_optimization_of_hydrogen-enhanced_integrated_energy_sys_md",
"Dynamic_energy_dispatch_strategy_for_integrated_energy_system_based_on_improved_md",
"Dynamic_Energy_Dispatch_Strategy_for_Integrated_Energy_System_Based_on_Constrain_md",
"Multi-agent_deep_reinforcement_learning_for_efficient_multi-timescale_bidding_of_md",
"Multi-agent_hierarchical_reinforcement_learning_for_energy_management_md",
"Optimal_price-taker_bidding_strategy_of_distributed_energy_storage_systems_in_th_md",
"A_market_feedback_framework_for_improved_estimates_of_the_arbitrage_value_of_ene_md",
"Optimal_dispatch_of_integrated_energy_system_based_on_deep_reinforcement_learnin_md",
"safe_Exploration_in_reinforcement_learning_a_generalized_formulation_and_algorit_md",
"Safe_Exploration_of_State_and_Action_Spaces_in_Reinforcement_Learning_md",
"addressing_function_approximation_error_in_actor-critic_methods_md",
"Liquid_air_energy_storage_md",
"Virtual_power_plant_participation_in_day-ahead_and_futures_markets_using_a_deep_md",
"Pricing_Strategy_for_Regional_Integrated_Energy_System_Considering_Privacy_Based_md",
"Incentive-oriented_power_carbon_emissions_trading-tradable_green_certificate_int_md",
"Decentralized_coordinated_planning_model_for_integrated_energy_systems_under_sea_md",
]
for name in targets:
    d = root/name
    doc = d/"document.md"
    print("="*90)
    print(name)
    if not doc.exists():
        print("MISSING"); continue
    head, keys, nlines, nbytes = head_tail(doc)
    print(f"lines={nlines} bytes={nbytes}")
    print("--- HEAD ---")
    print(head[:3500])
    print("--- KEY LINES ---")
    for i,l in keys[:25]:
        print(f"{i}: {l}")
