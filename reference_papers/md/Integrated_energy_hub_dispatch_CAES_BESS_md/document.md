# Integrated energy hub dispatch with a multi-mode CAES–BESS hybrid system



--- Page 1 ---

Contents lists available at ScienceDirect
Applied Energy
journal homepage: www.elsevier.com/locate/apenergy
Integrated energy hub dispatch with a multi-mode CAES–BESS hybrid
system: An option-based hierarchical reinforcement learning approach
Feifei Cui a, Dou Ana,∗, Huan Xib
a Faculty of Electronics and Information, Xi’an Jiaotong University, Xi’an, 710049, China
b Key Laboratory of Thermo-Fluid Science and Engineering of Ministry of Education, School of Energy and Power Engineering, Xi’an Jiaotong
University, Xi’an, 710049, China
G R A P H I C A L A B S T R A C T
A R T I C L E I N F O
Keywords:
Integrated energy management system
Hybrid energy storage
Energy dispatch
Hierarchical reinforcement learning
A B S T R A C T
The high penetration of renewable energy sources (RES) in power generation has driven demand for advanced
integrated energy management systems (IEMS). In this study, to address the challenges of insufficient
adaptability to dynamic supply–demand, a multi-type energy IEMS combining compressed air energy storage
(CAES) and a battery energy storage system (BESS) is proposed, which operates under a multi-mode energy
storage (MES) mechanism with rapid response, long-term balance, and synergic adjustment modes. To address
the complexity of sequential decisions, an option-critic based twin delayed deep deterministic policy gradient
(OCTD3) algorithm is firstly proposed within the hierarchical reinforcement learning (HRL) framework,
∗ Corresponding author.
E-mail address: douan2017@xjtu.edu.cn (D. An).
https://doi.org/10.1016/j.apenergy.2024.123950
Received 13 May 2024; Received in revised form 24 June 2024; Accepted 14 July 2024
Applied Energy 374 (2024) 123950 
Available online 26 July 2024 
0306-2619/© 2024 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies. 

--- Page 2 ---

F. Cui et al.
enhancing efficiency through encapsulation of subtasks within "options". Additionally, model precision is
refined by fitting the electricity–gas–heat conversion dynamics of CAES under off-design conditions. Dispatch
tasks are modeled as an option-based Semi-Markov Decision Process (SMDP) and optimized by the OCTD3 to
improve the power fluctuations index (PFI), comprehensive costs index (CCI), and system response synergy
index (SRSI). Comparative simulations reveal that the MES mechanism boosts SRSI by 91.8%, showcasing
high adaptability to varied supply–demand scenarios. The OCTD3 algorithm develops five hybrid strategies
for CAES–BESS across three modes, effectively cutting costs by reducing electricity purchases and fluctuations
expenses, and lowering PFI by 42.2% through balancing peak–valley loads and swiftly responding to transient
shifts.
1. Introduction
Global climate change has driven the implementation of environ-
mental actions worldwide, such as ‘‘carbon peak and carbon neutrality’’
goals announced by China [1]. Extensive integration of RES into energy
systems is one of the key strategies to reduce greenhouse gas emissions.
However, the variability and unpredictability of RES and demand make
traditional energy systems in urgent need of upgrading. This drives the
development of IEMS to optimize the energy production, storage, and
consumption. Despite these efforts, regions like California still expe-
rienced blackouts caused by surging thermal demand and insufficient
solar energy availability in 2020 [2], highlighting the need for more
flexible integration structures and more efficient dispatch strategies of
IEMS.
Energy storage technology [3] is essential for mitigating supply–
demand imbalances of IEMS. Common storage technologies include
BESS, CAES [4], flywheel energy storage (FES), supercapacitors (SC),
pumped thermal energy storage [5], hydrogen energy storage (HES),
etc. However, using them individually cannot meet both the high-
power and high-energy storage requirements of IEMS simultaneously.
Thus, hybrid energy storage systems (HESS) [6] have been devel-
oped, combining multiple storage forms to optimize energy flows and
enhance performance, such as SC/BESS, FES/BESS, HES/BESS [7],
HES/FES, and CAES/BESS. Argyrou et al. [8] combined the high-
capacity of BESS with the rapid response capabilities of SC, enhancing
the grid’s ability to consume RES. Jahanbin et al. [9] integrated HES
for long-term supply with BESS for rapid response, stabilizing energy
during peak demands in an academic setting. Yousri et al. [10] utilized
the high-capacity of HES and the rapid response characteristics of BESS
to provide dispatch strategies and demand response for microgrids.
Zeynali et al. [11] integrated BESS and CAES to participate in system
regulation that responds to both thermal and electrical user demands.
In IEMS, advanced dispatch methods are crucial for coordinating
RES generation, storage devices, and load demands, as well as for
adapting to changes such as weather fluctuations and market prices.
Existing traditional energy dispatch methods are categorized into three
types: deterministic algorithms [12], heuristic algorithms [13], and
hybrid algorithms [14]. For instance, Morshed et al. [15] combined
the imperialist competition with sequential quadratic method to op-
timize economic load dispatch. Iribarren et al. [16] employed the
mixed-integer linear programming to enhance thermal systems in res-
idential buildings, integrating various energy storage for better cost-
effectiveness and efficiency. However, traditional methods [17] rely
on complex mathematical models and struggle with the uncertainty
factors, making it challenging to develop accurate, real-time dispatch
strategies.
Deep reinforcement learning (DRL) [18] has significant advantages
for addressing dynamic and complex tasks due to their ability to
generate strategies through feedback interactions without the need for
mathematical models. In IEMS, mainstream DRL algorithms such as
TD3 [19], deep deterministic policy gradient (DDPG) [20], deep Q-
Network (DQN) [21], and proximal policy optimization (PPO) [22] are
increasingly applied. Wen et al. [23] developed a data-driven dispatch
method for a hydrogen/ammonia energy hub using a modified DQN,
focusing on the flexible operation of multi-type energy systems with
RES and energy storage. Zhang et al. [24] optimized dispatch strategies
for integrated electricity and natural gas systems using DDPG, formu-
lating it as a dynamic Markov decision process to enhance efficiency.
Zhang et al. [25] used a soft actor–critic (SAC) approach to dispatch
power, heat, and gas energies, aiming to balance costs and ensure
reliable energy supply. Liu et al. [26] presented a TD3-based framework
to handle the dynamics of IEMS in smart buildings, targeting cost
minimization and user satisfaction maximization. Wang et al. [27]
innovatively combined imitation learning with PPO, optimizing the
performance of building energy systems. To address the challenge
of capturing physical constraints in complex energy systems, Wang
et al. [28] proposed a constrained SAC method. Qiu et al. [29] em-
ployed a two-tier HRL, where the high-level decisions involve switching
between energy networks, and the low-level decisions focus on route
repair, thereby enhancing the system’s resilience.
Table 1 analyzes the aforementioned literature review. Although
existing research have improved the stability and efficiency of IEMS
by integrating HESS and applying DRL algorithms, challenges remain
in terms of adaptability to rapidly changing supply–demand scenarios
and the complexity of sequential decision-making processes. These
issues can be summarized as follows: (1) The current IEMS integrates
HESS operating in a single mode, not fully exploiting the synergies
of heterogeneous energy storage types. Thus, the responsiveness of
IEMS under peak load or extreme weather conditions needs further
enhancement. (2) The current IEMS system features high levels of
RES integration, diverse energy flows, and varied demand-side. These
complexities challenge traditional single-layer DRL approaches due to
expansive state spaces and intricate sequential decision-making paths,
leading to suboptimal decisions and low learning efficiencies.
To address the above issues, this study integrates a hybrid CAES–
BESS system equipped with a MES mechanism into the multi-type
energy IEMS. This setup offers three distinct energy storage modes
to dynamically balance energy supply and demand. Addressing the
complexities of indeterminate task durations and state-dependent de-
cisions in proposed IEMS, an option-based HRL dispatch algorithm
OCTD3 is innovatively proposed. This algorithm improves optimization
and efficiency by structuring subtasks into distinct ‘‘options’’, each
corresponding to an operational mode of the energy storage system,
enhancing the capability to adapt to dynamic and fluctuating energy
scenarios. To the best of our knowledge, it is the first time that the
IEMS dispatch strategy is developed by a option-based HRL approach.
The main contributions are summarized as follows:
(1) A multi-type energy IEMS is constructed, integrating electricity,
hydrogen, thermal, and gas energies. A MES mechanism for hybrid
CAES–BESS system is proposed to flexibly address dynamic energy
supply–demand, offering three modes: rapid response (RR), long-term
energy balance (LEB), and synergic adjustment (SA).
(2) The modeling of IEMS components is refined by fitting the re-
lationships for electricity–gas–heat conversion in a cogeneration CAES
system under off-design conditions. Three indices – PFI, CCI, and
SRSI – are defined to evaluate the stability, cost-effectiveness, and
responsiveness of the strategy.
(3) The dispatch process is modeled as an option-based SMDP, effec-
tively capturing the temporal decision-making complexities marked by
indeterminate durations and state-dependent characteristics. An hierar-
chical OCTD3 dispatch algorithm is innovatively proposed, enhancing
Applied Energy 374 (2024) 123950 
2 

--- Page 3 ---

F. Cui et al.
Nomenclature
Abbreviation
AST Air storage tank
BESS Battery energy storage system
CAES Compressed Air Energy Storage
CCI Comprehensive costs index
CHS Combined heat pump system
DDPG Deep deterministic policy gradient
DQN Deep Q-Network
DRL Deep reinforcement learning
FES Flywheel energy storage
HES Hydrogen energy storage
HESS Hybrid energy storage systems
HRL Hierarchical reinforcement learning
IEMS Integrated energy management systems
LEB Long-term energy balance
MES Multi-mode energy storage
OCTD3 Option-critic based twin delayed deep deterministic
policy gradient
PEM Proton exchange membrane
PFI Power fluctuations index
PPO Proximal policy optimization
PV Photovoltaic panels
RES Renewable energy sources
RR Rapid response
SA Synergic adjustment
SAC Soft actor–critic
SC Supercapacitor
SMDP Semi-Markov Decision Process
SOC State of charge
SRSI System response synergy index
WT Wind turbines
Symbols
𝐶 deg CAES degradation cost coefficients
𝐶 deg
𝑡 CAES degradation cost ($)
𝐶𝑠𝑢
𝑡 CAES start-up cost ($)
𝐶𝑟𝑎𝑡𝑒 Rate limiting factor
𝑐𝑠𝑢 Start-up cost coefficients
𝐷𝐸𝑙𝑧 Electrolyzer benefit–cost ratio ($/kW)
𝐸𝐵 BESS capacity (kWh)
ℎ Enthalpy (kJ/kg)
𝑀𝐻2 Molar mass (kg/mol)
𝑝𝑇𝐾 Air storage tank pressure (bar)
𝑃𝐵𝐸𝑆𝑆
𝑡 BESS power (kW)
𝑃𝐶𝐴𝐸𝑆
𝑡 CAES power (kW)
𝑃𝑑𝑖𝑠𝑝
𝑡 Dispatchable power (kW)
𝑃𝑒𝑙𝑒
𝑡 Electricity demand (kW)
𝑃𝐸𝑙𝑧
𝑡 Electrolyzer power (kW)
𝑃𝑔
𝑡 Interchange power (kW)
𝑃𝑃𝑉
𝑡 PV power (kW)
𝑃𝑊𝑇
𝑡 WT power (kW)
𝑆𝑂𝐶𝐵 BESS state of charge
𝑆𝑂𝐶𝐶 CAES state of charge
𝑇𝑇𝐾 Air storage tank temperature (◦C)
𝑇𝑃𝑉 Operating temperature (◦C)
𝑇𝑆𝑃𝑉 Module temperature (◦C)
𝑇𝐶𝑡 Transaction costs ($)
𝑉𝑇𝐾 Air storage tank volume ( m3)
𝑣𝑟 Cut-in wind speed (m/s)
𝑣𝑖𝑛 Rated speed (m/s)
𝑣𝑜𝑓𝑓 Cut-out wind speed (m/s)
̄𝑃𝑊𝑇 WT rated power (kW)
̇ 𝑚𝑎𝑖𝑟
𝑡 Air mass flow rate (kg/s)
𝐿𝐻𝑉 𝐻2 𝐻2 low heating value (kJ/mol)
𝑆𝑃𝑉 Photovoltaic panels area ( m2)
Greek letters
𝛿𝑡 Cycle depth
𝜂𝐸𝑙𝑧 Electrolyzer efficiency
𝜂𝑐ℎ
𝐵 BESS charging efficiency
𝜂𝑑𝑖𝑠
𝐵 BESS discharging efficiency
𝜂𝑐𝑜𝑚 Compressor isentropic efficiency
𝜂𝑡𝑢𝑟 Turbine isentropic efficiency
𝛾 Discount factor
𝜆 Price
𝛺 Option space
𝜔 Option
𝜏 Time interval
𝜏𝑠 Update rate
𝜃 Value network parameter
𝜃𝑡ℎ𝑟 Threshold
𝜑 Actor network parameter
𝜗 Termination function parameter
efficiency through encapsulation of subtasks within ‘‘options’’ that
target the selection of distinct energy storage modes with initial and
termination conditions. Simulation validations across multiple baseline
scenarios confirm the stability, economy and flexibility of proposed
MES mechanism and OCTD3 algorithm.
The remainder of this paper is organized as follows: Section 2
formulates the problem. Section 3 details the proposed methodology.
Section 4 presents the dispatch results. Finally, Section 5 concludes the
study.
2. Problem formulation
The IEMS structure depicted in Fig. 1 integrates wind turbines
and photovoltaic panels (PV) for power generation, proton exchange
membrane (PEM) electrolyzers for hydrogen production, a hydrogen
refueling station and a residential neighborhood with dynamic de-
mand, a hybrid CAES–BESS system for energy storage and a energy
hub for dispatch. The hybrid CAES–BESS system integrates a CAES
cogeneration system. It is based on the previous work [30], includes
compression heat harvesting and solar energy utilization for the inlet
air preheating of high-pressure and low-pressure turbines (HPT/LPT),
enhancing overall efficiency. Leveraging the rapid response of BESS
and the large capacity of CAES, the MES mechanism features three
modes. In RR mode, BESS regulates energy swiftly; in LEB mode,
CAES ensures long-term energy equilibrium; and in SA mode, both
CAES and BESS work to dispatch energy. Additionally, system can
purchase additional electricity or thermal energy from the main grid or
combined heat pump system (CHS) respectively, or sell excess energy
back to them. The dispatch hub interacts with the environment and
exchanges information, guiding agent to make real-time decisions. The
DRL algorithm agent, acting as the system’s brain, selects the operating
Applied Energy 374 (2024) 123950 
3 

--- Page 4 ---

F. Cui et al.
Table 1
Summary of the literature mentioned in the introduction.
Literature Year System RES Multiform demand HESS Multi-mode operation Method
[7] 2023 MG a ✓ × ✓(HES/BESS) × Traditional
[8] 2021 MG ✓ × ✓(BESS/SC) × Traditional
[9] 2024 MG ✓ ✓ ✓ (HES/BESS) × Traditional
[10] 2023 MG ✓ × ✓(HES/BESS) × Traditional
[11] 2021 IEMS ✓ ✓ ✓ (CAES/BESS) × Traditional
[15] 2014 MG ✓ × × × Traditional
[16] 2023 IEMS ✓ × × × Traditional
[23] 2023 IEMS ✓ × × × DQN
[24] 2020 IEMS ✓ ✓ × × DDPG
[25] 2021 IEMS ✓ ✓ × × SAC
[22] 2024 MG ✓ × × × PPO
[27] 2024 IEMS ✓ ✓ × × ILb-based PPO
[28] 2023 MG ✓ × × × Constrained SAC
[26] 2023 IEMS ✓ ✓ × × TD3
[29] 2023 IEMS ✓ × × × HRL
Proposed 2024 IEMS ✓ ✓ ✓ (CAES/BESS) ✓ OCTD3
a Micro Grid.
b Imitation Learning.
Fig. 1. The proposed IEMS structure with a multi-mode hybrid energy storage system.
modes and adjusts the output power of hybrid CAES–BESS system based
on the MES mechanism to optimize stability, costs, and flexibility.
In this section, the mathematical models and operating constraints
for the critical components of the IEMS are established. The optimiza-
tion objectives for the dispatch task are formulated.
2.1. Models
2.1.1. Grid
The main grid acts as a backup power source to maintain system
stability. At time 𝑡, the involved power balance equation is defined
as [31]:
𝑃𝑔
𝑡 =𝑃𝐶𝐴𝐸𝑆
𝑡 +𝑃𝐵𝐸𝑆𝑆
𝑡 +𝑃𝐸𝑙𝑧
𝑡 +𝑃𝑒𝑙𝑒
𝑡 −𝑃𝑃𝑉
𝑡 −𝑃𝑊𝑇
𝑡 (1)
where 𝑃𝑔
𝑡 represents the interchange power between IEMS and main
grid. 𝑃𝐶𝐴𝐸𝑆
𝑡 is the power consumed or produced by CAES. 𝑃𝐵𝐸𝑆𝑆
𝑡 de-
notes the BESS charging or discharging power. 𝑃𝐸𝑙𝑧
𝑡 denotes the power
consumed by the PEM electrolyzers.𝑃𝑒𝑙𝑒
𝑡 is the electricity demand.𝑃𝑃𝑉
𝑡
and 𝑃𝑊𝑇
𝑡 represent the power generated by PV and WT, respectively.
2.1.2. Photovoltaic system
The photovoltaic system converts solar radiation into electricity
energy, and its output power is calculated as [ 32]:
𝑃𝑃𝑉
𝑡 =𝜂𝑃𝑉 𝑆𝑃𝑉 𝐸𝑡(1 +𝑟(𝑇𝑃𝑉 −𝑇𝑆𝑃𝑉 )) (2)
where 𝜂𝑃𝑉 and 𝑆𝑃𝑉 are the conversion efficiency and the area of
photovoltaic panels. 𝐸𝑡 denotes the solar irradiance. 𝑟 denotes the
Applied Energy 374 (2024) 123950 
4 

--- Page 5 ---

F. Cui et al.
solar temperature coefficient. 𝑇𝑃𝑉 and 𝑇𝑆𝑃𝑉 are the operating tem-
perature and solar module temperature at the standard test condition,
respectively.
2.1.3. Wind turbine
The WT output power is determined by the kinetic energy of the
wind sweeping through its blades, which can be modeled as [11]:
𝑃𝑊𝑇
𝑡 =
⎧
⎪
⎪
⎨
⎪
⎪⎩
̄𝑃𝑊𝑇 ⋅
𝑣2
𝑡 −𝑣2
𝑖𝑛
𝑣2𝑟 −𝑣2
𝑖𝑛
,𝑣𝑖𝑛 <𝑣 𝑡 <𝑣 𝑟
̄𝑃𝑊𝑇 ,𝑣𝑟 <𝑣 𝑡 <𝑣 𝑜𝑓𝑓
0 ,𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
(3)
where ̄𝑃𝑊𝑇 is the WT rated power.𝑣𝑖𝑛,𝑣𝑟 and𝑣𝑜𝑓𝑓 are the cut-in, rated
and cut-out wind speed, respectively.
2.1.4. PEM electrolyzer
The PEM electrolyzer decomposes water into oxygen and protons at
the anode, transports the protons through a proton exchange membrane
to the cathode, and produces hydrogen by reacting them with electrons.
The hydrogen production rate can be calculated as [33]:
̇ 𝑚𝐻2
𝑡 =
𝜂𝐸𝑙𝑧𝑃𝐸𝑙𝑧
𝑡 𝑀𝐻2
𝐿𝐻𝑉 𝐻2
(4)
where 𝜂𝐸𝑙𝑧 is the electrolyzer efficiency. 𝐿𝐻𝑉 𝐻2 represents the hy-
drogen low heating value. 𝑀𝐻2 is the molar mass of hydrogen. The
electrolyzer is restricted by:
𝑃𝐸𝑙𝑧,min ≤𝑃𝐸𝑙𝑧
𝑡 ≤𝑃𝐸𝑙𝑧,max (5)
where 𝑃𝐸𝐿,max and 𝑃𝐸𝐿,min are the peak to base power of the elec-
trolyzer.
A linear degradation model of electrolyzer from [34] is employed,
which considers utilization frequency, current density, temperature
etc.:
𝑂𝐶𝐸𝑙𝑧
𝑡 =𝐷𝐸𝑙𝑧𝑃𝐸𝑙𝑧
𝑡 (6)
where 𝐷𝐸𝑙𝑧 is the electrolyzer benefit–cost ratio.
2.1.5. BESS
The BESS operation adheres to state continuity constraint, as fol-
lows:
𝑆𝑂𝐶𝐵
𝑡+1 =
⎧
⎪
⎨
⎪⎩
𝑆𝑂𝐶𝐵
𝑡 + 1
𝜂𝑑𝑖𝑠
𝐵
⋅
𝑃𝐵𝐸𝑆𝑆
𝑡 𝜏
𝐸𝐵
,𝑃 𝐵𝐸𝑆𝑆
𝑡 ≤ 0
𝑆𝑂𝐶𝐵
𝑡 +𝜂𝑐ℎ
𝐵 ⋅
𝑃𝐵𝐸𝑆𝑆
𝑡 𝜏
𝐸𝐵
,𝑃 𝐵𝐸𝑆𝑆
𝑡 > 0
(7)
where𝑆𝑂𝐶𝐵,min and𝑆𝑂𝐶𝐵,max are the minimum and maximum state of
charge (SOC). 𝜂𝑑𝑖𝑠
𝐵 and𝜂𝑐ℎ
𝐵 are the discharging and charging efficiency.
𝜏 is the dispatch time step. 𝐸𝐵 is the BESS capacity.
To optimize the BESS safety, efficiency, and longevity [10], the
power is limited by (8) and (9). The storage capacity is constrained
by (10):
−𝑃𝐵,max ≤𝑃𝐵𝐸𝑆𝑆
𝑡 ≤𝑃𝐵,max (8)
|||𝑃𝐵𝐸𝑆𝑆
𝑡
||| ≤𝐶𝑟𝑎𝑡𝑒⋅𝐸𝐵 (9)
𝑆𝑂𝐶𝐵,min ≤𝑆𝑂𝐶𝐵
𝑡 ≤𝑆𝑂𝐶𝐵,max (10)
where 𝑃𝐵,max represents the maximum power of BESS. 𝑆𝑂𝐶𝐵
𝑡 denotes
the SOC at time 𝑡. 𝐶𝑟𝑎𝑡𝑒 is the charge or discharge rate limiting factor.
During the charge–discharge cycles, there exists chemical reactions
and physical wear in BESS. This study employs a non-linear degradation
model from [35]. First, the cycle depth and life cycle degradation are
defined as:
𝛿𝑡 =
𝑃𝐵𝐸𝑆𝑆
𝑡
𝜂𝑑𝑖𝑠
𝐵 𝐸𝐵
+𝛿𝑡−1 (11)
𝜓(𝛿𝑡) = 𝑎0(𝛿𝑡)2.03 (12)
where𝛿𝑡−1 is the cycle depth at the last moment.𝑎0 is the multi-nominal
coefficient. The marginal degradation cost is:
𝑐𝑏
𝑡 =𝐷𝑏𝑒𝑠𝑠 𝜕𝜓 (𝛿𝑡)
𝜕𝑃𝐵𝐸𝑆𝑆
𝑡
= 2.03⋅𝑎0⋅ 𝐷𝑏𝑒𝑠𝑠
𝜂𝑑𝑖𝑠
𝐵 𝐸𝐵
⋅𝛿1.03
𝑡 (13)
where 𝐷𝑏𝑒𝑠𝑠 is the benefit–cost ratio. Finally, the degradation cost of
BESS at time t can be calculated as:
𝑂𝐶𝐵
𝑡 =𝑃𝐵𝐸𝑆𝑆
𝑡 𝑐𝑏
𝑡 (14)
2.1.6. CAES
This study presents the CAES thermodynamic models for key com-
ponents: the three-stage compressor units and HPT/LPT units. Detailed
models and assumptions for the CAES are based on [30]. The three-
stage compressor unit receives power consumption commands from the
system and responds by adjusting the air mass flow rate. In Fig. 1, the
interstage cooling temperatures of CAES at state points 2, 4, and 7,
denoted as 𝑇𝑐𝑜𝑜𝑙, are fixed. The consumption power by a compressor
can be calculated as:
𝑃𝑐𝑜𝑚
𝑡 =
̇ 𝑚𝑎𝑖𝑟,𝑐ℎ
𝑡 (ℎ𝑠
𝑐𝑜𝑚_𝑜𝑢𝑡 −ℎ𝑐𝑜𝑚_𝑖𝑛)
𝜂𝑐𝑜𝑚
(15)
where ̇ 𝑚𝑎𝑖𝑟,𝑐ℎ
𝑡 is the stored air mass flow rate. 𝑠 represents the ideal
process. ℎ𝑐𝑜𝑚_𝑜𝑢𝑡 and ℎ𝑐𝑜𝑚_𝑖𝑛 represents the actual enthalpy of air at the
compressor’s outlet and inlet, respectively. 𝜂𝑐𝑜𝑚 is the isentropic effi-
ciency. To further characterize the compression process, the pressure
ratio across the compressor stages can be described as:
𝜆= 𝑛
√𝑝𝑒𝑛𝑑
𝑝𝑖𝑛
(16)
where𝜆 and𝑛 represent the pressure ratio and number of compressors,
respectively. 𝑝𝑒𝑛𝑑 and 𝑝𝑖𝑛 denote the outlet pressure and inlet pressure
of compressor units, respectively.
The power generated by the turbines can be calculated as:
𝑃𝑡𝑢𝑟
𝑡 = ̇ 𝑚𝑎𝑖𝑟,𝑑𝑖𝑠
𝑡 (ℎ𝑡𝑢𝑟_𝑖𝑛 −ℎ𝑠
𝑡𝑢𝑟_𝑜𝑢𝑡)𝜂𝑡𝑢𝑟 (17)
where ̇ 𝑚𝑎𝑖𝑟,𝑑𝑖𝑠
𝑡 is the released air mass flow rate. ℎ𝑡𝑢𝑟_𝑖𝑛 and ℎ𝑡𝑢𝑟_𝑜𝑢𝑡 are
the actual enthalpy of air at the turbine’s inlet and outlet, respectively.
𝜂𝑡𝑢𝑟 is the isentropic efficiency of the turbines.
Due to the complex thermodynamic changes and nonlinear factors
in CAES such as 𝜂𝑐𝑜𝑚 and 𝜂𝑡𝑢𝑟, quadratic fitting is performed to derive
the relationship between the control variable (i.e., air mass flow rate)
and the compressor power as well as the turbine power, as follows:
̇ 𝑚𝑎𝑖𝑟,𝑐ℎ
𝑡 =𝛼1,𝑐𝑜𝑚𝑃𝑐𝑜𝑚
𝑡
2 +𝛼2,𝑐𝑜𝑚𝑃𝑐𝑜𝑚
𝑡 +𝛼3,𝑐𝑜𝑚 (18)
̇ 𝑚𝑎𝑖𝑟,𝑑𝑖𝑠
𝑡 =𝛼1,𝑡𝑢𝑟𝑃𝑡𝑢𝑟
𝑡
2 +𝛼2,𝑡𝑢𝑟𝑃𝑡𝑢𝑟
𝑡 +𝛼3,𝑡𝑢𝑟 (19)
where 𝛼1,𝑐𝑜𝑚, 𝛼2,𝑐𝑜𝑚, 𝛼3,𝑐𝑜𝑚, 𝛼1,𝑡𝑢𝑟, 𝛼2,𝑡𝑢𝑟, and 𝛼3,𝑡𝑢𝑟 are fitting coefficients
obtained in Section 4.1.1.
During the operation, the state continuity constraint of CAES is
shown in (20):
𝑆𝑂𝐶𝐶
𝑡+1 =𝑆𝑂𝐶𝐶
𝑡 +
𝑝𝑡𝑘
𝑡
𝑝𝑇𝐾, max −𝑝𝑇𝐾, min (20)
where 𝑆𝑂𝐶𝐶
𝑡 represents the SOC of the CAES system. 𝑝𝑇𝐾, min and
𝑝𝑇𝐾, max are the highest and lowest pressure of the air storage tank
(AST). 𝑝𝑡𝑘
𝑡 is the storage pressure at time 𝑡, calculated as:
𝑝𝑡𝑘
𝑡 =
( ̇m𝑎𝑖𝑟,𝑐ℎ
𝑡 𝑢𝑐
𝑡 − ̇m𝑎𝑖𝑟,𝑑𝑖𝑠
𝑡 𝑢𝑡
𝑡)𝑅𝑇𝑇𝑘
𝑉𝑇𝑘
(21)
where the binary variables𝑢𝑐
𝑡 and𝑢𝑡
𝑡 indicate the on or off state of com-
pressors and turbines, which are limited to 𝑢𝑐
𝑡 ⋅𝑢𝑡
𝑡 = 0. 𝑅 is the specific
gas constant for air. 𝑇𝑇𝐾 and 𝑉𝑇𝐾 represent the AST temperature and
volume, respectively.
Applied Energy 374 (2024) 123950 
5 

--- Page 6 ---

F. Cui et al.
To ensure the system operating within efficient and safe boundaries,
the pressure of AST is constrained by (22). The SOC of CAES is
constrained by (23). The compressors and turbines power constraints
are shown in (24) and (25), respectively.
𝑝𝑇𝑘, min ≤𝑝𝑡𝑘
𝑡 ≤𝑝𝑇𝑘, max (22)
𝑆𝑂𝐶𝐶,min ≤𝑆𝑂𝐶𝐶
𝑡 ≤𝑆𝑂𝐶𝐶,max (23)
𝑃𝑐𝑜𝑚,min ≤𝑃𝑐𝑜𝑚
𝑡 ≤𝑃𝑐𝑜𝑚,max (24)
𝑃𝑡𝑢𝑟,min ≤𝑃𝑡𝑢𝑟
𝑡 ≤𝑃𝑡𝑢𝑟,max (25)
where𝑆𝑂𝐶𝐶,min and𝑆𝑂𝐶𝐶,max represent the lowest and highest allow-
able SOC. 𝑃𝑐𝑜𝑚,min and 𝑃𝑐𝑜𝑚,max indicate the base and peak power for
the compressor units. 𝑃𝑡𝑢𝑟,min and 𝑃𝑡𝑢𝑟,max indicate the base and peak
power of turbines.
The operational costs of the CAES consider thermal losses and
mechanical wear from starting and stopping, along with degradation
costs from frequent compression and expansion, calculated as [36]:
𝑂𝐶𝐶
𝑡 =𝐶𝑠𝑢
𝑡 +𝐶 deg
𝑡 (26)
where 𝐶𝑠𝑢
𝑡 and 𝐶 deg
𝑡 are start-up and degradation costs, which are
respectively defined as [37]:
𝐶𝑠𝑢
𝑡 =𝑐𝑠𝑢
|||𝑢𝑐
𝑡 −𝑢𝑐
𝑡−1
||| +𝑐𝑠𝑢
|||𝑢𝑒
𝑡 −𝑢𝑒
𝑡−1
||| (27)
𝐶 deg
𝑡 =𝑐deg(𝑃𝑐𝑜𝑚
𝑡 +𝑃𝑡𝑢𝑟
𝑡 ) (28)
where 𝑐𝑠𝑢 and 𝑐deg are the empirical coefficients of start-up cost and
degradation cost respectively.
2.2. Optimization objectives
In this section, three optimization objectives are introduced, en-
hancing the stability, cost-effectiveness, and response flexibility of the
proposed IEMS. The mathematical expressions are described in the
following.
Obj. 1 : To maintain the stability of the main grid, minimize the
interchange power fluctuations, defined as:
min𝑃𝐹𝐼 =
∑
𝑡
||𝑃𝑔
𝑡 −𝑃𝑔,𝑎𝑣|| (29)
where 𝑃𝑔,𝑎𝑣 is the average interchange power.
Obj. 2 : The comprehensive costs include two parts: system trans-
action costs and the operational costs. The corresponding optimization
function is:
min𝐶𝐶𝐼 =
∑
𝑡
(𝑇𝐶𝑡 +𝑂𝐶𝐶
𝑡 +𝑂𝐶𝐵
𝑡 +𝑂𝐶𝐸𝑙𝑧
𝑡 ) (30)
where system transaction costs 𝑇𝐶𝑡 can be calculated as:
𝑇𝐶𝑡 =𝜆𝑒
𝑡𝑃𝑔
𝑡 +𝜆𝑓𝑃𝐹𝐼 − (𝜆𝑒
𝑡 +𝜆𝑠)𝑃𝑒𝑙𝑒
𝑡 −𝜆ℎ ̇ 𝑚𝐻2
𝑡 −𝜆𝑤 ̇ 𝑚𝑤
𝑡 (31)
where𝜆𝑒
𝑡 ,𝜆𝑠 and𝜆𝑓 are the purchasing electricity price, the service fee
charged to users and deviation fluctuation fee, respectively. 𝜆ℎ and𝜆𝑤
are the prices of hydrogen and hot water.
Obj. 3: To enhance the system’s responsiveness to rapid changes in
energy supply and demand, the SRSI is proposed to measure the system
flexibility. Its optimization function can be expressed as:
𝑆𝑅𝑆𝐼 =
∑
𝑡
⎧
⎪
⎨
⎪⎩
−
𝑃𝐵𝐸𝑆𝑆
𝑡
𝑃𝑑𝑖𝑠𝑝
𝑡
, if |||𝑃𝑑𝑖𝑠𝑝
𝑡
|||<𝜃 𝑡ℎ𝑟
−
𝑃𝐶𝐴𝐸𝑆
𝑡
𝑃𝑑𝑖𝑠𝑝
𝑡
, otherwise
(32)
where the dispatchable power 𝑃𝑑𝑖𝑠𝑝
𝑡 is the difference between the
unoptimized and the average predicted interchange power, indicating
the required power adjustment.𝜃𝑡ℎ𝑟 is the threshold that determines the
significance of the fluctuation deviation, which can be obtained based
on collected historical data and real-time information published by the
higher-level operators.
3. Dispatch solution via OCTD3 algorithm
In this section, the dispatch task is modeled as an option-based
SMDP [38]. To effectively address the uncertainties and the complexity
of the decision space, the OCTD3 algorithm for hierarchical decision
making is proposed.
3.1. Option-based SMDP
SMDP is developed for modeling continuous-time and discrete-event
systems, where the duration of each state is random and variable.
The option-based SMDP is developed by incorporating options into
Markov Decision Process (MDP) and extends actions that have random
durations [39], possessing characteristics of both SMDP and MDP.
In this study, the selection of energy storage modes, characterized
by indeterminate durations, displays SMDP traits, while the internal
policies adhere to state-based MDP characteristics. Consequently, the
dispatch task can be modeled as an option-based SMDP, including the
elements of state space 𝑆, action space 𝐴, option space 𝑂, reward
function𝑅, and transition function𝑃 , denoted as a tuple⟨𝑆,𝐴,𝑂,𝑅,𝑃 ⟩.
In this framework, the energy dispatch hub acts as an agent, and the
five elements are designed as follows:
State space 𝑆: The state space should incorporate crucial infor-
mation to comprehensively describe the dynamic environment. In this
study, the state at time 𝑡 is denoted as: 𝑠𝑡 =
[𝜆𝑒
𝑡,𝑃 𝑃𝑉
𝑡 ,𝑃 𝑊𝑇
𝑡 ,𝑃 𝑒𝑙𝑒
𝑡 ,𝑃 𝐸𝑙𝑧
𝑡 ,𝑆𝑂𝐶 𝐶
𝑡 ,𝑆𝑂𝐶 𝐵
𝑡 ,𝑢𝑐
𝑡,𝑢𝑡
𝑡].
Action space 𝐴: The action space contains all actions an agent
can take in a state. In this study, the actions at time 𝑡 are defined as
𝑎𝑡 = [𝑃𝐶𝐴𝐸𝑆
𝑡 ,𝑃 𝐵𝐸𝑆𝑆
𝑡 ]. The actions are constrained by Eqs. (7) to (10)
and Eqs. (20) to (25).
Option space 𝛺: The ‘‘option’’ concept introduces temporally ex-
tended actions to tackle complex decision-making challenges. Each
option 𝜔 ∈ 𝛺 consists of three core elements ⟨𝐼,𝜋,𝛽 ⟩: 𝐼 ⊆ 𝑆 is the
initiation set defining the states where an option can begin; 𝜋 is the
policy that directs action selection; and𝛽 ∶𝑆 → [0, 1] is the termination
condition, which specifies the probability of the option ending in state
𝑠. To support the MES mechanism, the option set is defined as 𝛺 =
{𝜔1,𝜔 2,𝜔 3}, corresponding to the three energy storage modes of RR,
LEB and SA.
Reward 𝑅: Reward is the feedback an agent receives after taking
a certain action. Immediate reward 𝑅𝑡 reflects the short-term impact
of an action on achieving long-term goals. According to the long-term
objectives in 2.2, 𝑅𝑡 is designed as:
𝑅𝑡 =𝑅𝑆𝐸
𝑡 +𝑅𝐹
𝑡 (33)
where 𝑅𝑆𝐸
𝑡 is the fluctuations and cost penalty function to optimize
system stability and economy, and 𝑅𝐹
𝑡 is the strategy flexibility reward
function to promote the use of BESS for rapid response and CAES for
long-term energy balance, which are respectively designed as:
𝑅𝑆𝐸
𝑡 = (1 −𝑤) ̃𝑃𝐹𝐼 𝑡 +𝑤 ̃𝐶𝐶𝐼𝑡 (34)
𝑅𝐹
𝑡 = 0.5⋅
[
𝟏(|𝑝𝑑𝑖𝑠𝑝|<𝜃ℎ𝑟 and 𝜔=𝜔1)
+ 𝟏(|𝑝𝑑𝑖𝑠𝑝|≥𝜃ℎ𝑟 and 𝜔∈{𝜔2,𝜔3})
] (35)
where ̃𝑃𝐹𝐼 𝑡 and ̃𝐶𝐶𝐼𝑡 are the normalized values of PFI and CCI at time
𝑡. 𝑤 represents the cost weight factor. 1𝑐𝑜𝑛𝑑𝑖𝑡𝑖𝑜𝑛 is an indicator function
that equals 1 when the ‘‘condition’’ is true, and 0 otherwise.
Transition function 𝑃 : The transition functions describe the prob-
abilities of moving to a new state𝑠′ from a current state𝑠. Specifically,
𝑃 (𝑠′|𝑠,𝜔 ) defines the probability of transitioning to 𝑠′ under option 𝜔,
while 𝑃 (𝑠′|𝑠,𝑎 ) outlines the probability given the action 𝑎.
Applied Energy 374 (2024) 123950 
6 

--- Page 7 ---

F. Cui et al.
Fig. 2. OCTD3: An option-based HRL algorithm that integrates OC framework and TD3.
3.2. OCTD3 algorithm
Fig. 2 illustrates the architecture of OCTD3 algorithm, depicting
how the agent dynamically interacts with the energy dispatch environ-
ment. The agent interacts with the dispatch environment, collecting
states 𝑠𝑡, executing actions 𝑎𝑡, and receiving feedback on rewards 𝑟𝑡
and state 𝑠𝑡+1. Through repeated trials and errors, the agent refines
its actions to maximize long-term rewards, ultimately developing an
optimal energy dispatch strategy for the energy hub. The learning
mechanism of OCTD3 deeply integrates the selection of top-level stor-
age modes, precise termination conditions for these modes, and the
policies governing each low-level option. The OCTD3 algorithm inno-
vatively combines a top-level OC framework with the TD3 algorithm,
specifically designed to manage and optimize the three energy storage
modes for various supply–demand scenarios. Hierarchical decision-
making within the OCTD3 is structured around the ‘‘call and return’’
model [ 40], where ‘‘call’’ involves the top-level policy selecting an
option and initiating its internal policy, while ‘‘return’’ occurs when
the internal policy accomplishes its designated task, handing back
control to the top-level policy. This cyclical interaction facilitates robust
and adaptable decision-making processes across different layers of the
system’s architecture.
This section further introduces the principles and training processes
of OCTD3 algorithm.
3.2.1. Principle description
The learning process within the top-level framework focuses on
selecting optimal energy storage modes and involves three components:
policy over options 𝜋𝛺(𝜔|𝑠), intra-option policies 𝜋𝜔(𝑎|𝑠), and termina-
tion functions𝛽𝜔(𝑠). In each option, the action policies and termination
functions are represented using parameterized neural networks, de-
noted as𝜋𝜔,𝜃(𝑎|𝑠) and𝛽𝜔,𝜗(𝑠). Three core value functions are as follows.
The expected cumulative reward from choosing an option 𝜔 in state 𝑠
is defined as:
𝑄𝛺(𝑠,𝜔 ) =
∑
𝑎
𝜋𝜔,𝜃(𝑎|𝑠)𝑄𝑈 (𝑠,𝜔,𝑎 ) (36)
where𝑄𝑈 (𝑠,𝜔,𝑎 ) is the expected cumulative reward of taking action 𝑎
under (𝑠,𝜔 ), defined as:
𝑄𝑈 (𝑠,𝜔,𝑎 ) = 𝑟(𝑠,𝑎 ) +𝛾
∑
𝑠′
𝑃 (𝑠′|𝑠,𝑎 )𝑈 (𝜔,𝑠 ′) (37)
where 𝑟(𝑠,𝑎 ) is the immediate rewards. The second term is the dis-
counted future returns. 𝛾 is the discount factor. 𝑈 (𝜔,𝑠 ′) represents the
value function of option 𝜔 reaching state 𝑠′ from state 𝑠, and 𝑃 (𝑠′|𝑠,𝑎 )
is the corresponding probability. 𝑈 (𝜔,𝑠 ′) is defined as:
𝑈 (𝜔,𝑠 ′) = (1 − 𝛽𝜔,𝜗(𝑠′))𝑄𝛺(𝑠′,𝜔 ) +𝛽𝜔,𝜗(𝑠′)𝑉𝛺(𝑠′) (38)
where 𝑉𝛺 represents the value function for state 𝑠, defined as:
𝑉𝛺(𝑠′) =
∑
𝜔
𝜋𝛺(𝜔′|𝑠′)𝑄𝛺(𝑠′,𝜔 ′) (39)
Applied Energy 374 (2024) 123950 
7 

--- Page 8 ---

F. Cui et al.
For each option, there exists updates for both intra-option policies
𝜋𝜔 and termination functions 𝛽𝜔. Sutton et al. [41] proposed that
the parameterized stochastic policy family can be optimized using
stochastic gradient descent to find the optimal policy. Based on this,
two gradient theorems were derived in the OC algorithm to update the
parameters 𝜃 and 𝜗.
Theorem 1. Intra-Option Policy Gradient.
𝜕𝑄𝛺(𝑠0,𝜔 0)
𝜕𝜃 =
∑
𝑠,𝜔
𝜇𝛺(𝑠,𝜔|𝑠0,𝜔 0)
×
∑
𝑎
𝜕𝜋𝜔,𝜃(𝑎|𝑠)
𝜕𝜃 𝑄𝑈 (𝑠,𝜔,𝑎 )
(40)
where 𝜇𝛺(𝑠,𝜔|𝑠0,𝜔 0) denotes the discount probability of the state–option
pair along the trajectory starting from the initial conditions (𝑠0,𝜔 0).
Theorem 2. Termination Gradient.
𝜕𝑈 (𝑠1,𝜔 0)
𝜕𝜗 = −
∑
𝑠′,𝜔
𝜇(𝑠′,𝜔|𝑠1,𝜔 0)
𝜕𝛽𝜔,𝜗(𝑠′)
𝜕𝜗 𝐴𝛺(𝑠′,𝜔 ) (41)
where 𝜇(𝑠′,𝜔|𝑠0,𝜔 0) denotes the discount probability of transitioning from
(𝜔0,𝑠 1) to (𝜔,𝑠 ′). 𝐴𝛺(𝑠′,𝜔 ) is the advantage function, defined as:
𝐴𝛺(𝑠′,𝜔 ) = 𝑄𝛺(𝑠′,𝜔 ) −𝑉𝛺(𝑠′) (42)
Based on this, the Bellman equation is derived to describe how the
value of a state evolves over time:
𝑄∗
𝑈 (𝑠,𝜔,𝑎 ) = 𝑟(𝑠,𝑎 )+𝛾
∑
𝑠′
𝑃 (𝑠′|𝑠,𝑎 )
[ (1 −𝛽𝜔,𝜗(𝑠′))𝑄∗
𝛺(𝑠′,𝜔 )
+ 𝛽𝜔,𝜗(𝑠′)max
̄ 𝜔
𝑄∗
𝛺(𝑠′, ̄ 𝜔) ] (43)
In 𝜋𝛺, a softmax function is included to generate the probability
of selecting each option, which uses a ‘‘temperature’’ parameter 𝜅 to
adjust the ‘‘flatness’’ or ‘‘sharpness’’ of the probability distribution,
balancing exploration and exploitation. The update of 𝜋𝛺 relies on
the value network, which is updated by the Temporal-Difference (TD)
method:
𝛿 =𝑔 −𝑄𝛺(𝑠,𝜔 ) (44)
where the target 𝑔 integrates the impacts of option termination and
continuation when estimating future returns, defined as:
𝑔 =𝑟′ +𝛾 [ (1 −𝛽𝜔,𝜗(𝑠′)) ×𝑄𝛺(𝑠′, ̄ 𝜔)
+ 𝛽𝜔,𝜗(𝑠′) max
̄ 𝜔
𝑄𝛺(𝑠′, ̄ 𝜔) ] (45)
In this study, the TD3 algorithm is innovatively integrated as the
intra-option policy. At option𝜔, suppose𝜃𝜔,1,𝜃𝜔,2, and𝜑𝜔 respectively
denote the parameters of TD3’s two critic networks 𝑄𝜃𝜔,1 and 𝑄𝜃𝜔,2,
and the actor network 𝜋𝜑𝜔, with 𝜃′
𝜔,1, 𝜃′
𝜔,2, and 𝜑′
𝜔 representing the
parameters of the corresponding target networks. Due to TD3’s utiliza-
tion of a twin critic-network structure and target policy smoothing, the
intra-option policy gradient Eq. (40) can be rewritten as:
∇𝜑𝜔𝐽 (𝜑𝜔) = 1
𝑁
∑
∇𝜑𝜔𝑄𝜃𝜔,1 (𝑠,𝑎 )|𝑎=𝜋𝜑𝜔 (𝑠) (46)
where𝑁 is the batch size. The estimated value𝑄𝜃𝜔,1 (𝑠,𝑎 ) from network
𝑄𝜃𝜔,1 represents the expected return for taking action 𝑎 in (𝑠,𝜔 ).
In this paper, the twin Critic structure of TD3 is adopted to estimate
the value of actions, which can reduce the common overestimation
issue in algorithms, namely the tendency to overly estimate future
rewards. Therefore, the update of the value network in Eqs. (44) and
(45) is changed to minimize the loss function, calculated as:
𝐿(𝜃𝜔,i) = 1
𝑁
∑
(𝑦 −𝑄𝜃𝜔,i (𝑠,𝑎 ))2,𝑖 = 1, 2 (47)
Algorithm 1 Training process of the proposed OCTD3 method
1: Initialize policy over options 𝜋𝛺, termination function 𝛽𝜔 and
intra-option TD3 networks 𝑄𝜃𝜔,1, 𝑄𝜃𝜔,2, and 𝜋𝜑𝜔.
2: Initialize target networks 𝑄′
𝜃𝜔,1
, 𝑄′
𝜃𝜔,2
, and 𝜋′
𝜑𝜔
.
3: Set delay update interval 𝐷, replay buffer , and epsilon 𝜖 =𝜖𝑚𝑎𝑥
4: for 𝑇𝑖𝑚𝑒𝑠𝑡𝑒𝑝𝑠 = 1 to 𝑇𝑆𝑚𝑎𝑥 do
5: 𝑠 ←𝑠0
6: Choose option 𝜔0 ∼𝜋𝛺(𝑠𝑡).
7: for 𝑡 = 1 to 𝑇 do
8: Select 𝑎 ∼𝜋𝜑𝜔 (𝑠) +𝜀 with 𝜖-greedy algorithm.
9: Execute action 𝑎 and obtain 𝑠′, 𝑟
10: Store transition (𝑠,𝑎,𝑟,𝑠 ′,𝜔 ) in replay buffer 
11: Sample mini-batch of 𝑁 transitions from 
12: Update parameters of each option:
13: 𝜃𝜔,𝑖 ← min
𝜃𝜔,𝑖
𝐿(𝜃𝜔,i),𝑖 = 1, 2
14: 𝜗 ←𝜗 −𝛼∇𝜗𝐽 (𝜗)
15: if 𝑡 mod 𝐷 == 0 then
16: Update actor network of each option:
17: 𝜑𝜔 ←𝜑𝜔 +𝛼∇𝜑𝜔𝐽 (𝜑𝜔)
18: end if
19: Softly update target networks with rate 𝜒:
20: 𝜃′
𝜔,𝑖 ←𝜏𝑠𝜃𝜔,𝑖 + (1 −𝜏𝑠)𝜃′
𝜔,𝑖, for 𝑖 = 1, 2
21: 𝜑′
𝜔 ←𝜏𝑠𝜑𝜔 + (1 −𝜏𝑠)𝜑′
𝜔
22: Decay epsilon 𝜖 by 𝛥𝜖, ensuring 𝜖 ≥𝜖min
23: if 𝛽𝜔,𝜗 terminates in 𝑠′ then
24: Choose 𝜔 ∼𝜋𝛺(𝑠′).
25: end if
26: 𝑠 ←𝑠′
27: end for
28: end for
where 𝑦 is the estimated value, which is calculated by minimum 𝑄
value:
𝑦 =𝑟 +𝛾 min
𝑖=1,2
𝑄′
𝜃𝜔,𝑖
(𝑠′,𝜋𝜑𝜔 (𝑠′)) (48)
Moreover, to avoid overestimation,𝑄𝑈 (𝑠,𝜔,𝑎 ) can be obtained from
the minimum value estimate output by the double critic networks of
TD3 given (𝜔,𝑠,𝑎 ), defined as:
𝑄𝑈 (𝑠,𝜔,𝑎 ) = min{𝑄𝜃𝜔,1 (𝑠,𝑎|𝜔),𝑄𝜃𝜔,2 (𝑠,𝑎|𝜔)} (49)
Thus concludes the description of the proposed OCTD3 algorithm.
This method combines enduring value assessment from the top-level
framework with robust action value estimation, characteristic of the
TD3 approach, enhancing the optimization of lower-level policies. This
integration yields a algorithm that can enhance both strategic decision-
making and dynamic policy adjustment.
3.2.2. Policy learning
Algorithm 1 delineates the training process for the OCTD3 method.
It starts by initializing the OCTD3 framework components, including
policy over options 𝜋𝛺, termination functions 𝛽𝜔, and 𝑄𝜃𝜔,1, 𝑄𝜃𝜔,2, 𝜋𝜑𝜔
in each TD3 of options. To promote stable learning, target networks
𝑄′
𝜃𝜔,1
, 𝑄′
𝜃𝜔,2
and 𝜋′
𝜑𝜔
are created, providing a consistent benchmark for
updates during the training phase. Training unfolds over a maximum
of 𝑇𝑆𝑚𝑎𝑥 steps with each episode lasting 𝑇 = 24 steps, segmented into
agent–environment interaction and parameter update phases according
to Fig. 2.
In the interaction phase, options 𝜔𝑡 are chosen based on 𝜋𝛺 using
a softmax function. Low-level actions 𝑎𝑡 are generated from the actor
network according to 𝑎 ∼𝜋𝜑𝜔 (𝑠) +𝜀 (where 𝜀 ∼  (0,𝜎 2)), adhering to
an𝜖-greedy approach. This strategy initially sets 𝜖 to a high value 𝜖𝑚𝑎𝑥
to encourage exploration and gradually decreases to a lower bound𝜖𝑚𝑖𝑛.
This reduction increases the likelihood of exploiting the best-known
Applied Energy 374 (2024) 123950 
8 

--- Page 9 ---

F. Cui et al.
Fig. 3. Variations of off-design parameters of the CAES system with the changes of partial load ratio: (a) Isentropic efficiency of compressors and turbines; (b) Expansion ratio
for HPT/LPT unit.
Fig. 4. Fitting results of the CAES system under off-design conditions: (a) Compressor power consumption versus stored air mass flow rate; (b) Turbine electricity generation
power versus released air mass flow rate.
actions as training progresses, effectively balancing exploration of new
actions with exploitation of known strategies. Actions yield new states
𝑠𝑡+1 and rewards 𝑟𝑡, stored as transitions (𝑠𝑡,𝑎𝑡,𝑟𝑡,𝑠𝑡+1,𝜔𝑡) in the replay
buffer . In the parameter update phase, a mini-batch of transitions is
sampled from . Transitions are sorted by𝜔 and used in the respective
intra-option TD3 networks. The critic networks𝑄𝜃𝜔,1 and𝑄𝜃𝜔,2 updates
by minimizing the loss𝐿(𝜃𝜔,i) through (47), which reflects the TD error.
The option termination function’s parameter 𝜗 updates by gradient
descent through (41), and the actor network 𝜋𝜑𝜔 updates via gradient
ascent by (46) at delayed intervals (𝐷) to maximize expected returns.
Target networks undergo soft updates at a rate of 𝜏𝑠, aligning closely
with the learning networks to mitigate risks from rapid value shifts.
Repeat the above steps until the termination condition is reached, and
then reselect an option.
4. Simulation results
In this section, energy hub dispatch simulations are conducted in a
comprehensive park. The CAES system under off-design conditions are
modeled. The OCTD3 algorithm is employed to dispatch energy. Six
comparative scenarios are set to verify the proposed MES mechanism
and OCTD3 algorithm. Finally, a sensitivity analysis is conducted to
validate the robustness under varying operational conditions.
4.1. Initialization
4.1.1. CAES modeling under off-design conditions
Based on previous work in [ 30], the compressors and turbines
power of the CAES operate at full load under design conditions, both
set 800 kW. Other designed parameters are detailed in Table 2 .
Under the off-design conditions, the compressors and turbines isen-
tropic efficiency varies with the partial load ratio, as shown in Fig. 3(a).
Similarly, the expansion ratios of the HPT/LPT unit change with the
partial load ratio, as depicted in Fig. 3 (b) [ 42]. Utilizing MATLAB
for mathematical modeling and REFPROP for accurate thermophysical
property calculations, quadratic models to the relationships between
stored air mass flow rate and compressor power, as well as released
air mass flow rate and turbine power are fitted. Fig. 4 displays both
simulated and fitted results for compressors and turbines under off-
design conditions. For the compressors, the fitting coefficients are
(𝛼1,com = 6 × 10 −6,𝛼 2,com = 0.0016,𝛼 3,com = 0.0058). For the turbines,
the fitting coefficients are (𝛼1,tur = 2 × 10 −5,𝛼 2,tur = 0 .0013,𝛼 3,tur =
0.1528). Compared to the curves ignoring the off-design conditions, the
fitted results modifies the bias of CAES model, enhancing the accuracy
of subsequent energy dispatch strategies. Additionally, during the air
expansion stage, water preheats the air at 368.15 K, exits the heat
exchanger at 353.15 K, and is sold to the heat network as by-product
(𝑚𝑤
𝑡 = 3.81 kg∕s , 𝑇 = 353.15 K ). Therefore, by fitting the relationships
for electricity–gas–heat conversion in this cogeneration CAES system,
the errors in the model under design conditions are corrected, and the
accuracy is improved.
4.1.2. Simulation settings
(1) Parameters and Data:The model parameters are listed in Table 2,
which are obtained from [ 10,11,30,32–34,36,43]. On a typical day,
the electricity purchasing prices 𝜆𝑒
𝑡 is based on time-of-use tariffs, as
detailed in Table 3 . The other energy market prices 𝜆𝑠, 𝜆ℎ and 𝜆𝑤
are set at 0.02 $/kWh, 1.5 $/kg and 0.0016 $/kg, respectively [ 43].
The irradiance data and wind speed data from [ 11], as well as the
electricity and hydrogen demand data from [ 44], are displayed in
Fig. 5. In the simulations, OCTD3 algorithm parameters from [ 45,46]
are listed in Table 4 . The experiments are carried out on a com-
puter with an i9-10900KF CPU and a NVIDIA GeForce RTX 3080
GPU. The programming framework employed Python 3.9 and PyTorch
1.13.1+cu116.
Applied Energy 374 (2024) 123950 
9 

--- Page 10 ---

F. Cui et al.
Fig. 5. Energy resources and demands data in the IEMS on a typical day: (a) Wind speed; (b) Solar irradiance; (c) Electricity demand; (d) Hydrogen demand.
Fig. 6. Training and evaluation reward curves of the OCTD3 method: (a) Training rewards; (b) Evaluation rewards.
Table 2
The model parameters of each component in the IEMS.
Components Parameters Values
PV and WT 𝜂𝑃𝑉 0.13
𝑆𝑃𝑉 3330 m2
𝑇𝑃𝑉 46 ◦C
𝑇𝑆𝑃𝑉 25 ◦C
𝑟 0.005
̄𝑃𝑊𝑇 500 kW
𝑣𝑖𝑛 2 m/s
𝑣𝑟 14 m/s
𝑣𝑜𝑓𝑓 25 m/s
PEM electrolyzer and BESS 𝜂𝐸𝑙𝑧 0.55
𝐿𝐻𝑉𝐻2 286 kJ∕mol
𝑃𝐸𝐿,min 1.2kW
𝑃𝐸𝐿,max 12kW
𝜆ℎ 0.313 $∕kWh
𝑃𝐵,max 250 kW
𝑆𝑂𝐶𝐵,min 0.1
𝑆𝑂𝐶𝐵,max 0.9
𝜂𝑑𝑖𝑠
𝐵 0.97
𝜂𝑐ℎ
𝐵 0.92
𝐸𝐵 1000 kWh
𝐶𝑟𝑎𝑡𝑒 0.4
𝑎0 0.00524
𝐷𝑏𝑒𝑠𝑠 400 $∕kWh
CAES 𝑝𝑇𝐾, min 40 bar
𝑝𝑇𝐾, max 60 bar
𝑇0 298.15 K
𝑝𝑖𝑛 101.325 kPa
𝑝𝑒𝑛𝑑 60 bar
𝑇𝑐𝑜𝑜𝑙 303.15 K
𝑃𝑐𝑜𝑚,max 800 kW
𝑃𝑡𝑢𝑟,max 800 kW
𝑆𝑂𝐶𝐶,min 0.2
𝑆𝑂𝐶𝐶,max 0.9
𝑉𝑇𝐾 1000 m3
𝑐𝑠𝑢 3.42 $
𝑐deg 0.1 $∕kWh
(2) Benchmarks and Metrics: The proposed method is compared
with six benchmark scenarios, divided into two groups, to validate the
effectiveness of the MES mechanism and OCTD3 algorithm respectively.
Table 3
Time-of-use electricity price.
Time period Electricity price ($/kWh)
0–8:00 0.0332
8:00–11:00 and 17:00–22:00 0.1364
15:00–17:00 and 22:00–24:00 0.0793
11:00–15:00 0.0520
Table 4
The parameter settings of the proposed OCTD3 method.
Parameters Values
Delay update interval 𝐷 2
Initial epsilon 𝜖𝑚𝑎𝑥 1.0
Minimum epsilon 𝜖𝑚𝑖𝑛 0.05
Epsilon decay 𝛥𝜖 6 × 10−6
Learning rate 𝛼 1 × 10−4
Soft update rate 𝜏𝑠 0.005
Batch size 𝑁 64
Discount factor 𝛾 0.99
Replay buffer size 10 000
Exploration noise variance 𝜎2 0.1
Temperature 𝜅 1
cost weight factors 𝑤 0.2
Scenarios 1 to 3 are set with same OCTD3 dispatch algorithms and
three different single-mode energy storage mechanisms, respectively
activating only BESS, activating only CAES, and simultaneously acti-
vating both CAES and BESS (referred to as HESS mode). The OCTD3
algorithm under three storage modes is implemented by fixing nodes 1,
2, and 3 in Fig. 2 respectively. Scenarios 4 to 6 are set with the same
MES mechanism and three different mainstream DRL dispatch meth-
ods, respectively TD3, DDPG, and PPO. In the subsequent discussions,
each scenario is denoted as a similar expression of ‘‘MES-OCTD3’’ to
represent the mechanism and algorithm settings. The three metrics PFI,
CCI, and SRSI in 2.2 are used to comprehensively evaluate the stability,
cost-effectiveness, and response flexibility of the dispatch strategies
obtained by various simulation scenarios.
Applied Energy 374 (2024) 123950 
10 

--- Page 11 ---

F. Cui et al.
Fig. 7. Electricity balance results for optimization dispatch and Sankey diagrams of energy flows under five obtained strategies.
4.2. Dispatch results of the proposed method
This section shows the performance of the proposed OCTD3 algo-
rithm under MES mechanism. Fig. 6 illustrates the training and evalu-
ation reward curves of the proposed method over 200,000 timesteps,
with evaluation rewards recorded every 2,400 steps. The original data
is smoothed for clarity using a moving average with a window size
of 20. In Fig. 6 (a), the training curve stabilizes near the minimum
value of the greed coefficient 𝜖. In Fig. 6 (b), after an exploration
period, the evaluation reward curve stabilizes around the 1000-step
mark, converging near a reward value of 3.5. Initially, policy gradients
fluctuate due to exploration, but as learning progresses, the model
converges to a stable policy, leading to convergence of both evaluation
and training reward curves, indicating optimal solution attainment for
the energy dispatch task.
Fig. 7 presents the electricity balance results and energy flow under
five dispatch strategies obtained by the MES-OCTD3. The labels 1,
2, and 3 in the electricity balance results correspond to the strategy
options selected by the decision-making agent, representing rapid re-
sponse, long-term balance, and hybrid adjustment modes, respectively.
Positive values for PV, WT, and grid output indicate power generation,
while negative values for the electrolyzer, CAES, BESS, and power users
denote power consumption. The results demonstrate that the proposed
method achieves power balance at every timestep. Sankey diagrams
reveal that the agent explores five types of dispatch strategies. Under
option 1, BESS engages in rapid response charging or discharging,
suitable for times with minimal supply–demand difference and rapid
fluctuations. Option 2 activates only CAES for long-term energy balance
when supply significantly exceeds demand. With option 3, both BESS
and CAES operate, flexibly meeting requirements for rapid response
and long-term balance. The MES-OCTD3 effectively maintains energy
balance in uncertain dispatch environments by adaptively selecting
different strategy networks, showcasing its capacity for efficient energy
management and system reliability.
4.3. Comparative analysis
In this section, the two groups of comparative experiments are
conducted. The results derived from different scenarios are compared
and analyzed.
(1) Metrics results : Figs. 8 and 9 depict the evaluation metrics
recorded during the training of the six comparative experiments, with
corresponding values listed in Table 5 . Fig. 8 illustrates the met-
rics evaluation curves from the OCTD3 method under various energy
Table 5
The test metric values of the trained models under comparison scenarios.
Scenarios PFI (kW) CCI ($) SRSI
Proposed 0: MES-OCTD3 538.1 613.3 0.71
Group 1 1: BESS-OCTD3 3501.4 669.0 0.10
2: CAES-OCTD3 1842.5 745.0 0.31
3: HESS-OCTD3 1197.6 648.5 0.37
Group 2 4: MES-TD3 931.3 629.0 0.43
5: MES-DDPG 1104.6 623.5 0.31
6: MES-PPO 1693.7 677.1 0.71
Table 6
The sensitivity analysis results of the cost weight factor 𝑤 and the learning rate 𝛼.
Index PFI CCI SRSI
cost weight factor 𝑤 0 462.6 602.0 0.605
0.2 536.2 611.2 0.728
0.4 1374.1 685.3 0.550
0.6 2326.5 727.9 0.418
0.8 1818.6 683.4 0.492
1.0 2427.6 738.8 0.429
Learning rate 𝛼 1𝑒−5 2515.5 728.2 0.407
1𝑒−4 536.2 611.2 0.728
1𝑒−3 1399.3 734.2 0.572
1𝑒−2 1664.4 771.9 0.484
1𝑒−1 1218.0 741.2 0.433
storage mechanisms. Compared to single-mode mechanisms, it demon-
strates the superior system stability and cost efficiency under proposed
MES mechanism. Table 5 indicates that under MES mechanism, PFI
and CCI reduce by 55.1% and 5.4%, respectively, with SRSI notably
higher by 91.8%. This underscores the MES mechanism’s ability to
synergize BESS’s rapid response with CAES’s long-term energy balance,
enhancing system resilience amidst environmental fluctuations. Simi-
larly, results in Fig. 9 highlight the significant improvements achieved
by the OCTD3 method. Table 5 indicates that the proposed OCTD3
algorithm achieves a 42.2% lower PFI compared to the best-performing
TD3 algorithm among the other DRL methods, indicating enhanced
system stability. Moreover, the PPO algorithm exhibits poorer perfor-
mance in stability, convergence speed, and metrics compared to TD3
and DDPG, suggesting the suitability of deterministic strategies in this
study. This reinforces the effectiveness of the proposed hierarchical
decision-making framework integrating internal deterministic policy
networks.
(2) Power optimization results: Figs. 10 and 11 depict power op-
timization results for two sets of experiments. The mean line serves
Applied Energy 374 (2024) 123950 
11 

--- Page 12 ---

F. Cui et al.
Fig. 8. Group1: Metrics evaluation curves of different energy storage mechanisms with OCTD3 method: (a) PFI; (b) CCI; (c) SRSI.
Fig. 9. Group2: Metrics evaluation curves of different DRL algorithms with MES mechanism: (a) PFI; (b) CCI; (c) SRSI.
Fig. 10. Group1: Power fluctuations of the interchange power under different energy
storage mechanisms with OCTD3 method.
Fig. 11. Group2: Power fluctuations of the interchange power under different DRL
algorithms with MES mechanism.
as a reference point, indicating the average interchange power 𝑃𝑔,𝑎𝑣
signed with the superior operator, which is set to 130 kW in this study.
Fig. 10 (Group1) shows that MES-OCTD3 exhibits smoother power
profiles compared to single-mode energy storage mechanisms. Fig. 11
shows that MES-OCTD3 achieves lowest power fluctuation compared
to three DRL algorithms. In particular, the proposed method optimizes
interchange power in two main aspects. First, during peak and valley
times, such as the 4–8 h period, the proposed method significantly
reduces the area of fluctuation. Second, in off-peak periods, as observed
in the zoomed-in view from hours 9 to 24, while other methods barely
optimize the quick fluctuations, MES-OCTD3 closely aligns the inter-
change power with the mean line. Thus, the comparative results from
Fig. 12. Group1: Cost analysis under different energy storage mechanisms with OCTD3
method.
Fig. 13. Group2: Cost analysis under different DRL algorithms with MES mechanism.
both groups indicate that the MES mechanism and OCTD3 algorithm
successfully achieve a high degree of interchange power smoothness.
The system is capable not just of compensating for or absorbing sig-
nificant energy during the peak and valley periods but also effectively
Applied Energy 374 (2024) 123950 
12 

--- Page 13 ---

F. Cui et al.
Fig. 14. CAES and BESS charging and discharging strategies under seven scenarios: (a) MES-OCTD3; (b) BESS-OCTD3; (c) CAES-OCTD3; (d) HESS-OCTD3; (e) MES-TD3; (f)
MES-DDPG; (g) MES-PPO;.
tracking and smoothing out rapid fluctuations during off-peak periods,
demonstrating its adaptability in managing complex energy situation.
(3) Comprehensive cost analysis: Figs. 12 and 13 showcase the cost
analysis results of the two sets of comparative scenarios, using radar
charts for visual comparison. According to Eqs. ( 30) and ( 31), the
accumulated comprehensive cost includes the system’s transaction costs
(electricity purchase cost, denoted as 𝐶𝐸𝑃,𝑠𝑦𝑠 ; the deviation fluctu-
ations cost, denoted as 𝐶𝐷𝑃,𝑠𝑦𝑠 ), operational costs of CAES (startup
costs 𝐶𝑆𝑈 and degradation costs 𝐶𝑑𝑒𝑔,𝐶𝐴𝐸𝑆 ), BESS degradation costs
𝑂𝐶𝐵𝐸𝑆𝑆 and electrolyzers degradation costs 𝑂𝐶𝐸𝑙𝑧. Fig. 12 indicates
that compared to the single-mode BESS-OCTD3, the proposed mecha-
nism achieves a lower cost primarily by significantly reducing 𝐶𝐷𝑃,𝑠𝑦𝑠
and𝐶𝐸𝑃,𝑠𝑦𝑠 . Moreover, compared to the single-mode CAES-OCTD3 and
HESS-OCTD3, the proposed mechanism further reduces the comprehen-
sive cost by decreasing the startup costs of CAES. Fig. 13 demonstrates
that OCTD3 algorithm mainly reduces costs by lowering 𝐶𝐷𝑃,𝑠𝑦𝑠 and
𝐶𝐸𝑃,𝑠𝑦𝑠 . In summary, the MES-OCTD3 strategically reduces electricity
purchase and deviation penalty costs, showcasing its efficient energy
expenditure management in integrated storage systems.
(4) Dispatch strategies results: Fig. 14 illustrates the charging and dis-
charging actions of CAES and BESS under seven scenarios. It provides a
visual representation of how each method manages the energy storage
components throughout a typical day. In Fig. 14 (a), the proposed
method utilizes BESS under option 1 for quick adaptability during
times of minor supply–demand deviations. When energy production
exceeds demand, it activates CAES under option 2, and when demand
surpasses supply, it deploys both CAES and BESS under option 3 to
track rapid changes and compensate for shortages. Fig. 14 (b) shows
single-mode BESS’s limited capacity results in a low response level. For
example, at time 7, its maximum adjustment covers only 37% of the
required amount. In Fig. 14 (c), single-mode CAES management leads
to frequent charging and discharging actions during minor supply–
demand differences, resulting in high startup costs. Fig. 14(d) indicates
that single-mode HESS neglects short-term fluctuations optimization.
In Figs. 14 (e), (f), and (g), although 𝑅𝐹
𝑡 is used to guide the agent
in activating different storage modes under various conditions for the
comparison methods, their resulting strategies are suboptimal. TD3
tends to overuse BESS during 4–8 h, lacking a sufficient reserve for
rapid fluctuations. DDPG fails to distinguish CAES and BESS function-
alities, while PPO lacks response volume despite recognizing CAES
advantages.
In summary, comparative experiments highlight the proposed MES
mechanism’s ability to optimize energy management by synergizing
CAES’s long-term storage with BESS’s rapid adjustment capabilities,
while highlight the OCTD3’s superiority in balancing CAES and BESS
roles effectively for comprehensive energy dispatch.
4.4. Sensitivity analysis
In this part, the sensitivity analysis are performed to explore the
impacts of the cost weight factor 𝑤 and learning rate 𝛼.
Table 6 shows the optimization results under different cost weight
factor 𝑤 varied from 0 to 1, and learning rate 𝛼 ranged from 1𝑒−5
to 1𝑒−1. Compared to 𝑤 = 0.0, when 𝑤 = 0.6, the PFI increased by
5.03 times. The CCI increased by about 20.9%, and the SRSI decreased
by approximately −30.9%. These significant changes demonstrate the
sensitivity of the metrics to the cost weight factor within the range of
[0, 0.6], which is because higher weight factors prioritize cost optimiza-
tion, increasing the term 𝜆𝑓𝑃𝐹𝐼 in Eq. (31), consequently raising both
costs and fluctuations. Optimal performance is observed at 𝛼 = 1𝑒−4,
where the values of PFI, CCI, and SRSI are 536.2, 611.2, and 0.728
respectively. This shows that a moderate learning rate can balance the
introduction of new information with the stability of learning. A higher
learning rate can accelerate learning but may lead to over-fitting, while
a lower learning rate can stabilize learning but may slow it down (such
Applied Energy 374 (2024) 123950 
13 

--- Page 14 ---

F. Cui et al.
as 𝛼 = 1𝑒−5). Additionally, with 𝑤 ranging from 0.6 to 1 and 𝛼 from
1𝑒−3 to 1𝑒−1, the variations in the indicators are not drastic, indicating
that the proposed method is robust within certain parameter ranges.
5. Conclusions
In this study, an innovative MES mechanism and the OCTD3 al-
gorithm framework are introduced within a multi-type energy IEMS
to effectively handle the high dynamics of demand and energy sup-
ply. Models for various system components are established, with a
specific focus on the electricity–heat–gas conversion relationships of
CAES under off-design conditions. The dispatch task is formulated as a
SMDP and solved by the OCTD3 method, which aims to optimize three
critical indices: PFI, CCI, and SRSI. This ensures the stability, economic
performance, and flexibility of the dispatch strategy. Two sets of com-
parative simulations involving six scenarios, along with a sensitivity
analysis, are conducted. Significant simulation results underscore the
efficacy of the proposed MES mechanism and OCTD3 approach, with
key conclusions as follows:
(1) Compared to traditional single-mode storage mechanisms, the
proposed MES mechanism improves the SRSI by 91.8%, which high-
lights its enhanced capability to adapt to intermittent renewable energy
outputs and abrupt demand changes.
(2) The proposed MES mechanism reduces the PFI and CCI by 55.1%
and 5.4% respectively, demonstrating significant economic benefits and
system efficiency.
(3) The proposed OCTD3 method develops five types of action
strategies for hybrid CAES–BESS system to exploit the synergistic ad-
vantages of CAES and BESS.
(4) Compared to three benchmark DRL algorithms, the proposed
OCTD3 method reduces total costs by lowering electricity purchase
and deviation fluctuations costs, and decreases PFI by 42.2% through
balancing peak and valley energy loads and swiftly responding to
transient shifts.
Based on the work presented in this paper, future research will
focus on three main areas: integrating various types of RES and storage
technologies to accommodate regional variations in energy supply
and demand; expanding the SMDP framework to include multi-agent
systems, which will promote collaborative strategies among energy
hubs, optimizing regional energy distribution and enhancing system
resilience; and exploring the potential for scalability of the proposed
OCTD3 algorithm within large-scale systems, specifically assessing how
its hierarchical and modular architecture can be adapted to manage
increasingly complex and extensive energy management scenarios.
CRediT authorship contribution statement
Feifei Cui: Writing – original draft, Visualization, Validation, Soft-
ware, Methodology, Investigation, Formal analysis. Dou An: Writing
– review & editing, Investigation, Funding acquisition, Conceptualiza-
tion. Huan Xi: Supervision, Resources, Project administration, Funding
acquisition.
Declaration of competing interest
The authors declare that they have no known competing finan-
cial interests or personal relationships that could have appeared to
influence the work reported in this paper.
Data availability
All the data have been well presented in the manuscript.
Acknowledgments
This work was supported by the National Natural Science Founda-
tion of China (NO. 62173268, NO. 61833015 and No. 52076163).
References
[1] Shang W-L, Ling Y, Ochieng W, Yang L, Gao X, Ren Q, Chen Y, Cao M.
Driving forces of CO2 emissions from the transport, storage and postal sectors:
A pathway to achieving carbon neutrality. Appl Energy 2024;365:123226. http:
//dx.doi.org/10.1016/j.apenergy.2024.123226.
[2] Samani E, Mohsenian-Rad H. Understanding convergence bids during black-
outs: Analytical results and real-world implications. IEEE Trans Power Syst
2023;38(4):3642–53. http://dx.doi.org/10.1109/TPWRS.2022.3208227.
[3] Bullich-Massagué E, Cifuentes-García F-J, Glenny-Crende I, Cheah-Mañé M,
Aragüés-Peñalba M, Díaz-González F, Gomis-Bellmunt O. A review of energy
storage technologies for large scale photovoltaic power plants. Appl Energy
2020;274:115213. http://dx.doi.org/10.1016/j.apenergy.2020.115213.
[4] Teng S, Xi H. Experimental evaluation of vortex tube and its application in
a novel trigenerative compressed air energy storage system. Energy Convers
Manage 2022;268:115972. http://dx.doi.org/10.1016/j.enconman.2022.115972.
[5] Tian W, Xi H. Comparative analysis and optimization of pumped thermal
energy storage systems based on different power cycles. Energy Convers Manage
2022;259:115581. http://dx.doi.org/10.1016/j.enconman.2022.115581.
[6] Leon JI, Dominguez E, Wu L, Marquez Alcaide A, Reyes M, Liu J. Hybrid energy
storage systems: Concepts, advantages, and applications. IEEE Ind Electron Mag
2021;15(1):74–88. http://dx.doi.org/10.1109/MIE.2020.3016914.
[7] Giovanniello MA, Wu X-Y. Hybrid lithium-ion battery and hydrogen energy
storage systems for a wind-supplied microgrid. Appl Energy 2023;345:121311.
http://dx.doi.org/10.1016/j.apenergy.2023.121311.
[8] Argyrou MC, Marouchos CC, Kalogirou SA, Christodoulides P. A novel power
management algorithm for a residential grid-connected pv system with battery-
supercapacitor storage for increased self-consumption and self-sufficiency. Energy
Convers Manage 2021;246:114671. http://dx.doi.org/10.1016/j.enconman.2021.
114671.
[9] Jahanbin A, Abdolmaleki L, Berardi U. Techno-economic feasibility of integrating
hybrid-battery hydrogen energy storage in academic buildings. Energy Convers
Manage 2024;309:118445. http://dx.doi.org/10.1016/j.enconman.2024.118445.
[10] Yousri D, Farag HE, Zeineldin H, El-Saadany EF. Integrated model for optimal
energy management and demand response of microgrids considering hybrid
hydrogen-battery storage systems. Energy Convers Manage 2023;280:116809.
[11] Zeynali S, Rostami N, Ahmadian A, Elkamel A. Robust multi-objective thermal
and electrical energy hub management integrating hybrid battery-compressed
air energy storage systems and plug-in-electric-vehicle-based demand response.
J Energy Storage 2021;35:102265. http://dx.doi.org/10.1016/j.est.2021.102265.
[12] Chen Y, Hu S, Zheng Y, Xie S, Yang Q, Wang Y, Hu Q. Coordinated optimization
of logistics scheduling and electricity dispatch for electric logistics vehicles
considering uncertain electricity prices and renewable generation. Appl Energy
2024;364:123147. http://dx.doi.org/10.1016/j.apenergy.2024.123147.
[13] Wang C, He Q, Li Z, Yu J, Bello IT, Zheng K, Han M, Ni M. A novel in-tube
reformer for solid oxide fuel cell for performance improvement and efficient
thermal management: A numerical study based on artificial neural network and
genetic algorithm. Appl Energy 2024;357:122030. http://dx.doi.org/10.1016/j.
apenergy.2023.122030.
[14] Dinh HT, haeng Lee K, Kim D. Supervised-learning-based hour-ahead demand
response for a behavior-based home energy management system approximating
MILP optimization. Appl Energy 2022;321:119382. http://dx.doi.org/10.1016/j.
apenergy.2022.119382.
[15] Morshed MJ, Asgharpour A. Hybrid imperialist competitive-sequential quadratic
programming (HIC-SQP) algorithm for solving economic load dispatch with
incorporating stochastic wind power: A comparative study on heuristic optimiza-
tion techniques. Energy Convers Manage 2014;84:30–40. http://dx.doi.org/10.
1016/j.enconman.2014.04.006.
[16] Pérez-Iribarren E, González-Pino I, Azkorra-Larrinaga Z, Odriozola-Maritorena M,
Gómez-Arriarán I. A mixed integer linear programming-based simple method for
optimizing the design and operation of space heating and domestic hot water hy-
brid systems in residential buildings. Energy Convers Manage 2023;292:117326.
http://dx.doi.org/10.1016/j.enconman.2023.117326.
[17] Rahim S, Wang Z, Ju P. Overview and applications of Robust optimization in
the avant-garde energy grid infrastructure: A systematic review. Appl Energy
2022;319:119140. http://dx.doi.org/10.1016/j.apenergy.2022.119140.
[18] Yi Z, Luo Y, Westover T, Katikaneni S, Ponkiya B, Sah S, Mahmud S, Raker D,
Javaid A, Heben MJ, Khanna R. Deep reinforcement learning based optimization
for a tightly coupled nuclear renewable integrated energy system. Appl Energy
2022;328:120113. http://dx.doi.org/10.1016/j.apenergy.2022.120113.
[19] Ren K, Liu J, Wu Z, Liu X, Nie Y, Xu H. A data-driven DRL-based home
energy management system optimization framework considering uncertain house-
hold parameters. Appl Energy 2024;355:122258. http://dx.doi.org/10.1016/j.
apenergy.2023.122258.
[20] Ma S, Liu H, Wang N, Huang L, Goh HH. Incentive-based demand response under
incomplete information based on the deep deterministic policy gradient. Appl
Energy 2023;351:121838. http://dx.doi.org/10.1016/j.apenergy.2023.121838.
Applied Energy 374 (2024) 123950 
14 

--- Page 15 ---

F. Cui et al.
[21] Yin L, Li Y. Hybrid multi-agent emotional deep q network for generation control
of multi-area integrated energy systems. Appl Energy 2022;324:119797. http:
//dx.doi.org/10.1016/j.apenergy.2022.119797.
[22] Rezaeimozafar M, Duffy M, Monaghan RF, Barrett E. A hybrid heuristic-
reinforcement learning-based real-time control model for residential behind-the-
meter PV-battery systems. Appl Energy 2024;355:122244. http://dx.doi.org/10.
1016/j.apenergy.2023.122244.
[23] Wen D, Aziz M. Data-driven energy management system for flexible operation
of hydrogen/ammonia-based energy hub: A deep reinforcement learning ap-
proach. Energy Convers Manage 2023;291:117323. http://dx.doi.org/10.1016/
j.enconman.2023.117323.
[24] Zhang B, Hu W, Li J, Di Cao, Huang R, Huang Q, Chen Z, Blaabjerg F. Dy-
namic energy conversion and management strategy for an integrated electricity
and natural gas system with renewable energy: Deep reinforcement learning
approach. Energy Convers Manage 2020;220:113063. http://dx.doi.org/10.1016/
j.enconman.2020.113063.
[25] Zhang B, Hu W, Di Cao, Li T, Zhang Z, Chen Z, Blaabjerg F. Soft actor-critic
–based multi-objective optimized energy conversion and management strategy
for integrated energy systems with renewable energy. Energy Convers Manage
2021;243:114381. http://dx.doi.org/10.1016/j.enconman.2021.114381.
[26] Liu J, Li Y, Ma Y, Qin R, Meng X, Wu J. Coordinated energy management for
integrated energy system incorporating multiple flexibility measures of supply
and demand sides: A deep reinforcement learning approach. Energy Convers
Manage 2023;297:117728. http://dx.doi.org/10.1016/j.enconman.2023.117728.
[27] Wang Z, Xiao F, Ran Y, Li Y, Xu Y. Scalable energy management approach of
residential hybrid energy system using multi-agent deep reinforcement learn-
ing. Appl Energy 2024;367:123414. http://dx.doi.org/10.1016/j.apenergy.2024.
123414.
[28] Wang Y, Qiu D, Sun M, Strbac G, Gao Z. Secure energy management
of multi-energy microgrid: A physical-informed safe reinforcement learn-
ing approach. Appl Energy 2023;335:120759. http://dx.doi.org/10.1016/j.
apenergy.2023.120759, URL: https://www.sciencedirect.com/science/article/pii/
S030626192300123X.
[29] Qiu D, Wang Y, Zhang T, Sun M, Strbac G. Hierarchical multi-agent re-
inforcement learning for repair crews dispatch control towards multi-energy
microgrid resilience. Appl Energy 2023;336:120826. http://dx.doi.org/10.1016/j.
apenergy.2023.120826, URL: https://www.sciencedirect.com/science/article/pii/
S0306261923001903.
[30] Cui F, An D, Teng S, Lin X, Li D, Xi H. Cogeneration systems of solar energy
integrated with compressed air energy storage systems: A comparative study
of various energy recovery strategies. Case Stud Therm Eng 2023;51:103521.
http://dx.doi.org/10.1016/j.csite.2023.103521.
[31] Huy THB, Truong Dinh H, Ngoc Vo D, Kim D. Real-time energy scheduling for
home energy management systems with an energy storage system and electric
vehicle based on a supervised-learning-based strategy. Energy Convers Manage
2023;292:117340. http://dx.doi.org/10.1016/j.enconman.2023.117340.
[32] Zhao D, Xia Z, Guo M, He Q, Xu Q, Li X, Ni M. Capacity optimization and
energy dispatch strategy of hybrid energy storage system based on proton
exchange membrane electrolyzer cell. Energy Convers Manage 2022;272:116366.
http://dx.doi.org/10.1016/j.enconman.2022.116366.
[33] Nojavan S, Zare K, Mohammadi-Ivatloo B. Application of fuel cell and elec-
trolyzer as hydrogen energy storage system in energy management of electricity
energy retailer in the presence of the renewable energy sources and plug-in
electric vehicles. Energy Convers Manage 2017;136:404–17. http://dx.doi.org/
10.1016/j.enconman.2017.01.017.
[34] Abomazid AM, El-Taweel NA, Farag HEZ. Optimal energy management of
hydrogen energy facility using integrated battery energy storage and solar
photovoltaic systems. IEEE Trans Sustain Energy 2022;13(3):1457–68. http://
dx.doi.org/10.1109/TSTE.2022.3161891.
[35] Xu B, Zhao J, Zheng T, Litvinov E, Kirschen DS. Factoring the cycle aging
cost of batteries participating in electricity markets. IEEE Trans Power Syst
2018;33(2):2248–59. http://dx.doi.org/10.1109/TPWRS.2017.2733339.
[36] Wu D, Bai J, Wei W, Chen L, Mei S. Optimal bidding and scheduling of
AA-CAES based energy hub considering cascaded consumption of heat. Energy
2021;233:121133. http://dx.doi.org/10.1016/j.energy.2021.121133.
[37] Xu X, Hu W, Di Cao, Huang Q, Liu W, Liu Z, Chen Z, Lund H. Designing a
standalone wind-diesel-CAES hybrid energy system by using a scenario-based
bi-level programming method. Energy Convers Manage 2020;211:112759. http:
//dx.doi.org/10.1016/j.enconman.2020.112759.
[38] Isohätäiä J, Haskell WB. Risk-aware semi-Markov decision processes. In: 2017
IEEE 56th annual conference on decision and control. 2017, p. 4303–8. http:
//dx.doi.org/10.1109/CDC.2017.8264293.
[39] Sutton RS, Precup D, Singh S. Between MDPs and semi-MDPs: A frame-
work for temporal abstraction in reinforcement learning. Artificial Intelligence
1999;112(1):181–211. http://dx.doi.org/10.1016/S0004-3702(99)00052-1.
[40] Ling Z, Hu F, Liu T, Jia Z, Han Z. Hierarchical deep reinforcement learning
for self-powered monitoring and communication integrated system in high-
speed railway networks. IEEE Trans Intell Transp Syst 2023;24(6):6336–49.
http://dx.doi.org/10.1109/TITS.2023.3248161.
[41] Sutton RS, McAllester D, Singh S, Mansour Y. Policy gradient methods for rein-
forcement learning with function approximation. MIT Press; 1999, p. 1057–63,
https://dl.acm.org/doi/10.5555/3009657.3009806.
[42] Daneshvar Garmroodi A, Nasiri F, Haghighat F. Optimal dispatch of an energy
hub with compressed air energy storage: A safe reinforcement learning approach.
J Energy Storage 2023;57:106147. http://dx.doi.org/10.1016/j.est.2022.106147.
[43] Wen D, Aziz M. Data-driven energy management system for flexible operation
of hydrogen/ammonia-based energy hub: A deep reinforcement learning ap-
proach. Energy Convers Manage 2023;291:117323. http://dx.doi.org/10.1016/
j.enconman.2023.117323.
[44] El-Taweel NA, Khani H, Farag HEZ. Analytical size estimation methodologies
for electrified transportation fueling infrastructures using public–domain market
data. IEEE Trans Transp Electrif 2019;5(3):840–51. http://dx.doi.org/10.1109/
TTE.2019.2927802.
[45] Wang Z, He H, Peng J, Chen W, Wu C, Fan Y, Zhou J. A comparative study
of deep reinforcement learning based energy management strategy for hybrid
electric vehicle. Energy Convers Manage 2023;293:117442. http://dx.doi.org/
10.1016/j.enconman.2023.117442.
[46] Zhang Y, Zhang C, Fan R, Huang S, Yang Y, Xu Q. Twin delayed deep determinis-
tic policy gradient-based deep reinforcement learning for energy management of
fuel cell vehicle integrating durability information of powertrain. Energy Convers
Manage 2022;274:116454. http://dx.doi.org/10.1016/j.enconman.2022.116454.
Applied Energy 374 (2024) 123950 
15 