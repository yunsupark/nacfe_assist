<!-- page: 1 -->
Cover page.

<!-- page: 2 -->
Acknowledgements page.

<!-- page: 3 -->
Table of Contents.

<!-- page: 4 -->
Table of Contents (continued).

<!-- page: 5 -->
Table of Figures.

<!-- page: 6 -->
Table of Figures (continued).

<!-- page: 7 -->
Table of Figures (continued).

<!-- page: 8 -->
Executive Summary.

<!-- page: 9 -->
Executive Summary (continued).

<!-- page: 10 -->
Executive Summary (continued).

<!-- page: 11 -->
Executive Summary (continued).

<!-- page: 12 -->
Executive Summary (continued).

<!-- page: 13 -->
Executive Summary (continued).

<!-- page: 14 -->
Executive Summary (continued).

<!-- page: 15 -->
Executive Summary (continued).

<!-- page: 16 -->
Executive Summary (continued).

<!-- page: 17 -->
Executive Summary (continued).

<!-- page: 18 -->
Executive Summary (continued).

<!-- page: 19 -->
Executive Summary (continued).

<!-- page: 20 -->
Executive Summary (continued).

<!-- page: 21 -->
Executive Summary (continued).

<!-- page: 22 -->
4. Introduction

This report focuses on charging infrastructure considerations for North American commercial battery electric vehicles (CBEVs). In its previous Guidance Reports, *Electric Trucks – Where They Make Sense* and *Medium Duty Electric Trucks – Cost of Ownership*, NACFE found that the benefits of electric vehicles can be huge, particularly for certain duty cycles, but so are the power requirements for charging them. In fact, the previous reports identified charging infrastructure as one of the largest unknowns and sources of anxiety for fleets considering near-term adoption of this technology.<sup>1 2</sup> As one interviewee declared, “The trucks are the easy part; it’s the charging that’s the challenge.” Another went so far as to refer to charging as electrification’s “Achilles heel”. In fact, according to a 2018 UPS/GreenBiz research study, 44% of respondents identified a lack of EV charging infrastructure at facilities as one of the top challenges to electrifying fleets, making infrastructure the second highest ranked barrier, after only the initial purchase price.<sup>3</sup> Similarly, according to The Climate Group’s 2019 EV100 Annual Report, over 70% of EV100 companies identified a lack of charging infrastructure as a "significant" or "very significant" barrier.<sup>4</sup> However, fleets are adopting electric truck technologies at an unprecedented pace, with initial models already on the road and many more on the way, on reserve and pre-order until production ramps up. But will fleets be ready for these vehicles once they’re delivered? Will the electricity grid? Is the trucking industry putting the cart before the horse by focusing on the vehicles themselves before a clear path to fuel — or in this case, charge — them? 

Throughout this report, NACFE’s intent is to answer this question by providing an unbiased report detailing the multiple factors to consider in infrastructure planning for charging CBEVs. These include the Electric Vehicle Supply Equipment (EVSE) itself, grid preparedness and integration, relevant utility rate structures, on-site logistics, timelines, business models, and financial assistance. We also examine considerations such as safety, scaling, grid services, integrating renewables, and automation. 

While there is no “one size fits all” solution to charging, there are common steps and considerations that any fleet considering deployment of electric trucks should undertake in order to ensure they have a complementary and cost-effective charging strategy in place.

> **Figure — (Untitled)**
> Type: Photograph / illustration
> Axes: none
> Series: none
> Values: none
> Notes: Illustration of an electric delivery van plugged into a charging station.

<!-- page: 23 -->
This is the third in a series of NACFE guidance reports on electric trucks. It will be followed by guidance reports on Class 7 day cabs and Class 8 long-haul electric vehicles.

## 5. Scope

The report covers charging considerations for CBEVs currently in production for freight delivery. Because most CBEVs are currently being deployed in the goods movement sector in the medium-duty urban delivery and drayage sectors, much of the best practices and lessons learned come from these applications. And while we will touch on considerations for long-haul CBEVs, much of this information is speculative at this point in time as electric trucks have yet to be deployed for this application in any meaningful way. 

NACFE’s mission is to improve the efficiency of freight hauling by commercial vehicles. As such, this report excludes discussion of charging infrastructure for buses, recreational vehicles, vocational vehicles, ambulances, and other special purpose vehicles. 

This report includes relevant references to charging infrastructure for battery electric buses and cars, where parallels exist and field data on production trucks are not yet available. 

The report assumes that vehicles are charged via electrons provided by the shared electricity grid, though the potential for on-site renewables integration and/or microgrid development will be touched on briefly in Section 13.5. The report focuses on how electricity flows from the grid to the charging site, i.e. distribution, and then to the truck’s batteries. An overview of the full grid, including generation and transmission, can be found in the first Guidance Report in this series, *Electric Trucks – Where They Make Sense* and in the Charging System Basics section below.<sup>5</sup>

The report focuses on charging of commercial battery electric vehicles and does not touch on charging/fueling considerations for other types of electric vehicles such as fuel cell electric vehicles. The report does not discuss use of range extension alternatives such as with various hybrid approaches. 

## 6. NACFE’s Mission

NACFE’s overriding principle in reporting on technologies is to provide an unbiased perspective. NACFE recognizes that it also has vested interests and an agenda. The mission of the North American Council for Freight Efficiency is simply to improve the efficiency of North American goods movement. NACFE pursues this goal in two ways: By improving the quality of the information flow and by highlighting successful adoption of technologies.

## 7. Report Methodology

NACFE’s research for this report included interviewing key people with first-hand knowledge of electric vehicle charging infrastructure at fleets, manufacturers, and industry groups. The report includes an extensive list of references to assist readers interested in pursuing more detail. These references were researched with the same diligence and thoughtful processes NACFE

<!-- page: 24 -->
uses with its technology Confidence Reports and its previous Guidance Reports. Interviewees were specifically asked what they would want to see in this report and NACFE has taken care to include these wants in the final report. This report builds off the NACFE Guidance Reports: *Electric Trucks – Where They Make Sense*, published in May 2108, and *Medium Duty Electric Trucks – Cost of Ownership*, published October 2018. Subsequent reports will focus on various vehicle classes not covered in previous reports.

## 8. Charger Basics

When planning for charging infrastructure, fleets must plan for three separate but related components: hardware, software/networking, and maintenance. 

> **Figure 1: CBEV Charging Infrastructure Components (NACFE)**
> Type: Diagram / schematic
> Axes: none
> Series: none
> Values: none
> Notes: Diagram showing three components: Hardware (physical charging stations, ports, panels, transformers, etc., including wiring/conduit, transformer upgrades, and installation), Software/Networking (can be built-in to chargers or purchased from third-party vendors, enables cost-effective charging management, integration of DERs and grid services, provides data and analytics), and Maintenance (timely repair of equipment, service packages available to monitor and repair equipment).

### 8.1. Charger Types

Much has been written on electric vehicles themselves, but much less is known about the system for charging them. However, the vehicles are only as good as they are functional, and charging is clearly necessary for keeping them running. That said, it is important that fleets considering electric vehicle adoption understand the basics of charging.

As electric trucks run, performing their daily operations, they gradually deplete their batteries, the same way diesel trucks deplete their fuel. These batteries need to be recharged, which is done via a charging station, also known as Electric Vehicle Supply Equipment (EVSE).

<!-- page: 25 -->
#### 8.1.1. Plug-In Chargers

The most common type of EVSE is a plug-in charging station. Similar to how a diesel fuel pump fits into a port on the side of the truck to refuel, a plug-in charging station, as the name implies, plugs into a port on the truck to recharge it, in this case with electrons. However, unlike diesel pumps, which were standardized some time ago, for electric vehicles, there are a number of competing charging station connector types in the world. This is somewhat similar to what a traveler experiences trying to plug in portable devices like phone chargers or hair dryers to wall outlets across the world, needing an adapter kit and accepting different regional operating parameters. One example of a charging station and connection is shown in Figure 2.<sup>6</sup>

> **Figure 2: Example Electric Charging Station and Connection (Blink Charging)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing a Blink charging station and a connector cable.

The European Union is evolving standards under the International Electrotechnical Commission (IEC) under standards IEC 61851 Electric Vehicle Conductive Charging System and IEC 62196 Plugs, Socket-Outlets, Vehicle Couplers and Vehicle Inlets.<sup>7 8 9</sup> The U.S. is evolving under the Society of Automotive Engineers standard SAE J1772 SAE Electric Vehicle and Plug in Hybrid Electric Vehicle Conductive Charge Coupler and various other commercial products from Tesla.<sup>10</sup> In early 2010, Japan formalized their DC charging standard, CHAdeMO, which was the only DC charging option until the emergence of CCS in 2012–2013.<sup>11</sup> However, CHAdeMO appears to be falling behind in its DCFC lead, as both vehicles and charging networks in both Europe and the U.S. are now overwhelmingly favoring CCS as the standard charging architecture. Though as heavy-duty commercial vehicles demand faster and faster charging, CCS may be rivaled by new connector design standards currently being developed by the High Power Charging for Commercial Vehicles (HPCCV) Task Force, part of the Charging Interface Initiative e. V. (CharIN e. V.), a worldwide industry alliance founded by Audi, BMW, Daimler, Mennekes, Opel, Phoenix Contact, Porsche, TÜV SÜD and Volkswagen and focused on

<!-- page: 26 -->
drafting requirements to accelerate the evolution of charging related standards.<sup>12</sup> The HPCCV Task Force, which has over 160 members from 80 companies, is hoping new, open standards can allow for even faster charging, perhaps up to 1 MW, which is being discussed as needed for rapid charging of Class 8 electric trucks with ranges of 400 or more miles.<sup>13 14</sup> An additional standard, SAE J3068, is under development for higher rates of AC charging using three-phase power, which is common at commercial and industrial locations in the U.S.<sup>15</sup> SAE J3068 targets power levels between 6kW and 130kW and is planned to be used for perhaps the first time in North America as part of Volvo’s Low Impact Green Heavy Transport Solutions (LIGHTS) project in southern California.<sup>16</sup> Tesla also appears to be developing its own type of charging port, with an 8-pin configuration, as seen at the Tesla Semi unveil in November 2017 (and shown in Figure 5 below).<sup>17</sup> The variety of automotive connectors on the market today are illustrated in Figure 3. And a handful of the actual connectors are shown in Figure 4.

> **Figure 3: Charging Connector Types (ChargeHub)**
> Type: Table
> Axes: none
> Series: Connectors vs Level, Asian Makes, US / EU Makes, Tesla
> Values: 
> - Wall outlets (Nema 515, Nema 520): Level 1, Asian Makes: With adapter, US / EU Makes: With adapter, Tesla: With adapter
> - Port J1772: Level 2, Asian Makes: Yes, US / EU Makes: Yes, Tesla: With adapter
> - Nema 1450 (RV plug): Level 2, Asian Makes: With adapter, US / EU Makes: With adapter, Tesla: With adapter
> - CHAdeMO: Level 3, Asian Makes: Yes, US / EU Makes: No, Tesla: With adapter
> - SAE Combo CCS: Level 3, Asian Makes: No, US / EU Makes: Yes, Tesla: No
> - Tesla HPWC: Level 2, Asian Makes: No, US / EU Makes: No, Tesla: Yes
> - Tesla supercharger: Level 3, Asian Makes: No, US / EU Makes: No, Tesla: Yes
> Notes: Source: ChargeHub [18]

<!-- page: 27 -->
> **Figure 4: Side-by-side Comparison of Common Charging Connector Types (from left to right: CHAdeMO, CCS, J3068)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing CHAdeMO, CCS, and J3068 connectors side by side. Source: Wikimedia Commons [19]

> **Figure 5: Tesla Semi Prototype Charging Port (Teslarati)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing the Tesla Semi prototype charging port with an 8-pin configuration. Source: Teslarati [5]

Some connector types provide flexibility in charging. For example, the J1772 combo (also known as the combined charging system or CCS) connector is unique because a driver can use the same charge port when charging with Level 1, 2, or DCFC equipment. The only difference is that the DCFC connector has two additional bottom pins.

<!-- page: 28 -->
Standardizing connectors may eventually occur for regional marketplaces as one configuration wins significant market share advantage over others. In the near term, commercial vehicles may be developed with several adapters to deal with various charging station constraints, or forced to use proprietary connections and be limited to proprietary charging stations. The connector choice may not be an issue for fleets with only one CBEV model and with dedicated A-B-A type routes where the vehicle only charges from its home base. Where that fleet may be using competing CBEV models from different manufacturers, but wanting to use the same charging system, there may be need for adapters. This situation is somewhat akin to needing to have diesel, gasoline and natural gas systems to support today’s existing mixed fleets.

Obviously, it is important to pair electric trucks with the appropriate type of connector. Some vehicles carry multiple connections to allow flexibility in charging station types and capacities. An example is the Mitsubishi Fuso eCanter shown in Figure 6 with two charging methods, a standard 230 VAC single-phase J1772 connection and CHAdeMO 50 kW DC connection.<sup>20</sup> Others, such as the TransPower terminal tractor use only one as shown in Figure 7, a 208V 3-phase, 200A 70 kWh fed on-board system.<sup>21 22</sup> So for fleets who choose their vehicles first, they will need to know what type of port the truck has in order to plan which charger type(s) to purchase. 

> **Figure 6: Mitsubishi Fuso eCanterJ1772 and CHAdeMO Charging Connections (Mihelic)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing Mitsubishi Fuso eCanter charging connections.

<!-- page: 29 -->
> **Figure 7: TransPower EV Yard Tractor 208V Connector (Mihelic, TransPower)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing TransPower EV yard tractor connector.

Maneuverability is also a consideration, as the wires used to carry the energy from the charging station to the vehicle increase in size and complexity with increasing power level demands. Though cable diameter can be reduced by incorporating a dielectric liquid cooling system. The reduced size permits higher level charging rates at a safe temperature with cabling that can be maneuvered more easily by an individual.<sup>23</sup> Figure 8 shows the temperature difference over time. Figure 9 outlines the charge levels.

> **Figure 8: Charging Temperature With and Without High Power Charging Technology (Phoenix Contact)**
> Type: Line chart
> Axes: Time ($t$) vs Temperature Difference ($\Delta T [K]$)
> Series: Temperature characteristic without cooling, Temperature characteristic with the HPC cooling system ($I = 500\text{ A}$)
> Values: Qualitative chart showing higher temperature rise without cooling and stable lower temperature with cooling up to $\Delta T_{max}$.
> Notes: Integrated temperature sensors measure the development of heat. The cooling system ensures that the normative limit temperature $\Delta T_{max}$ is not exceeded. Source: Phoenix Contact [24]

<!-- page: 30 -->
> **Figure 9: Liquid Cooled Cables Enable Higher Level Charging (ITT Cannon)**
> Type: Diagram / Infographic
> Axes: Power (2.2kW to 500kW) vs Categories (Europe / USA: AC, DC STANDARD, DC COOLED HPC)
> Series: Europe, USA
> Values: 
> - Europe AC: 2.2kW to 43kW
> - Europe DC Standard: 22kW, 50kW, 100kW, 150kW
> - Europe DC Cooled HPC: 150kW to 500kW
> - USA AC: 2.2kW to 22kW
> - USA DC Standard: 22kW to 100kW
> - USA DC Cooled HPC: 150kW to 500kW
> Notes: Source: ITT Cannon [23]

For example, though not yet available in the U.S., Phoenix Contact is developing a 500-amp cooled cable solution with a 35.7mm diameter.<sup>25</sup> Cable ergonomics and safety will be a consideration for high power/high speed charging systems, where cable weight and flexibility will be challenged at megawatt levels. Figure 10 illustrates the ergonomics of a TransPower charging system.

> **Figure 10: High Speed Charging Requires Large Diameter Cabling (TransPower)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing individual handling a large cable from a TransPower charging system.

#### 8.1.2. Wireless Chargers

An alternative to charging through wires and plugs is termed wireless power transfer (WPT). Wireless charging can be via inductive or capacitive methods. A technical overview of these two methods can be found in a National Academies Press report by Khurram Afridi, Wireless

<!-- page: 31 -->
Charging of Electric Vehicles.<sup>26</sup> Inductive charging employs magnetic field coupling between an on-ground or in-ground coil(s) and coil(s) mounted on vehicle. A variation of this is called magnetic resonance charging, and a Wi Tricity example is shown in Figure 11.<sup>27</sup> Capacitive methods use electric field coupling between in-ground or on-ground plates and those on the vehicle. Through these methods, electricity is able to be transferred through air, water, and even ice. Because wireless charging doesn’t involve plugging anything in to the truck, proponents see it as a way to mitigate issues with workers forgetting to plug in, doing it improperly, or driving away while still plugged in.

Wireless charging protocols are in use with automobiles and some buses. An example is the 2017 Recommended Practice SAE J2954 Wireless Power Transfer for Light-Duty Plug-In/Electric Vehicles and Alignment Methodology.<sup>28</sup> Applicability of wireless charging to trucks is being investigated both in static situations where the vehicle is not moving, and in on-road methods were the vehicle is moving. Static charging presents the least technical challenge for wireless. The efficiency of transferring energy to the vehicle wirelessly is stated as 90%-93%, and energy levels from 3.6 to 22 kW (Level 2) are in use with automobiles.<sup>29 30</sup>

Transit bus wireless charging at 200 kW level are in use. For example, Momentum Dynamics was reported in April 2018 as having deployed a 200-kW system in Wenatchee, Washington servicing a BYD K9S bus as shown in Figure 12.<sup>31 32</sup> The system demonstrated rapid charging while the bus makes a brief transit stop. However, for the time being, wireless charging technology appears too expensive for the trucking market, with a few exceptions for niche markets. For example, wireless charging may be an opportunity for heavy-duty trucks that sit for long periods waiting to pick up loads from ports to charge while they’re waiting. It is also being considered as a solution in port applications where union contracts may prevent workers from physically plugging in charging cables. However, some see a bigger opportunity for wireless charging in the trucking sector, as evidenced by Volvo’s announcement in early 2019 that it had invested an undisclosed amount in Momentum Dynamics.<sup>33</sup>

> **Figure 11: Example Magnetic Resonance Wireless Charging System (Wi Tricity)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing magnetic resonance wireless charging pads with vehicles. Source: Wi Tricity [27]

<!-- page: 32 -->
> **Figure 12: Example 200kW Wireless Charging (Momentum Dynamics)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing a BYD K9S bus charging wirelessly over a Momentum Dynamics pad in Wenatchee, WA. Source: Momentum Dynamics [32]

#### 8.1.3. Conductive Chargers

Overhead or in-ground conductive charging systems are also being investigated for both on-the-road and stationery depot charging.<sup>34</sup> As far as on-the-road systems, according to Trafikverket, the Swedish Transport Administration, a “test stretch of the electric road will be inaugurated on the E16 in Sandviken (Figure 13). With that, Sweden will become one of the first countries in the world to conduct tests with electric power for heavy transports on public roads.”<sup>35</sup>

In-ground conductive charging is also being tested as seen in Figure 14. “Approximately two kilometers of electric rail have been installed along public road 893, between the Arlanda Cargo Terminal and the Rosersberg logistics area outside Stockholm. The electrified road works by transferring energy to the vehicle from a rail in the road through a movable arm. The arm detects the location of the rail in the road and as long as the vehicle is above the rail, the contact will be in a lowered position. The electrified road will be used by electric trucks developed as part of the project.”<sup>36</sup>

<!-- page: 33 -->
> **Figure 13: Overhead Conductive Charging (Trafikverket)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing a Scania truck utilizing overhead catenary lines on the E16 in Sweden. Source: Trafikverket [35]

> **Figure 14: In-Ground Conductive Charging (eRoadArlanda)**
> Type: Photograph and diagram
> Axes: none
> Series: none
> Values: none
> Notes: Photos and diagram showing in-ground rail and movable arm on an eRoadArlanda truck. Source: eRoadArlanda [36]

<!-- page: 34 -->
A stationary version of the pantograph system illustrates an alternative to a manual plug-in conductive charging. The vehicle drives under the charging point and the pantograph, whether roof-mounted or inverted, deploys to make a physical connection for charging (Figure 15). An example of this is shown in Figure 16, where an overhead stationary charging system, installed in 2015, is in use in Louisville, Kentucky.<sup>37 38</sup> A video of a Louisville bus engaging the overhead charging system is provided by the U.S. Department of Energy.<sup>39</sup> King County Metro’s all-electric buses also use this same pantograph system for charging in Washington (Figure 17), and a similar video is provided by Proterra.<sup>40</sup> An alternative system from ABB is providing 450kW level fast charging for buses via the system shown in Gothenburg, Sweden.<sup>41</sup>

> **Figure 15: Roof-Mounted vs. Inverted Pantograph Systems (Proterra)**
> Type: Diagram
> Axes: none
> Series: none
> Values: none
> Notes: Diagram showing Roof-Mounted Pantograph vs Inverted Pantograph. Source: Proterra [42]

> **Figure 16: Stationary Pantograph Conductive Fast Charging (Ryan & DOE)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo sequence showing bus engaging overhead pantograph charging in Louisville, KY. Source: Ryan & DOE [37, 38]

<!-- page: 35 -->
> **Figure 17: Stationary Pantograph Conductive Fast Charging in King County (Proterra)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing electric bus charging via stationary pantograph in King County. Source: Proterra [39]

Notably, these sort of on-road charging options, whether mobile or stationary, make sense for vehicles like buses that have a known route that infrastructure can be planned around. However, many commercial vehicles do not have this luxury. Rather, many trucks have operations more akin to a honeybee’s. That is, their routes are constantly changing, but they always return to their home or hive – in this case, often a depot. Therefore, because the depot is the only location fleets are able to predict with certainty that the trucks will return, most charging infrastructure is installed there.

The use of battery swapping as a charging scheme somewhat disappeared from active discussion for several years but has recently reappeared.<sup>43 44</sup> Battery swapping is advertised as a mechanism to rapidly charge vehicles by simply replacing the battery packs. The Smith-Newton CBEV had this as a possible option with the battery packs located outside of the frame for easy access as shown in Figure 18. The swapping operation was never refined to be quick, and overnight charging evolved as the primary method for restoring battery charge. The Navistar/Modec eStar was designed in 2009-2010 with a “quick-change battery (that) can be swapped out in under 20 minutes.”<sup>45</sup> An engineer tied to that program felt this battery swapping was best suited for fleets with two- or three-shift operations. Ones shift operations relied on plug-in power for single shift users.<sup>46</sup> In automotive, the Better Place Company introduced battery swapping as a service in Israel with goals of marketing the scheme worldwide. The concept was to develop quick-and-drop battery switch stations with interchangeable batteries pre-charged and waiting for the next vehicle.<sup>47</sup> Ultimately this business proposal never materialized into long-term viability.<sup>48</sup> The complexity of rapid battery pack swapping highlights the need for uniform battery configurations across OEMs with consistent system requirements that include rapid swapping.<sup>49</sup> A variation on this was proposed by Mihelic in 2017 where the battery packs would be located in Class 8 semi-trailers, which inherently get swapped regularly at facilities and due

<!-- page: 36 -->
to the 3-to-1 trailer to tractor average U.S. ratio, sit idly for long periods allowing slow, inexpensive charging rates.<sup>50</sup> The medium-duty market with fixed box vans and step vans does not have this option.

> **Figure 18: Smith Newton Battery Pack (Smith)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing battery pack location on Smith Newton truck. Source: Smith [43, 44]

This charging system overview highlights that innovative solutions to charging infrastructure are being pursued in hardware and in the operational field, not just on paper. There are a myriad of challenges that illustrate that buying an electric truck involves significant planning regarding the charging infrastructure. Though because the infrastructure is essentially not established, fleets, OEMs, utilities, suppliers and other companies have many possible opportunities for inventing new business models to monetize services and vehicle designs as complete systems including the infrastructure.

### 8.2. Charging Speeds

Generally speaking, when it comes to charging speed, there are three types of EVSEs: Level 1, Level 2, and DC Fast Charging. 

> **Figure 19: Types of EVSE (NACFE)**
> Type: Table
> Axes: none
> Series: Type of EVSE vs Voltage, Power (kW), Price, Installation Requirements
> Values: 
> - Level 1: Voltage: 120 V, Power: 1.9 kW, Price: Usually included with vehicle purchase (for passenger EVs), Installation Requirements: Most plug-in electric light-duty vehicles come with a cord set capable of plugging into a standard home wall outlet, so no additional charging equipment is required
> - Level 2: Voltage: 208 - 240 V, Power: 7.2 - 19.2 kW, Price: A few thousand dollars per charger, Installation Requirements: Requires installation of charging equipment and a dedicated circuit of 20 to 100 amps
> - DC Fast Charge (sometimes called Level 3): Voltage: Typically 480 V AC input, Power: 72 kW–1 MW (in discussion), Price: $15,000–$90,000 per charger, Installation Requirements: Requires installation of charging equipment and dedicated circuit
> Notes: Source: NACFE

<!-- page: 37 -->
Level 1 chargers are typically used only for light-duty passenger vehicles, allowing them to plug-in to a standard 120-volt home wall outlet to charge. Because of the size of most commercial battery electric truck batteries, a Level 1 charger would take more than a full day to charge the truck. Therefore, this type of charger is not appropriate for charging of commercial fleets. 

Rather, when it comes to EVSE type, fleets will need to decide between Level 2 or DC Fast Chargers (or a mix of both) in order to keep their vehicles charged. 

> **Figure 20: ClipperCreek CS-100 Level 2 Charger and ChargePoint Express Plus Station (ClipperCreek, ChargePoint)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photos of ClipperCreek CS-100 Level 2 Charger and ChargePoint Express Plus Station. Source: [51, 65]

Simply put, Level 2 EVSEs charge vehicles faster than Level 1 chargers but slower than DC fast chargers. Like Level 1 chargers, they typically use alternating current (AC) from the facility, which is then converted to direct current (DC) by the truck’s on-board vehicle charger since EV batteries are DC. AC Level 2 chargers typically have a voltage of 240 volts and cost a few thousand dollars per charger (not including installation or any grid/facility upgrades that may be required). Level 2 chargers offer upwards up 7.2 kW of power, and some now offer over 19 kW.<sup>51 52</sup> Depending on duty cycle, many fleets that employ “return to base” or “depot” charging find Level 2 EVSEs adequate for charging overnight.

According to a joint NACFE/ACT fleet survey conducted for the first guidance report in this series (and shown in Figure 21), over 75% of vehicles in the fleet survey in the range of Class 3 to 8 segments are operated on shift schedules where they are parked for more than six hours per day.

<!-- page: 38 -->
| Parked for (in 24 hour day) | Light Duty Delivery Truck (Class 3) | Medium Duty Box Truck (Class 4-6) | Heavy Duty City Tractor (Class 7/8) | Heavy Duty Regional Tractor (Class 7/8) | Heavy Duty Long Haul (Class 7/8) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| less than 1 hour | 0% | 0% | 0% | 1% | 0% |
| from 1 to 2 hours | 0% | 0% | 2% | 4% | 13% |
| from 3 to 4 hours | 0% | 0% | 18% | 15% | 9% |
| from 5 to 6 hours | 8% | 0% | 12% | 10% | 10% |
| from 7 to 8 hours | 25% | 50% | 3% | 14% | 15% |
| Greater than 9 hours | 67% | 50% | 65% | 56% | 54% |

> **Figure 21: Vehicle Time Parked Per Day by Segment (NACFE/ACT Survey)**
> Type: Table (matrix / bar equivalents in table form)
> Axes: Parked duration vs Vehicle Segment
> Series: Light Duty Delivery Truck, Medium Duty Box Truck, Heavy Duty City Tractor, Heavy Duty Regional Tractor, Heavy Duty Long Haul
> Values: See table above.
> Notes: Source: NACFE/ACT Survey

Therefore, many of these trucks, depending on battery size, may be able to utilize Level 2 charging during their “dwell time” between shifts. 

> **Figure 22: Electric Truck Charging Between Shifts (HDT)**
> Type: Photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photo showing electric truck being charged between shifts. Source: HDT [53]

However, trucks with larger battery packs and/or shorter dwell times may need to consider DC Fast Chargers (DCFCs). Unlike Level 1 and 2 chargers, and as the name implies, DCFCs use direct current and are capable of charging vehicles much more quickly. For example, perhaps the most well known DCFC, Tesla’s Supercharger, delivers “a nearly consistent 72 kilowatts (kW) of power,” even in urban areas, and can deliver up to 120 kW.<sup>54</sup> Meanwhile, the fastest DCFC on the market today (or rather, under UL review), Tritium’s Veefil-PK DC Ultra-Fast Charger, is capable of supplying up to 475 kW using liquid-cooled cables.<sup>55</sup> And, as mentioned earlier, manufacturers across the charging space are already whispering about 1 MW chargers.

<!-- page: 39 -->

Though chargers this fast are also much more expensive. Not including installation or any
grid/facility upgrades that may be required, experts estimate that current DCFC stations can
cost upwards of $35,000.56 57 Level 2 chargers on the market can range from $2,000 to
$7,000.58 59 The cost generally multiplies with the quantity of electric vehicles. And not only is
this capital expense (capex) significantly higher for DCFCs, but operational expenses (opex)
may also be higher since fleets with DCFCs are more likely to run into demand charges from the
utility due to the large amount of power used at any one time. (More on this in Section 9.)
Because of this hefty price tag, fleets tend to invest in DCFCs only as a last resort after Level 2
options have been exhausted.

> **Figure 23 — Tritium Veefil-PK DC Ultra-Fast Charger**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Tritium

Again, deciding which level of charging is right for your fleet depends on how many trucks need
to be charged, the size of their batteries, how long they each have to charge, and electricity rate
structures. For example, the Tesla Model S P100D has a 100 kWh battery, giving it a range of
315 miles.60 For comparison, Chanje’s V8100 medium- duty panel van also has a 100 kWh
battery, allowing for a range of 150 miles. Meanwhile, Freightliner’s eCascadia semi truck
boasts a 550 kWh battery which, according to the company, allows for a range of 250 miles on a
full charge.61

March 1, 2019
Purchaser’s Internal Use Only
39

---

<!-- page: 40 -->

> **Figure 24 — Chanje V8100 Medium Duty Panel Van**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Chanje62

> **Figure 25 — Freightliner eCascadia Class 8 Truck**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Freightliner63

Evaluating, for the example, the Freightliner eCascadia at a Clipper Creek Level 2 CS-100
station, the electric delivery van may be able to recharge its batteries in 4-6 hours using a Level
2 charger, whereas the eCascadia electric Class 8 tractor may require the same amount of time
to recharge using a Tesla Supercharger DCFC.

March 1, 2019
Purchaser’s Internal Use Only
40

---

<!-- page: 41 -->

| Truck | Battery Size | Range | Charge Time with Level 2\* \*\* | | Charging Time with DCFC\* \*\*\* | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | To 80% | To 100% | To 80% | To 100% |
| Chanje V8100 | 100 kWh | 150 miles | 3–4 hours | 4–6 hours | 30–40 minutes | 1–2 hours |
| Freightliner eCascadia | 550 kWh | 250 miles | 17–18 hours | 23–26 hours | 2.5–3.5 hours | 4–6 hours |

*\* Assuming 20% state of charge*
*\*\* Assuming 19.2 kW*
*\*\*\* Assuming 120 kW from charger and that vehicle capable of receiving 120 kW*

> **Figure 26 — Potential Real-world Charging Scenarios**
> Type: table
> Axes: none
> Series: none
> Values: see table above
> Notes: NACFE

The estimates in Figure 26 assume a 20% starting state of charge for the batteries, that the
Level 2 charger delivers 19.2 kW, and that the DCFC delivers 120 kW. It also assumes that both
vehicles are capable of receiving 120 kW.

In practice however, just because a truck is “DC compatible” doesn’t mean that it can receive
the max amount of power that the charger is capable of providing. Equipment, both on- and offboard, will have limitations that will need to be observed. Again, standards for fast charging are
still being developed.

> **Figure 27 — Chanje V8100 Charging**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Chanje

Even within DCFC chargers, infrastructure costs can increase dramatically with the speed of
charging. An example of this is provided by ChargePoint in the overview of their Express Plus
EVSE system, references above. Their modular approach to charging stations uses a Power
Module inserted into a Power Block powering an Express Plus Station, as shown in Figure 28.64
65 Adding additional Power Modules into the Power Block increases the power level achievable
in charging. Each Power Block can accommodate up to four Power Modules, as shown in

March 1, 2019
Purchaser’s Internal Use Only
41

---

<!-- page: 42 -->

Figure 29. The illustration shows that adding four Power Blocks (each with four Power Modules
for a total of 16 modules) achieves 500 kW of energy. Though again, not only does this
significantly increase capex, but it may also increase opex since fleets with higher simultaneous
power needs may be more likely to run into expensive demand charges, depending on the local
utility rate structure. This also highlights the need for fleets and utilities to work together to
develop appropriate rate structures for fast charging situations.

> **Figure 28 — ChargePoint EVSE System Components**
> Type: schematic / illustration
> Axes: none
> Series: none
> Values: none
> Notes: adapted from ChargePoint

> **Figure 29 — Adding Charging Equipment increases capacity**
> Type: schematic and table
> Axes: none
> Series: Example Configurations (A, B, C, D)
> Values:
> - Configuration A - Maximum Power/Port (kW): 125, Simultaneous Power/Port (kW): 62.5
> - Configuration B - Maximum Power/Port (kW): 250, Simultaneous Power/Port (kW): 125
> - Configuration C - Maximum Power/Port (kW): 375, Simultaneous Power/Port (kW): 187
> - Configuration D - Maximum Power/Port (kW): 500, Simultaneous Power/Port (kW): 250
> Notes: adapted from ChargePoint

March 1, 2019
Purchaser’s Internal Use Only
42

---

<!-- page: 43 -->

ChargePoint has also unveiled a concept design for a 2-megawatt electric charging connector
that can be used for high-powered charging of both electric semi-trucks and aircraft.66 The
concept is shown in Figure 30.

> **Figure 30 — ChargePoint High-Powered Connector Concept**
> Type: render / illustration
> Axes: none
> Series: none
> Values: none
> Notes: ChargePoint

Tesla is currently creating a “Megacharger”, estimated to have a power output of approximately
1.6 MW, or 13 times the power level of a standard Supercharger.67 While the final Megacharger
design is not yet public, a temporary prototype – a hub linked to three Supercharger stations –
was recently sighted at the Madonna Inn in San Luis Obispo.68

> **Figure 31 — Tesla Megacharger Prototype at Madonna Inn**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Teslarati, courtesy of jerryswhip/Instagram

March 1, 2019
Purchaser’s Internal Use Only
43

---

<!-- page: 44 -->

Charging time, while greatly reliant on the size of the charger, also depends on the batteries’
level of depletion – their state of charge (SOC). That is, batteries charge much quicker from a
depleted state than when nearly fully charged. The dashed curve in the example in Figure 32
shows how charge capacity changes with charging time. The knee of this curve, the point where
the charging process begins to slow is at about 80% of capacity. In the example graph, 80% of
capacity can be charged in the first hour, while the remaining 20% requires two more hours.
This is typical for lithium-ion vehicle batteries. Therefore, charging at a DCFC station may only
be effective if your battery SOC is below 80%. After that point, charging will slow down
significantly. In fact, batteries will always charge more slowly the closer they get to 100% SOC.

> **Figure 32 — Battery Charging Basics**
> Type: line chart
> Axes: Charge time (hr) (x axis) vs Cell voltage (V), Charge current (%) (left y axes), Charge capacity (%) (right y axis)
> Series: Constant voltage, Charge capacity, Charger float voltage, Charge current, Charge rate = 1C
> Values: approximate values from curve (knee at approx. 1.0 hr / 80% capacity)
> Notes: BatteryUniversity69

This behavior of batteries creates opportunities for smart charging systems to prioritize the order
of charging vehicles and to overlay time of day electricity rate variables to optimize charging
from a fleet site perspective rather than by individual truck. NACFE interviews with fleets and
OEMs highlighted that smart charging capabilities simplify operating charging systems and
improve battery life. These can also create further opportunities to reduce costs. More
information on Battery Charging can be found in Section 9.2 of the second guidance report in
this series, *Medium-Duty Electric Trucks: Cost of Ownership*.70

March 1, 2019
Purchaser’s Internal Use Only
44

---

<!-- page: 45 -->

## 8.3. Charger Communication

So how do chargers know a battery’s SOC? And how do they know how much power it can
accept? This information is all communicated through the EVSE protocol, which is responsible
for much more than just electricity transfer.

Thanks to this two-way communication between the charger and vehicle, the EVSE sets the
correct charging current based on the maximum current the charger can provide as well as the
maximum current the car can receive.71 There’s also a safety feature that will prevent current
from flowing when the charger is not connected to the vehicle or when there is not proper
grounding. And EVSE is also capable of detecting hardware faults and disconnecting the power
in order to prevent battery damage, electrical shorts, or fire.

> **Figure 33 — Example EVSE Charging Cord**
> Type: photograph / diagram
> Axes: none
> Series: none
> Values: none
> Notes: Green Car Reports, labeled with pointers to Power, Safety, Ground, Communication

## 8.4. Charger Software and Networking

As prices for charger components themselves have been dropping as the technology becomes
commoditized, according to the Edison Electric Institute, software is now the main differentiator
between EVSE provider companies.72 Charging software, for example, is key for easily
managing fleet charging operations. For example, software is what allows multiple chargers onsite to be able to communicate with one another and what ensures that a fleet is charging
smartly. That is, algorithm-based tools may be built in that can, without human intervention, help
cycle trucks through charging while lowering electricity costs.

Sometimes, software comes built-in to chargers and is paired with proprietary physical assets.
For example, ABB’s HVC-Overnight Charger offers a compact, single power cabinet paired with
up to three charge boxes so that rows of up to three vehicles each can charge sequentially.73

March 1, 2019
Purchaser’s Internal Use Only
45

---

<!-- page: 46 -->

That is, after the first vehicle has finished charging, the next will start charging automatically,
according to the software’s algorithms, thereby maximizing vehicle availability and reducing the
initial infrastructure and installation investments and subsequent operational costs such as
demand charges.

> **Figure 34 — ABB Smart Charging Solution for Depots**
> Type: illustration
> Axes: none
> Series: none
> Values: none
> Notes: ABB

Software can also be purchased from third party vendors to complement the chargers’ built-in
software and allows automatic sequencing of vehicle charging across the fleet. Some can even
pause or stop charging by vehicle if necessary, almost like pausing a download on Spotify.

For example, electriphi, Inc. provides software for energy management and charging control of
electric vehicle fleets. Their software is based on open industry standards and designed to
interoperate with any charging infrastructure or electric vehicle type. Their platform uses
machine learning algorithms to manage fleet operations and reduce energy costs, using a
multitude of factors such as energy rates, route dynamics, battery characteristics, weather
conditions and more (Figure 35). It also allows operators to manage the transition from
conventional to electric fleets.74

March 1, 2019
Purchaser’s Internal Use Only
46

---

<!-- page: 47 -->

> **Figure 35 — Example Charging Dashboards**
> Type: dashboard screenshot / UI graphic
> Axes: none
> Series: none
> Values: various dashboard metrics (Total Vehicles: 200, Enroute: 80, Ready to dispatch: 27, Charging: 40, In Queue: 30, Stand by: 5, V2G: 11, Dispensed Today: 68 MWh, Daily Average: 93 MWh, 8.7 MW)
> Notes: electriphi, Inc.

Software is also capable of collecting data and providing analytics to help fleet managers make
informed charging decisions. Some can even notify on-call technicians when issues arise. For

March 1, 2019
Purchaser’s Internal Use Only
47

---

<!-- page: 48 -->

example, they may receive a notification through a connected app on their phone that a
charging cable is damaged and needs to be repaired (Figure 36).

> **Figure 36 — Example Mobile Dashboard**
> Type: screenshot / illustration
> Axes: none
> Series: none
> Values: Task metrics shown (Most Urgent Tasks: 2, Pending Tasks: 3, Today's Total Task: 7, Completed: 2, This month Total services: 75)
> Notes: electriphi, Inc.

Additional features can include distributed energy resources (DER) integration, predictive
scheduling, and integrated telematics. Figure 37 outlines some example services offered by
charging station networks.

> **Figure 37 — Example Charging Station Network Services**
> Type: diagram / concept map
> Axes: none
> Series: none
> Values: none
> Notes: Greenlots

March 1, 2019
Purchaser’s Internal Use Only
48

---

<!-- page: 49 -->

Generally speaking, there are three types of charging station networks: non-networked, closed,
and open (Figure 38).75

> **Figure 38 — Charging Station Network Types**
> Type: diagram / illustration
> Axes: none
> Series: Non-networked, Closed networks, Open networks
> Values: none
> Notes: Greenlots

Particularly when fleets are first dipping their toe into electrification and piloting charging
solutions, they may want to opt for open, standards-based networks in case they want to test
multiple chargers but manage them all together on one network or in case they want to switch or
mix and match chargers in the future.

Communication standards known as protocols are what allow for nearly instantaneous and
seamless communication between devices and systems, enabling interoperability. This is no

March 1, 2019
Purchaser’s Internal Use Only
49

---

<!-- page: 50 -->

different for chargers and networks, which use the internationally recognized Open Charge
Point Protocol (OCPP).76

> **Figure 39 — Open Charge Point Protocol (OCPP)**
> Type: diagram
> Axes: none
> Series: none
> Values: none
> Notes: Greenlots

OCPP allows communication between charging stations and central systems, regardless of
vendors, as shown in Figure 39. The most recent version of the Open Charge Point Protocol,
released in 2018, the OCPP 2.0, has extended smart charging functionality and cyber security
compared to previous versions of OCCP. OCPP is the de facto network protocol throughout
Europe and is used in 78 countries on every continent (Figure 40).

> **Figure 40 — Countries Currently Using Open Charge Point Protocol**
> Type: map
> Axes: none
> Series: none
> Values: none
> Notes: Greenlots

## 8.5. Charger Maintenance

Similar to networking, companies may offer very different maintenance packages. These may
include services such as proactive monitoring and repair of equipment if needed. Monitoring is
important in order to spot and address issues before they snowball into crises. And timely repair

March 1, 2019
Purchaser’s Internal Use Only
50

---

<!-- page: 51 -->

of charging equipment is essential for ensuring mission-critical vehicle uptime. For example, if
repairs take too long to schedule and complete, fleets may lose profits and even long-term
contracts due to unexpected downtime. Therefore, maintenance packages should be carefully
reviewed to ensure they meet fleet needs.

## 8.6. Charging Locations

Charging will roll out in stages, first at a fleets’ home depot. Later, fleets may share charging,
where a truck goes from its home depot to someone else’s home depot, both equipped with
chargers. Eventually, remote public charging is expected to emerge on high density freight
corridors where distances demand a mid-trip boost or recharge. Charging will evolve as demand
grows.

### 8.6.1. Depot Charging

Similar to the personal vehicle market, most commercial vehicles currently charge at “home”, or
private, “depot-” or “return-to-base” charging stations. Due to the unpredictable “hub & spoke”
nature of commercial trucking operations (mentioned above in Section 8.6 as akin to a
honeybee’s path).77 most fleets currently adopting electric truck technology will want to place
chargers at a central homebase such as a warehouse, distribution center, headquarters, etc.,
where trucks start from and return to each day. This type of “return-to-base charging” also
makes sense because fleets have full control over site access, charger type, placement, and
timing.

The ramifications of on-site charging for a fleet of vehicles are apparent using an example of
early electric vehicle supply equipment (EVSE) installation shown in Figure 41 and documented
in the 2014 *NREL Field Evaluation of Medium-Duty Plug-in Electric Delivery Trucks*.78 Ten
Clipper Creek CS-100 EVSE Chargers were installed at the PepsiCo Frito Lay North America
depot at Federal Way, Washington.

> **Figure 41 — Ten Charging Stations and Facility Power Supply**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: NREL

March 1, 2019
Purchaser’s Internal Use Only
51

---

<!-- page: 52 -->

The footprint of the power supply and the individual charging stations is fairly minimal. However,
the operational implications are that the vehicles must be co-located with the chargers for some
extended period of time to allow charging. This implies that specific parking spots or warehouse
docks need to be dedicated to a vehicle for the time it takes to charge.

Figure 42 illustrates how a dedicated parking area was reserved for vehicle charging at the
Federal Way facility. Interviews conducted by NACFE with other fleets operating battery electric
vehicles highlighted that in some cases the vehicle is charged at the loading dock, meaning that
a dock is dedicated to that specific truck for perhaps one or two shifts, depending on how fast
the company chooses to charge their vehicles.

> **Figure 42 — Aerial View of Federal Way FLNA Distribution Center**
> Type: aerial photograph / map
> Axes: none
> Series: none
> Values: none
> Notes: Base map: Google Earth

Some sites, where vehicles are currently parked extremely close to one another, may need to
be redesigned from a physical space and/or temporal management standpoint in order to allow
for adequate space for charging.

March 1, 2019
Purchaser’s Internal Use Only
52

---

<!-- page: 53 -->

In order to avoid costly conduit and trenching expenses, fleets will want to be sure to place
chargers as close to the transformer and load panel as possible. And because charging stations
are a significant capital expense, fleets will have an incentive to maximize their use. For
example, a cooperative agreement may emerge between fleets that operate on different shift
schedules whereby chargers achieve max utilization.

### 8.6.2. Supplemental Chargers on Dedicated Routes

Ideally, charging vehicles at the fleet’s base during dwell times between shifts will be sufficient
to support operations. However, this may not be sufficient for vehicles with larger battery packs
and/or longer routes. One potential solution, at least for dedicated regional routes, might be to
install charging stations not only at the fleet depot, but also at the customer’s site(s). This could
allow vehicles with relatively long A-B-A routes to charge at point B while unloading, giving them
enough of a charge to make it back to their home base for further charging between shifts.

However, because of the significant capital expenditure for charging infrastructure, fleets
considering this option would need to carefully negotiate the terms with their customer, from
whom they would likely require a long-term contract. An example of a route that includes
charging along the way is shown in Figure 43. As depicted in the figure, charging along the
route can allow for a smaller battery on-board the truck, which can equate to a lower capital
expenditure (for the vehicle) and a higher payload.

March 1, 2019
Purchaser’s Internal Use Only
53

---

<!-- page: 54 -->

> **Figure 43 — Example Charging Pattern with On-Route Charging**
> Type: diagram / illustration
> Axes: none
> Series: none
> Values: Application segment: 110 kWh, 200 km range, 0.46 kWh per km; Specific use case: 60 kWh, 120 km range, 0.42 kWh per km
> Notes: McKinsey & Company79, ¹Charging at hub with 22 kilowatts (kW) and on the route with 50 kW.

## 8.6.3. Opportunity and Public Charging

In addition to depot charging, fleets may also consider “opportunity charging” on the road. For
example, vehicles may take advantage of the quickly developing public charging network if
needed for range extension or in emergencies.

According to the Alternative Fuels Data Center (AFDC), as of January 2019, the U.S. alone has
over 67,230 electric charging station outlets, not including residential chargers80 81

March 1, 2019
Purchaser’s Internal Use Only
54

---

<!-- page: 55 -->

> **Figure 44 — U.S. Alternative Fueling Stations by Fuel Type**
> Type: bar chart (stacked)
> Axes: Year (1992 to 2016) vs Number of Stations (0 to 80,000)
> Series: Electric, Propane, Methanol (M85), LNG, Hydrogen, Biodiesel, CNG, E85
> Values: approximate values read from chart
> Notes: AFDC. Last updated: September 2018. Printed on: January 8

Of those electric charging stations, over 60,530 are public stations.82 However, these public
stations are not spread evenly across the country. Rather, there is a large disparity in where
these public charging stations are located. For example, over 19,000 outlets are in California,
whereas Texas, the state with the next highest number, has less than a fifth that number, with
just over 3,100 public station outlets.

> **Figure 45 — Public Electric Charging Stations by State**
> Type: map (choropleth)
> Axes: none (scale 0 to 19,061)
> Series: none
> Values: California over 19,000; Texas over 3,100
> Notes: AFDC

Luckily, many options exist to help fleets and drivers find local charging stations. For example,
ChargeHub offers an interactive map feature that lets you search for a charging station near
you, not only by location, but also by level (1, 2, DCFC, etc.), hours of operation (24/7),
networks (eVgo, ChargePoint, Tesla, Electrify America, etc.), connector type (CHAdeMO,
J1772, CCS1, 2, Tesla, etc.), and minimum power output. Some stations will even tell you if
they have an available port or if the charger is busy.83

March 1, 2019
Purchaser’s Internal Use Only
55

---

<!-- page: 56 -->

> **Figure 46 — Charging Stations Map**
> Type: screenshot / map GUI
> Axes: none
> Series: none
> Values: none
> Notes: ChargeHub

DOE’s Alternative Fuels Data Center also has its own station locator that lets you filter by
network, as do most of the charging networks themselves. And Electrify America, the company
created by Volkswagen to meet the $2 billion piece of their settlement requiring them to create a
National Zero Emission Vehicle (ZEV) Investment Plan and build out ZEV infrastructure, is
currently installing a public network of hundreds of DCFCs capable of charging vehicles with
either CHAdeMO or CCS ports at speeds of up to 350 kW.84

> **Figure 47 — An Electrify America Charger with both CCS (green) and CHAdeMO (blue) Connectors**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Electrify America

March 1, 2019
Purchaser’s Internal Use Only
56

---

<!-- page: 57 -->

Electrify America also offers an online map to help drivers find charging stations in their network
across the U.S.85

> **Figure 48 — Electrify America Charger Map**
> Type: map with regional counts
> Axes: none
> Series: none
> Values: Regional counts shown on map (e.g., 13, 5, 5, 3, 4, 16, 4, 7, 3, 4, 9, 6, 12, 14, 13, 12, 6, 6, 10, 6, 6)
> Notes: Electrify America

Though because of the costs of using public chargers and the uncertainty of availability, for the
time being, vehicles may only want to rely on public charging in case of emergency. Though
knowing that this option exists should relieve some of the “range anxiety” that fleet managers
and drivers may feel about potentially running out of power while away from their homebase.
And as charging networks are built out across the country, fleets may advocate for public
charging hubs with significant numbers of fast chargers specifically for commercial freight - a
trend happening in Europe and Asia.

Public chargers may eventually also offer a solution for electric long-haul sleeper cabs like the
Tesla Semi. While a public network capable of charging these large battery packs in the 30
minute timeframe quoted in the truck’s marketing materials does not yet exist, Tesla has plans
to build out a network of solar-powered “Megachargers” (described earlier in this report), similar
to their Supercharger network.86 However, the company understandably is focused first and
foremost on working with its biggest Semi reservation holders – the likes of UPS, PepsiCo, and
Anheuser-Busch, who have 265 Tesla Semi trucks on order between them – to install
Megacharging stations at their facilities.87

# 9. Charging System Overview

Though charging infrastructure goes far beyond the charger itself, and in order to make smart
charging decisions, it is important to understand the broader charging system. In fact, as
explained in the first two reports in this series, for simplicity, NACFE defines the “charging

March 1, 2019
Purchaser’s Internal Use Only
57

---

<!-- page: 58 -->

system” as starting at the point where the electricity enters the national electrical grid, labeled
“generation” in Figure 49. Generators can be coal- or natural gas-fired power plants, solar
gardens, wind farms, or even hydroelectric dams. The grid consists of all of the infrastructure
from the generating station to the end user’s facility, usually defined as the utility meter.

> **Figure 49 — Electricity Grid Components**
> Type: diagram / schematic
> Axes: none
> Series: Generation, Transmission, Distribution
> Values: none
> Notes: Rocky Mountain Institute88

The meter tracks the energy consumed for billing purposes. Energy is the combined factor of
power used, expressed typically in kilowatts, multiplied by time of use, typically hours – hence
the unit of electrical energy is kWh (kilowatt-hours).

Energy (kWh) = Power (kW) x Time (h)

> **Figure 50 — Power & Energy References**
> Type: table / image grid
> Axes: none
> Series: Generation (Hoover Dam, Panda Power Plant, Tractor Solar Panel), Consumption (US Household 2017, Tesla Model S, ENERGY STAR Certified Refrigerator)
> Values: Hoover Dam: Production Capacity = 2,080,000 kW, Average Annual Energy Output = 4,000,000,000 kWh; Panda Power Plant: Production Capacity = 100,000 kW, Average Annual Energy Output = Unknown; Tractor Solar Panel: Production Capacity = 0.3 kW, Average Annual Energy Output = 452 kWh; US Household 2017: Average Annual Electricity Use = 10,399 kWh; Tesla Model S: Average Annual Electricity Use = 4,447 kWh; ENERGY STAR Certified Refrigerator: Average Annual Electricity Use = 615 kWh
> Notes: NACFE89 90 91 92 93 94 95 96 97 98

March 1, 2019
Purchaser’s Internal Use Only
58

---

<!-- page: 59 -->

Some power and energy references are included in Figure 50. For example, as you may know
from the first guidance report in this series, *Electric Trucks - Where They Make Sense*, the
Hoover Dam Powerplant has a nameplate capacity (or maximum production at any one time) of
about 2,080 megawatts. Multiply this by the 8,760 hours in a year, and you would expect an
energy output of 18,220,800 megawatt-hours (MWh) or over 18 billion kilowatt-hours (kWh).
However, because the turbines are not always operating, the plant actually generates, on
average, about 4 billion kWh of hydroelectric power each year.99

Similarly, an individual solar panel on a tractor may be rated as 300 watts, but this is the
maximum amount of power that the panel can produce in ideal conditions (full, direct sun; cool
temperatures; etc.). Actual energy production varies dramatically depending on the amount of
sunlight, the angle of the sun, the outside temperature, etc. And of course, at night when the
sun isn’t shining, the panel isn’t producing any power at all. Therefore, rather than having an
annual energy output of 2,628 kWh (300 watts x 24 hrs/day x 365 days/yr), an average panel
operating near Phoenix, AZ would likely only actually produce about 452 kWh each year.100

This difference between energy output and capacity is important to keep in mind, especially
when considering renewable generators, whose output tends to be much more intermittent or
variable (though predictable) in nature. After all, the sun only shines during the day, and the
wind doesn’t always blow. We will discuss this challenge – as well as how electric vehicles may
help solve it – in more detail in Section 13.5.

Energy consumption is also measured in power over time. For example, your home utility power
bill charges you for kWh energy used. The average household annual electricity use in the U.S.
is 10,399 kWh. Of that, a refrigerator alone consumes about 615 kWh on average per year.

Simply speaking, the amount of energy an electric vehicle needs to charge depends on the size
of the battery. Therefore, it is fairly straightforward to understand that in order to fully charge the
eCascadia, which has a 550-kWh battery, approximately 550 kWh of electricity must be
generated, transmitted across the grid (plus some margin to deal with system inefficiencies),
distributed to the charging site, and finally, delivered to the vehicle through the charger itself. In
order to complete this last step, the electricity must be moved through the on-site infrastructure
from the meter to the vehicle as shown in Figure 51.101 If the facility does not have an adequate
power supply from the utility, additional grid work may be needed to run additional lines to the
site and install transformers. Other “behind the meter” solutions also exist such as installing onsite battery storage, on-site solar or wind generation, on-site generators run by natural gas, or
others and even combinations of these. The nature of electricity is that it can be produced,
stored and delivered in many ways; it’s not limited to one solution.

March 1, 2019
Purchaser’s Internal Use Only
59

---

<!-- page: 60 -->

> **Figure 51 — Vehicle Charging Infrastructure**
> Type: schematic / diagram
> Axes: none
> Series: Service Connection, Supply Infrastructure, Charger Equipment
> Values: none
> Notes: EEI

What is clear, as far as the overall charging system, is that electric trucks will increase demand
on electricity. Because of this, grid capacity will need to be improved. New generation may need
to be added if increased efficiency in other sectors (buildings, industry, etc.) is not enough to
counterbalance the new load from the quickly electrifying transportation sector. Utilities may
also need to develop new demand management and/or storage solutions to help balance timing
concerns with electricity supply and demand. Similarly, new tariff structures may be necessary
in order to encourage “smart” charging which prioritizes times when electricity supply is
available and economical. Systems that help users prioritize around other factors – including the
carbon footprint of the grid at any one time – are now available.

Generally speaking, utility tariffs are the pricing structures that energy providers charge
customers for energy use. There are many different types of utilities – from investor-owned to
municipal utilities to cooperatives (see Figure 52) – though 68% of electricity customers are
served by investor-owned utilities (IOUs). In exchange for the right to sell electricity in a given
service territory, these IOUs are guaranteed profits or an “allowed rate of return” by regulators
such as Public Utilities Commissions (PUCs). This rate of return is used to calculate the utility’s
“revenue requirement” through the traditional “cost of service” model.

March 1, 2019
Purchaser’s Internal Use Only
60

---

<!-- page: 61 -->

> **Figure 52 — Utility Types**
> Type: diagram / flowchart
> Axes: none
> Series: Investor-Owned Utilities (IOUs), Municipal Utilities ("Munis"), Cooperatives ("Co-Ops")
> Values: IOUs: Only 6% of utilities operating in the US are investor-owned, but 68% of electricity customers are served by IOUs; Munis: 60% of utilities operating in the US are municipally owned, but only 15% of electricity customers are served by munis; Co-ops: 26% of utilities operating in the US are cooperatives, 13% of electricity customers are served by co-ops
> Notes: Rocky Mountain Institute102

This traditional “cost-of-service” model (Figure 53) ensures that fees from customers or
“ratepayers” cover not only the costs of infrastructure and fuel, but also the utility’s regulatorapproved “allowed rate of return.” It is important to note that the allowed rate of return applies to
a company’s assets (e.g. power plants, transmission lines, etc.), whereas recurring operating
expenses such as fuel are simply passed through to customers. Because of this historical
dynamic in which utilities earn a profit based on assets rather than fuel use, utilities may
experience a phenomenon known as the “Averch-Johnson effect”, in which they are incentivized
to make investments in order to increase their “rate base” (i.e. the value of the company’s
assets minus accumulated depreciation) and therefore, their profits.103 (See Section 13.8 for
information on utility business model reform and how this dynamic is changing.)

March 1, 2019
Purchaser’s Internal Use Only
61

---

<!-- page: 62 -->

> **Figure 53 — Utility Cost of Service Model**
> Type: diagram / flowchart and equation
> Axes: none
> Series: none
> Values: RR = O + T + D + r*(RB)
> Notes: John Chowdhury104

Once one understands this dynamic, it becomes clear why utilities would be in favor of
transportation electrification, a trend which will likely require them to invest in new assets and
therefore earn more profits. As the Edison Electric Institute (EEI) puts it, “against the backdrop
of slowing growth in the electric power industry, bringing electricity to the transportation sector is
a huge, albeit long-term opportunity for load growth.”105

However, utilities also want to ensure that this new demand is met is met in a smart way which
does not put further strain on the electrical grid. After all, the grid is built to meet “peak” demand
of electricity in order to avoid blackouts. And because of the nature of the bureaucracy that
controls a utility’s ability to make new investments, building new assets can be a very long
process, often taking years from proposal or “rate case” to approval and “steel in the ground.”
Therefore, given constraints of the current grid, utilities would prefer that electric vehicles not
charge during “peak” times when electricity demand is highest, typically in the late afternoon or
early evening when people return home from work and begin doing energy-intensive chores
(e.g. cooking, washing clothes, dishes, etc.). However, peak and off-peak times can vary
significantly by region due to regional power generation and use differences.

March 1, 2019
Purchaser’s Internal Use Only
62

---

<!-- page: 63 -->

Rather, utilities are interested in encouraging charging (and other energy-intensive tasks) during
“off-peak” hours when the grid has more excess capacity. In order to incentivize customers to
charge their vehicles at low demand times, utilities may offer them cheaper rates for power
during those times. This type of tariff is called a “time-of-use” (TOU) rate. TOU rates can be split
between “peak” and “off-peak” times, or sometimes, between three time-of-use zones: “peak”,
“shoulder” and “off-peak”.

> **Figure 54 — Example Time of Use Pricing**
> Type: pie chart / circular diagram
> Axes: none
> Series: off-peak (green), shoulder (yellow), on-peak (red)
> Values: none
> Notes: Xcel Energy. \*Not included on weekends.

In Figure 54 above, for example, the green area is considered off-peak, while the red area is onpeak and the yellow area is the shoulder period.106 Though even in this example, focused on
residential customers (fleets would substitute “charge your vehicles” where it says “run your
major appliances”), the charges vary by day (weekday vs. weekend vs. holidays) and season
(summer vs. winter).

Time of use rates can be a challenging new concept, especially for fleet managers who
historically haven’t had to take temporal differences in fuel pricing into account. As Edmunds
writes, “What if a gallon of gasoline cost $3 at breakfast time, was free at lunch, dinged you $8
in the afternoon, but was only $2 in the middle of the night? Welcome to the world of charging
up plug-in electric vehicles.” 107

And unlike light-duty EVs, where TOU pricing has been effective at shifting charging to off-peak
hours, electric truck charging cannot easily be shifted due to the nature of the trucking business.
That is, trucking fleets’ main objective is providing timely and reliable service to their customers.
Therefore, trucks generally operate on set schedules and need to ensure their vehicles are
charged and ready to go to support these operations. As a result, truck fleets do not have the

March 1, 2019
Purchaser’s Internal Use Only
63

---

<!-- page: 64 -->

same flexibility to shift charging based on utility price signals. As one interviewee explained,
trying to do so would be akin to “the tail wagging the dog”.

Similarly, utilities may also employ a tool known as “demand charges”. Demand charges allow
utilities to charge customers based on their individual peak demand or highest use in a given
timeframe and are designed to cover utilities' fixed costs of providing a certain level of energy to
their customers (whereas energy costs cover the variable costs that are dependent on kWh
used).108 Because utilities need to build and maintain a lot of expensive equipment, such as
generating stations, substations, wires, and transformers, in order to meet peak demand,
demand charges help cover these costs without relying on all ratepayers to subsidize the larger
energy users. Demand charges also encourage customers to reduce power usage during peak
hours and to shift their usage from peak to non-peak hours. Though different from TOU rates,
which can apply to customers of any size or type (e.g. residential), demand charges typically
only apply to larger commercial and industrial customers.

Demand charges vary from utility to utility (and sometimes by season as well), but the key
takeaway is that if energy use is not kept in check, demand charges can really add up, even to
the point where they are more expensive than the energy charges themselves. For this reason,
demand charges are widely considered to be one of the most prohibitive barriers to entering the
commercial vehicle electrification market, which, by its nature, requires large amounts of power
all at once in order to charge vehicles. In fact, according to a CALSTART report on *Electric
Truck & Bus Grid Integration*, demand charges may make charging medium- and heavy-duty
vehicles economically unfeasible.109 See Figure 55 below for an example of how demand
charges can impact fuel cost for electric buses.

> **Figure 55 — Impact of Peak Demand Charges on E-Buses**
> Type: bar chart (stacked)
> Axes: Vehicle type / Charging scenario (Diesel, CNG, Electric On-route (1 bus), Electric On-route (4 bus), Electric On-route (6 bus), Electric Overnight (8 bus)) vs Fuel Cost ($ per mile) ($0.00 to $1.20)
> Series: Demand Charge, Energy Charge
> Values:
> - Diesel: Energy Charge $1.00, Total approx. $1.00
> - CNG: Energy Charge $0.57, Total approx. $0.57
> - Electric On-route (1 bus): Demand Charge $0.65, Energy Charge $0.25, Total approx. $0.90
> - Electric On-route (4 bus): Demand Charge $0.17, Energy Charge $0.25, Total approx. $0.42
> - Electric On-route (6 bus): Demand Charge $0.08, Energy Charge $0.25, Total approx. $0.33
> - Electric On-route (8 bus): Demand Charge $0.04, Energy Charge $0.25, Total approx. $0.29
> - Electric Overnight (8 bus): Demand Charge $0.00, Energy Charge $0.24, Total approx. $0.24
> Notes: CALSTART. With demand charges @ $20/kW.

March 1, 2019
Purchaser’s Internal Use Only
64

---

<!-- page: 65 -->

However, as we mentioned earlier, utilities are incentivized to find a way to meet this growing
demand and to make it worth fleets’ while to invest in electrification. Because of this desire,
combined with state-level greenhouse gas reduction goals and mandates, some utilities are
starting to rethink their tariff structures and even designing new tariffs specifically to support EV
charging for commercial and industrial customers.

For example, after California passed Senate Bill (SB) 350, the Clean Energy and Pollution
Reduction Act of 2015, which, among other things, directed the state’s PUC to order utilities to
make investments to accelerate transportation electrification, utilities began doing just that. 110
Southern California Edison (SCE), for example, proposed new EV tariffs that would suspend
monthly demand charges during a five-year introductory period and recover more costs through
energy charges, and then phase in demand charges for a five-year intermediate period. As the
demand charges increase, the energy charges will decrease. During this intermediate period,
the demand charges would collect an increasing share of distribution capacity-related costs,
while the remaining distribution capacity costs would be collected via TOU energy charges. At
that point, SCE claims that the demand charges will “still be lower than what new EV customers
would pay on their otherwise applicable (non-EV) commercial rates today.”111 For an example of
the EV tariffs offered by utilities, see Figure 56 below.

| Utility | SCE | SCE | SCE | Georgia Power |
| :--- | :--- | :--- | :--- | :--- |
| Rate Schedule | TOU-EV-3 | TOU-EV-4 | TOU-8 Option A | ET-15 |
| Maximum Demand | <20kW | >20kW <500kW | >500kW | N/A |
| EV Submetering | Required | Required | N/A | Yes |
| Energy Charge | Max. $0.36/kWh Min. $0.06/kWh | Max. $0.29/kWh Min. $0.06/kWh | Max. $0.39/kWh Min. $0.06/kWh | Max. $0.08/kWh Min. $0.00/kWh |
| Demand Charge | A - $0.00/kW B - $7.23/kW | $13.20/kW | $15.57/kW | $0.00/kW |
| Notes | No EV demand charges for Option B if EV account demand does not exceed General Service account demand of associated facility. | No EV demand charges if EV account demand does not exceed General Service account demand of associated facility. | For cold ironing pollution mitigation programs (vessels hoteling at the Port of Long Beach and the Port of Hueneme, and long-haul trucks hotelling at truck stops). | For the operation of electric transportation at 3Ø, 60 Hz and 19.8kV or higher. |

> **Figure 56 — Examples of Utility Rates Designed for Electric Trucks**
> Type: table
> Axes: none
> Series: none
> Values: see table above
> Notes: CALSTART

March 1, 2019
Purchaser’s Internal Use Only
65

---

<!-- page: 66 -->

These new rates are attractive to customers because they allow fleets to test vehicles in real
world operations that may not have been economically feasible with demand charges. They also
give customers time to scale up. This is important because demand charges can be very painful
early on when fleets have only a few electric vehicles that need charging. But once fleets
integrate more EVs into their operations, they can essentially distribute the demand charges
over their total energy use, thereby impacting the cost to charge each vehicle only
incrementally.

SCE also worked with private, public, and governmental organizations to develop a Charge
Ready Program to help deploy and serve EV charging infrastructure for customers.112 While the
Charge Ready Program was targeted at supporting charging for electric passenger vehicles, the
utility expanded their transportation electrification (TE) offerings to also support medium- and
heavy-duty vehicle charging throughout their service territory.113 114 The program helps
customers design a site plan for charging and then works with them to install the charging
system on a new, dedicated circuit with its own panel, meter, and service, separate from any
existing electrical service infrastructure on site.

> **Figure 57 — New Charge Ready Service Installed Separate from Existing Service**
> Type: diagram / schematic
> Axes: none
> Series: none
> Values: none
> Notes: SCE

The program covers the cost of the new electric infrastructure, including any transformers,
trenching, conduit, conductors, or stub-outs.

March 1, 2019
Purchaser’s Internal Use Only
66

---

<!-- page: 67 -->

> **Figure 58 — Infrastructure Covered by the Charge Ready Program**
> Type: diagram / schematic
> Axes: none
> Series: none
> Values: none
> Notes: SCE

The Charge Ready Program also offers rebates to help reduce the cost of the charger
equipment and installation, though vendors and charging stations must be selected from the
approved list and chargers must be either Level 1 or 2. (SCE is also offering a DCFC pilot
program available to customers who provide parking to the public.) Participants must also agree
to select an eligible TOU rate for EV charging.

March 1, 2019
Purchaser’s Internal Use Only
67

---

<!-- page: 68 -->

> **Figure 59 — Charge Ready Program How It Works**
> Type: flowchart / illustration
> Axes: none
> Series: You Submit the Required Forms, We Evaluate Your Site, You Confirm Participation, Together We Design the Site Plan, Construction Begins, We Verify Installation, You Get the Rebate
> Values: none
> Notes: SCE

A stipulation of the Charge Ready Program is that participants must allow collection of usage
data from the chargers. They must also agree to participate in any future demand response
programs designed for the Charge Ready Program. SCE is particularly interested in supporting
customers located in disadvantaged communities (DACs), which are defined by the California
Communities Environmental Health Screening Tool (CalEnviroScreen) according to geographic,
socio-economic, public health, and environmental criteria.115

March 1, 2019
Purchaser’s Internal Use Only
68

---

<!-- page: 69 -->

> **Figure 60 — Map of Disadvantaged Communities**
> Type: map
> Axes: none
> Series: SB 535 Disadvantaged Communities
> Values: none
> Notes: California Environmental Protection Agency116

Because many warehouses, distribution facilities, manufacturing sites, and goods movement
corridors tend to be located within or adjacent to disadvantaged communities, targeting these
sites and removing the pollution from the gasoline- and diesel-powered vehicles serving these
sites will primarily benefit the local communities.117 This synergy – that electrifying medium- and
heavy-duty transport not only helps reduce greenhouse gas emissions, but also supports
environmental justice in the most disadvantaged communities – is especially important when it
comes to ensuring electrification and the rate structure to support it is equitable.

And SCE is certainly not the only utility working to create programs and rate plans that are
supportive of and compatible with electric vehicle charging. For example, Pacific Gas and
Electric (PG&E) recently launched their new EV Fleet Program, through which it plans to spend
$236 million over the next five years to support 6,500 new EVs at over 700 sites.118 Through the
program, PG&E will cover the costs of “make-ready” infrastructure, such as design, permitting,
and construction from power pole to charger. They are also planning to offer additional
incentives for disadvantaged communities, school buses, and transit buses, such as a rebate on

March 1, 2019
Purchaser’s Internal Use Only
69

---

<!-- page: 70 -->

charger equipment and installation costs. Like SCE’s program, all EV Fleet Program participants
will be responsible for maintenance of the vehicles and chargers for at least 10 years.

> **Figure 61 — PG&E’s EV Fleet Program Structure**
> Type: diagram and table
> Axes: none
> Series: Disadvantaged Communities, Public Transit, and School Buses vs Everywhere else
> Values: Disadvantaged Communities, Public Transit, and School Buses: Maximum 50% rebate per charger, PG&E pays for make-ready infrastructure as well; Everywhere else: No rebate for chargers, PG&E pays for make-ready infrastructure ONLY
> Notes: PG&E

PG&E also recently submitted a proposal to the California PUC that would allow customers to
choose the amount of power they need for their charging stations and pay for it with a flat
monthly fee — similar to picking a cell phone data plan.119 This innovative subscription pricing
would replace demand charges by creating a new rate class for commercial EV charging and
would offer two types of rates within that: one for customers with charging needs up to 100
kilowatts and one for customers with charging over 100 kilowatts. “Charging an electric vehicle
is different than powering a building. EV charging will be simpler, more affordable and more
consistent under this proposed plan," said PG&E’s senior vice president for Energy Supply and
Policy, Steve Malnight.120

March 1, 2019
Purchaser’s Internal Use Only
70

---

<!-- page: 71 -->

> **Figure 62 — PG&E’s Proposed Commercial EV Rate Structure**
> Type: infographic / diagram
> Axes: none
> Series: 1) Customers choose subscription level, 2) Subscription remains consistent, 3) Energy usage is billed based on time-of-day pricing
> Values: Subscription Charge: $184 / 50 kW connected charging; Energy Charge: Midnight to 9am Off-peak 11¢/kWh, 9am to 2pm Superoffpeak 9¢/kWh, 2pm to 10pm Peak 30¢/kWh
> Notes: Greentech Media

PG&E’s modeling shows that it could save a medium-duty fleet over 40% on what it is paying
today for energy.

> **Figure 63 — Estimated Bill Savings with PG&E’s Proposed New EV Rates**
> Type: bar chart
> Axes: Category (DCFC, Workplace, Multifamily, Transit, MediumDuty) vs $/kWh (0.00 to 0.40) and $/gal equiv ($0.00 to $4.00)
> Series: C&I Rate (2017 GRC Phase 2), CEV Rate (proposed), Gas/Diesel
> Values: approximate values from chart
> Notes: Greentech Media. Rate and billing estimates are preliminary and only reflect the sample site modeled.

March 1, 2019
Purchaser’s Internal Use Only
71

---

<!-- page: 72 -->

California’s other major utility, San Diego Gas & Electric, has also been working with fleets to
help support charging of electric vehicles by building on the success of its Power Your Drive
Program.121 The utility took on new transportation electrification projects in 2018, including
adding charging stations at the Port of San Diego, San Diego International Airport, delivery fleet
hubs and shuttle hubs to support electric vehicles, including semi-trucks, forklifts, and other
medium/heavy-duty equipment. According to its website, it plans to install charging stations for
approximately 90 fleet delivery vehicles at multiple locations.122

And these programs aren’t being developed in a vacuum. Rather, PG&E, SCE, and San Diego
Gas & Electric have been working together to develop a new “Modern Rate Architecture”
framework for designing rates. Their Modern Rate Architecture is based on four key principles:
transparency, equity, sustainability, and access.123

> **Figure 64 — Key Principles Informing Modern Rate Architecture**
> Type: table / diagram
> Axes: none
> Series: Transparency, Equity, Sustainability, Access
> Values: none
> Notes: Public Utilities Fortnightly

While California is clearly the leader when it comes to supporting charging for electric trucks,
other utilities throughout the country are also stepping up to the challenge. For example,
Michigan recently approved its first utility EV charging infrastructure pilot.124 Consumers
Energy’s PowerMiDrive initiative, a $10 million, three-year plan to incentivize the deployment of
EV chargers through rebates of up to $70,000 and consumer education campaigns, was
approved by the Michigan Public Service Commission. The pilot also incorporates TOU rates to
encourage off-peak charging.125

As evidenced from the examples above, utilities are beginning to rethink their tariff structures
and program offerings in order to better support EV charging. However, change can be slow,
particularly in such a heavily regulated industry. For example, individual utilities must file a “rate
case” in order to present new tariff structures for approval to the regulating agency (e.g. PUC), a
process that can take years. But luckily, even in today’s context, there are many business
models out there that can make charging feasible for most any fleet.

March 1, 2019
Purchaser’s Internal Use Only
72

---

<!-- page: 73 -->

# 10. Charging Business Models

## 10.1. Infrastructure Business Models

As the market for electric vehicle charging develops, many new and innovative business models
are popping up on both the infrastructure and electricity sides. Based on research and
interviews, NACFE has determined that there are currently two main models for procuring
charging stations and associated infrastructure. The most common is by buying the stations
outright, often through an RFP process. In this scenario, fleets may hire a consultant to help
make these decisions and set up the infrastructure (and potentially also help manage the
relationship with the utility), but in the end, the fleet owns and manages the chargers, which are
then considered a capital expense.

However, leasing options are also available through some charging station suppliers. In this
scenario, the supplier owns the stations and the fleet simply pays a fee for using them. This
model allows the fleet to pay for the stations out of their operational expense budget.

In both the lease and own options, fleets often pay charging suppliers not just for the physical
stations but also for access to their fleet management networks, which again, are a recurring
operational expense.

Business models may vary for procurement of additional charging infrastructure and even
electricity as well. As discussed in *Electric Trucks – Where They Make Sense*, the traditional
breakdown of electricity distribution to support vehicle charging would have the utility
responsibility end at the meter. The site owner would be responsible from the meter up to and
including the charging station. However, the evolving nature of the electric vehicle marketplace
is creating opportunities to redefine the traditional business models for power distribution. For
example, Kellen Schefter of the Edison Electric Institute presented three potential alternatives in
a 2017 presentation shown in Figure 65, redefining where the utilities may step in to provide
portions or the entire electric vehicle charging infrastructure.126

> **Figure 65 — Charging Infrastructure Alternatives**
> Type: diagram / table
> Axes: none
> Series: Business As Usual, "Make Ready", Charger Only, Full Ownership
> Values: none
> Notes: EEI

March 1, 2019
Purchaser’s Internal Use Only
73

---

<!-- page: 74 -->

For example, SCE’s Charge Ready Program mentioned earlier is an example of the “make
ready” model.

This also suggests more innovative business arrangements may be possible, including third
parties that step in with capital to create turn key systems, with various usage rates that could
remove the site owner from the complexity of managing part or all of the charging system.

Those third parties, similar to an energy service provider (ESP) in the buildings sector, may
specialize, not just in infrastructure procurement and installation, but also in optimizing charging,
which can have large financial implications. Especially for fleets with little experience or interest
in optimizing charging, this sort of “charging as a service” model can be a good option since
these third party companies specialize in this area and therefore may be better able to maximize
efficiency and avoid load spikes and demand charges, which, according to Amply, one of the
companies now offering this service, can be three times more for unrestrained charging
compared to controlled, optimized charging. In other words, it pays to know what you’re doing.
Amply’s model, for example, has them cover the bulk of up-front capital expenditures associated
with charging and then pass these costs on to the customer through a “per-electric-mile-driven”
rate, inclusive of all capex, opex, energy, demand, insurance, and taxes, that they offer
customers over a five- or 10-year contract.127 This business model (Figure 66), which was
deployed at the Port of Long Beach, is unique in that it gives fleet managers, who may be
worried about surprises on monthly bills, certainty on electric fuel costs with uptime
guarantees.128

> **Figure 66 — Amply’s EV Fleet Charging Model**
> Type: table / comparison diagram
> Axes: none
> Series: Amply's Role vs Fleet Customer's Role
> Values: none
> Notes: Amply

March 1, 2019
Purchaser’s Internal Use Only
74

---

<!-- page: 75 -->

Third parties might also include the vehicle manufacturers themselves. As OEMs work to
address the fueling challenges of their customers, vehicle purchase or lease arrangements for
new electric vehicles may include agreement to install and manage charging systems for their
customers. For example, as mentioned in Section 8.1.1 above, Tesla is working directly with its
Semi clients to plan for Megacharger stations on site, charged via solar power through Tesla’s
Powerpack product.129 And Chanje is developing an “energy as a service” offering, which it
describes as “a cost-effective, turnkey fleet depot model that will provide a fully integrated
electric vehicle infrastructure solution.”130 The modular platform, they say, will address charging
infrastructure, renewable energy, storage, and grid services. And Volkswagen recently
announced that it is setting up a new unit, Elli Group GmbH, to offer renewable power and
charging systems to households.131 As the company’s new website explains, Elli will “offer a
seamless and holistic energy and charging experience for electric car drivers and fleet
managers.”132 Elli intends to fill a gap when it comes to companies that are conversant across
these “‘energy meets mobility’ issues”. It will also offer infrastructure for offices and shopping
centers and is planned to launch simultaneously with VW’s I.D. electric-car line in 2020.

> **Figure 67 — Chanje’s Energy as a Service Model will Address Charging Infrastructure, Renewable Energy, Storage, And Grid Services**
> Type: diagram / circle graphic
> Axes: none
> Series: none
> Values: none
> Notes: Chanje

## 10.2. Electricity Business Models

As described above in the Charging System section, most fleets procure electricity the
traditional way – through the local utility’s electric grid. Depending on whether the region is a
regulated or deregulated electricity market, fleets may have options with respect to which
company they buy their energy from. In thinking through electricity pricing, fleets must be aware
of their utility’s rate card and should keep in mind that installing charging stations may change
the rate card used for their account from a basic rate to a much more complicated EV rate.
Fleets should also be aware of if and how demand charges are integrated into their rate card.

March 1, 2019
Purchaser’s Internal Use Only
75

---

<!-- page: 76 -->

Demand charges can be applied for either total facility use or time of use. In other words, the
price for electricity may be determined not just by how much energy you use, but by how fast
you use it, the peak rate, etc. Though emerging automated demand response (ADR)
technologies may be able to help manage peak loads via smart charging.

However, the grid is not the only place to get electricity from. In fact, on-site “behind the meter”
solutions such as microgrids and renewables like solar PV are slowly gaining popularity as a
means of reducing grid demand, especially during peak times. However, integrating systems
like these into electric fleet charging systems is a very new concept and no data is yet available
as far as best practices.

Regardless of whether fleets choose to install charging infrastructure themselves or to hire a
third-party company to do it for them, they must still consider how they plan to pay for it.

# 11. Financial Assistance

Fortunately for fleets, in addition to the special utility programs and tariff structures mentioned
earlier, depending on the location and project, there are a myriad of financial assistance
programs available to help make vehicle electrification more economically feasible. While some
of these funding mechanisms are focused more on the vehicles themselves, some can also help
cover the cost of charging infrastructure.

Utilities are typically aware of any financial incentives offered within their service territory, so
speaking with a utility representative is usually a good place to start. There are also directories
available online that allow fleets to search for funding support by location. For example, ACT
News offers a Funding & Incentives database on their website that includes federal, state, and
local incentive programs that are currently available to assist fleet operators in the deployment
of clean vehicles and equipment and new infrastructure developments.133 It users to
search by location or eligible fuels such as electricity.

> **Figure 68 — ACT News Funding & Incentives Database Screenshot**
> Type: screenshot
> Axes: none
> Series: none
> Values: none
> Notes: ACT News

March 1, 2019
Purchaser’s Internal Use Only
76

---

<!-- page: 77 -->
March 1, 2019
Purchaser’s Internal Use Only
77

For example, a quick search of “Colorado” tells us that the state has a program called Charge Ahead Colorado (CAC) that can be used to help fund electric vehicles. A quick Google search then shows us that CAC, a grant program administered by Colorado Energy Office and the Regional Air Quality Council’s Clean Air Fleets program, offers funding to cover up to 80% of the cost of a charging station up to $9,000 for a Level 2, dual port station or $30,000 for a standard DCFC station.134

> **Figure 69: Charge Ahead Colorado Program Summary (Clean Air Fleets)**
> Type: table
> Axes: n/a vs n/a
> Series: n/a
> Values: 
> | Funding | RAQC | CEO |
> |---|---|---|
> | Eligible Fleets | Fleets and entities located in the seven county Denver Metro Area (Adams, Arapahoe, Boulder, Broomfield, Denver, Douglas and Jefferson Counties). | Entities located in Colorado outside of the seven county Denver Metro Area. |
> | EV Funding Available | RAQC will fund 80% of the incremental cost differential between an EV and the comparable gasoline vehicle up to $8,260. | CEO is not funding EVs. |
> | Charging Station Funding Available | RAQC and CEO will fund 80% of the cost of a charging station up to the following set maximums:<br>• Level 2, Dual Port Station: $9,000<br>• Level 3, Multiple Connection Standard Station: $30,000<br><br>Please see the Application Guide for more information on charging station types. |
> | Funding Priority | Priority is directed to those organizations that are excluded from existing state tax credits and incentives. For both charging stations and EV funding, eligible applications include local governments, school districts, State agencies, and non-profit agencies. Apartment/condominium complexes and businesses that own multi-vehicle parking facilities for fleet, public or guest / visitor are also eligible for charging station funding. | Funding is directed to private non-profit or for-profit corporations, state agencies, federal agencies, public universities, and public transit agencies, in addition to local governments, landlords of multi-family apartment buildings and homeowner associations (as defined more specifically in C.R.S. Article 33.3 of title 38). |
> Notes: Charge Ahead Colorado Program Summary (Clean Air Fleets)

Programs like this exist across much of the country, and many of them are made possible thanks to the Volkswagen (VW) settlement funds. As part of the 2016 settlement for its emissions violation, VW was legally required to contribute almost $3 billion to an Environmental Mitigation Trust set up to fund projects that reduce nitrogen oxide (NOx) emissions from vehicles. The funding was distributed state-by-state, based on the number of affected vehicles in their jurisdiction, and each state’s allocation is managed by beneficiaries who determine, through a beneficiary mitigation plan (BMP), how, when, and by whom the funds can be used.

---

<!-- page: 78 -->
March 1, 2019
Purchaser’s Internal Use Only
78

> **Figure 70: VW Settlement Funds Dashboard (VW Settlement Clearinghouse)**
> Type: dashboard / screenshots containing multiple charts and maps
> Axes: n/a vs n/a
> Series: Region 1 through Region 9, and various states
> Values: 
> - Total Funding: $2,925M
> - 2.0L Allocation Funding: $2,700M
> - 3.0L Allocation Funding: $225M
> - Funding by U.S. EPA Region: Region 1: $216M, Region 10: $211M, Region 2: $208M, Region 3: $318M, Region 4: $457M, Region 5: $404M, Region 6: $283M, Region 7: $90M, Region 8: $141M, Region 9: $512M
> - Funds by State: California: $423M, Texas: $209M, Florida: $166M, New York: $128M, Pennsylvania: $119M, Washington: $113M, Illinois: $109M, Virginia: $94M, North Carolina: $92M, Maryland: $76M, Ohio: $75M
> - Other Settlement Allocations: Puerto Rico State Trust Administration Cost Subaccount: $8,125,000; Tribal: $25,422,769; Tribal Administration: $54,447,921; Total: $4,916,189, $92,911,880
> Notes: This dashboard shows the allocation of funds by state from the Environmental Mitigation Trust. This dashboard was created by Atlas Public.

Many states are still drafting their BMPs, but repowering or replacing vehicles and building out electric vehicle charging station infrastructure are both types of projects (known as “eligible mitigation actions”) allowed under the settlement.135

The online VW Settlement Clearinghouse features an interactive dashboard that allows users to see their state’s settlement-related progress, including key contacts at lead agencies, upcoming events (such as informational sessions and public comment meetings), and website links.136

---

<!-- page: 79 -->
March 1, 2019
Purchaser’s Internal Use Only
79

> **Figure 71: VW Settlement State Progress and Contacts Dashboard filtered for Tennessee (VW Settlement Clearinghouse)**
> Type: dashboard / screenshots
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Screenshots show State and Local Contacts, State Lead Agencies, Events, Websites, and Other Information by State for Tennessee.

For example, by filtering for Tennessee (as shown above in Figure 71), one can see that the Tennessee Department of Environment and Conservation (TDEC) has released its final BMP for implementing the State’s allocation of over $45 million. By clicking the link, one can view the BMP itself and can see that TDEC plans to release project solicitations for Class 4-7 Local Freight Trucks, Class 8 Local Freight Trucks and Port Drayage Trucks, and Light Duty Zero Emission Vehicle (ZEV) Supply Equipment.137

> **Figure 72: State of Tennessee’s Beneficiary Mitigation Plan (TDEC)**
> Type: photograph of document cover
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Cover of State of Tennessee's Beneficiary Mitigation Plan, Volkswagen Settlement Environmental Mitigation Trust, Prepared by the Tennessee Department of Environment and Conservation. Date of Release: September 21, 2018.

---

<!-- page: 80 -->
March 1, 2019
Purchaser’s Internal Use Only
80

Similarly, one can filter by Pennsylvania to see that they have also released their final BMP in the form of the Driving PA Forward initiative. By clicking the link, one can view the BMP and can see that the state plans to spend over $17.5 million of its $118.5 settlement funds on grants and rebates for EVSEs.138

> **Figure 73: Driving PA Forward Program Funding Chart (PDEP)**
> Type: pie chart
> Axes: n/a vs n/a
> Series: Ferries, Tugs, Freight Switchers; Multi Category Rebates; ZEV Supply Equipment Grants/Rebates; Heavy Duty Trucks & Transit Buses; DERA; GSE, Forklifts, Port Cargo; Ocean Going Shorepower
> Values: Total = $118.5 (million); Ferries, Tugs, Freight Switchers: 5.9; Multi Category Rebates: 5.9; ZEV Supply Equipment Grants/Rebates: 16.1; Heavy Duty Trucks & Transit Buses: 29.9; DERA: 34.1; GSE, Forklifts, Port Cargo: 17.7; Ocean Going Shorepower: 8.9
> Notes: Driving PA Forward Diesel Programs ($million)

One can also see that they are currently running a $30 Million Rebate Program for Class 4-7 Trucks, Port Drayage Trucks, School Buses, and Shuttle Buses that will cover up to $135,000 for fleets to replace one of their Class 4-5 vehicles with an electric vehicle, and up to $142,500 if they also purchase charging infrastructure as part of the project.

> **Figure 74: Maximum Rebate Amounts for Government and Non-Government Owned Electric Replacement or Repower Projects (PDEP)**
> Type: table
> Axes: n/a vs n/a
> Series: n/a
> Values:
> | Project Type | Maximum Rebate Electric - Vehicle Only | Maximum Rebate Electric - with Infrastructure | OR (whichever is less) | Maximum % of Total Project Cost (Electric) |
> |---|---|---|---|---|
> | NG and G - Class 4-5 Vehicle Replacement or Repower | $135,000 | $142,500 | or | 75% |
> | NG and G - Class 6-7 Vehicle Replacement or Repower | $157,500 | $165,000 | or | 75% |
> | NG and G - Drayage Truck Replacement or Repower | $225,000 | $232,500 | or | 75% |
> | NG and G - School Bus Replacement or Repower | $187,500 | $195,000 | or | 75% |
> | NG and G - Shuttle Bus Replacement or Repower | $151,500 | $159,000 | or | 75% |
> Notes: Maximum Rebate Amounts for Government and Non-Government Owned Electric Replacement or Repower Projects (PDEP)

Joe Annotti with Gladstein, Neandross & Associates (GNA) summarizes the near-term funding horizon for electric vehicles as “as bright as it ever has been. The problem faced by fleets and other stakeholders will no longer be where can they find the funds, but how they can secure the

---

<!-- page: 81 -->
March 1, 2019
Purchaser’s Internal Use Only
81

right funding opportunity. In other words, fleets no longer have to wonder if there are funds, but how much they can win.”139

While some of the funding mentioned above – the VW settlement funds, for example – is finite and may only last a few years, the longevity of others is expected to be much greater. For example, goals like those found in California's Zero Emission Vehicle (ZEV) plans, have long implementation windows and are expected to be incentivized for some years until EV and charger volumes are high enough that their cost is comparable to existing options.

Hopefully knowing that there are so many financial assistance opportunities across the country encourages fleets to start transitioning to electric vehicles and to, in doing so, invest in the necessary charging infrastructure.

# 12. Implementation Considerations

It should be clear by now that fleets planning for vehicle electrification need consider many variables for implementation. And while each project by necessity involves some bespoke engineering (since each site and project is different), there are some common factors to consider. A suggested chronological roadmap, including key considerations is outlined below and shown in Figure 75.

---

<!-- page: 82 -->
March 1, 2019
Purchaser’s Internal Use Only
82

> **Figure 75: Charging Procurement Roadmap (NACFE)**
> Type: schematic / flowchart
> Axes: n/a vs n/a
> Series: n/a
> Values: 
> Steps shown along a winding road:
> 1. Engage Utility: to evaluate existing infrastructure, programs, case studies, etc.
> 2. Choose Vehicle(s): and consider duty cycle, range, dwell time, battery capacity, charge port, etc.
> 3. Determine Charging Needs: accounting for daily kWh needed, charging time(s), charging speed, utility tariffs, software, etc.
> 4. Assess Financing: to explore utility programs/incentives, local, state, and federal grants and rebates, ownership model (capex vs. opex), etc.
> 5. Procure Charging Components: including hardware, software, and maintenance and repair service plan
> 6. Design Site Plan: including charging location and spacing
> 7. Apply for Permits: before construction or installation
> 8. Deploy Charging Infrastructure: construction, installation, software licensing, and connection
> Notes: Charging Procurement Roadmap (NACFE)

---

<!-- page: 83 -->
March 1, 2019
Purchaser’s Internal Use Only
83

## 12.1. Step 1. Engage the Utility

Because of the nature of EV charging infrastructure, utilities will need to be involved in the planning and implementation processes as partners, and because they are subject to much regulation and government bureaucracy, it’s best to engage them early, as planning and permitting can take over a year. Not only can your local utility provide you with information on your current distribution-level and on-site electrical infrastructure – everything from your substation to your transformer through to your meter – but they can also provide information on any programs or special tariff structures they offer for commercial and industrial (C&I) EV customers.

Utilities are also worth engaging early because they will have the best sense of what local funding opportunities (grants, rebates, etc.) may be available for the types of projects you’re considering. This type of information is good to know early because it may impact not just what charging infrastructure you choose, but what vehicle(s) you choose to purchase. For example, some grant programs require fleets to purchase vehicles that are made (or at least assembled) in the U.S. And since many grant programs will want proof that you’ve engaged with your utility, why wait?

Your utility may also have experience (or even case studies) working with other fleets in their service territory and may be able to share lessons learned and best practices with you. And even if you’re the first trucking fleet they’ve ever helped to electrify, the utility still has decades of expertise when it comes to electrical infrastructure and can act as a sounding board and advisor through scenario planning to understand what new infrastructure and/or upgrades may be involved.

There are approximately 3,300 electric utilities in the U.S. alone, and each one has varying tariff structures, demand rates, time of use energy charges, and programs (or lack thereof) to work with fleets on EV charging, so it is important to make sure you’ve got the latest information from *your* utility.

> **Figure 76: Map of U.S. Investor Owned Utility Service Territories (EEI)**
> Type: map with callouts
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Map of U.S. Investor Owned Utility Service Territories (EEI). Includes legend for Transmission-Only Utilities and Map Abbreviation Key.

---

<!-- page: 84 -->
March 1, 2019
Purchaser’s Internal Use Only
84

## 12.2. Step 2. Choose Vehicle(s)

Once fleets have spoken with their local utility about their goals and have a better sense of what is possible (both technologically and economically), they should choose which vehicle(s) they want to deploy (if they haven’t already). When choosing vehicles, fleet should consider what duty cycle they’re trying to replace or supplement, the anticipated operations of the vehicle (e.g. daily range, dwell time, etc.), its battery capacity, charge port type, size, weight, and any additional features that the fleet wants to prioritize (e.g. autonomous software, ability to customize, etc.). See the first two guidance reports in this series for more information on what to consider when choosing an electric vehicle.

## 12.3. Step 3. Determine Charging Needs

Once fleets have chosen the electric vehicle(s) they will be deploying, they should then be able to determine their charging needs. In order to do this, they will need to anticipate how much the truck’s batteries will be depleted during each shift and will also need to anticipate how much time the truck will have for charging throughout the day, including opportunity charging.

For example, if a fleet is planning to purchase BYD’s Class 6 truck, with a battery capacity of 221 kWh and a range of 124 miles but knew that their two-shift-per-day operations would only require the truck to travel approximately 65 miles each day, therefore only depleting the battery about halfway, they could anticipate needing to recharge the truck only about 110 kWh worth in order to return it to full charge each night. And because they know that a 19.2 kW Level 2 charger can accomplish this in less than six hours, they can safely rely on a Level 2 charger to meet their needs, knowing that they will have a full eight-hour timeframe in which to charge overnight.

> **Figure 77: BYD Class 6 Truck (BYD)**
> Type: photograph
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: BYD Class 6 Truck (BYD)

---

<!-- page: 85 -->
March 1, 2019
Purchaser’s Internal Use Only
85

> **Figure 78: Example Two-Shift-Per-Day CBEV Operation (NACFE)**
> Type: schematic / diagram
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Diagram showing flow: Vehicle Parked (Load/Charge) -> Driver 1 Shift (Deliveries) -> Vehicle Parked (Load/Charge) -> Driver 2 Shift (Deliveries).

Or suppose the fleet has three shifts-per-day rather than two. This minimizes the amount of dwell time the truck will have to charge, which may make Level 2 charging unfeasible. The fleet may then need to plan on DCFC options. The same might also be true if the fleet expects longer routes and therefore plans to deplete the battery down to a lower SOC by the end of the day.

> **Figure 79: Example Three-Shift-Per-Day CBEV Operation (NACFE)**
> Type: schematic / diagram
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Diagram showing flow: Vehicle Parked (Load/Charge) -> Driver 1 Shift (Deliveries) -> Vehicle Parked (Load/Charge) -> Driver 2 Shift (Deliveries) -> Vehicle Parked (Load/Charge) -> Driver 3 Shift (Deliveries).

More information on representative duty cycles can be found in the earlier *Medium-Duty Electric Trucks: Cost Of Ownership* report.

When anticipating charging needs, fleets will also want to ensure they choose EVSEs with connector types (J1772, CCS, CHAdeMO, etc.) that are compatible with the port(s) on the truck they’ve chosen. They will also need to identify the number of charge ports needed. For example, if the fleet is planning to purchase five vehicles, it’s important to understand if they will need to be charged all at once (thereby requiring five ports) or if they can be charged sequentially (thereby requiring fewer ports).

Utilities should be able to help fleets think through their charging needs, likely even modeling multiple scenarios, though fleets may also want to consider hiring a consultant to help them think through these questions, particularly for their first electrification project or a relatively complex one. And per SCE’s guidance, fleets may also want to consider planning their electric infrastructure to accommodate more charging stations in the future (to realize economies of scale and minimize inconvenience from construction).141 After all, if current EV adoption trends continue, the population of EVs at the fleet’s site will likely grow. One key to keep in mind is that

---

<!-- page: 86 -->
March 1, 2019
Purchaser’s Internal Use Only
86

fleet want to ensure that charger utilization is high, so that they get the most value out of their capital expense. This is especially important for fleets that are subject to demand charges, which can be less troublesome when spread between lots of vehicle charging hours.

Finally, fleets will also need to determine whether they will require energy management capabilities to help manage costs.

## 12.4. Step 4. Assess Financing

Once the fleet has determined their charging needs, they should then work with their utility, their state-level VW settlement lead agencies, and online databases to identify any funding incentives that may be used to help cover the cost of the charging infrastructure they’re planning for. We recommend that fleets take this step before actually procuring the equipment since information learned through the funding application process may impact the exact equipment that a fleet chooses. Information on financial support options can be found above in the Financial Assistance Section.

Keep in mind that many funding opportunities may be limited in both time (e.g. application deadlines) and scope (e.g. only available for DCFCs, in certain geographic areas, up to a max amount, etc.), and try to find the best match for your project given its characteristics and constraints.

During this step, fleets should also work closely with their finance department to understand whether ownership or leased business models will work best for them, depending on their capex and opex preferences.

## 12.5. Step 5. Procure Charging Infrastructure and Services

Once the fleet has determined their charging needs and come up with a plan for financing the project, they should then begin the procurement process for charging equipment and services. Fleets may want to evaluate several charging equipment models. And they should consider that stations with multiple charge ports may be more cost-effective in terms of both hardware and installation costs.

Fleets will need to work directly with their utility in the event that electrical upgrades to the site (such as new transformers or panels) are required. These sorts of upgrades will be much more common for larger deployments (in terms of both number of trucks and battery size), and may takes months to finalize, so if a fleet suspects these upgrades may be necessary, they should begin the procurement process for them as soon as possible.

Fleets should also procure any services they will need for data management, payment, load management, maintenance and repair.

---

<!-- page: 87 -->
March 1, 2019
Purchaser’s Internal Use Only
87

## 12.6. Step 6. Design Site Plan

Once charging equipment and services have been procured, fleets should create a site design. This design is particularly important as it will influence your construction costs, including utility-side infrastructure upgrades and on-site equipment and installation costs. While the charger itself should already be ordered, fleets should not underestimate other site needs. For example, according to the Edison Electric Institute, “trenching concrete and running wire have proven to be much more expensive than the actual EVSE. Depending on the number of electric vehicles planned for a particular location, a circuit upgrade may also be needed but how those costs are handled will vary from company to company.”142 Fleets can try to minimize costs by designing charging locations on site near existing electrical infrastructure.

## 12.7. Step 7. Apply for Permits

This is a short but important step. Fleets should always be sure to apply for all necessary permits before starting any construction or electrical work on site. This will help avoid any issues – legal or otherwise – in the future. Depending on who you procure your charging equipment from, and who is handling installation, they may apply for permits on your behalf.

## 12.8. Step 8. Deploy Charging Infrastructure

Once you’ve got your permits, it’s finally time to begin construction. After completion of the electric infrastructure, your contractors will connect the charging equipment and test them for operation. They should also schedule and pass any required inspections for your deployment prior to operation of the charging equipment.

The roadmap outlined above will have the same general steps regardless of number or size of trucks; however, as fleets scale the number of electric vehicles at each site, the charging procurement process will become exponentially more complex and time-consuming.

> **Figure 80: Charging Implementation Complexity (NACFE)**
> Type: line chart
> Axes: Number of e-Trucks vs Time Intensity of Planning Charging / Complexity of Planning Charging
> Series: Complexity curve
> Values: Exponential upward curve showing increasing complexity and time intensity as the number of e-trucks increases.
> Notes: Charging Implementation Complexity (NACFE)

---

<!-- page: 88 -->
March 1, 2019
Purchaser’s Internal Use Only
88

The charging procurement process may also increase in complexity if advanced components like on-site renewables or storage – which are not included in the roadmap above – are incorporated.

This implementation process may be lengthy, both because of external (e.g. RFPs) and internal steps (e.g. legal reviews), but as more and more fleets and utilities gain more and more experience, this process will become more streamlined as a common “cookbook” approach evolves.

> **Figure 81: Dozens of Electric Vehicles Charging at their Homebase (Hard Working Trucks)**
> Type: photograph
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Dozens of Electric Vehicles Charging at their Homebase (Hard Working Trucks)143

# 13. Additional Considerations

In addition to the opportunities and challenges mentioned above, additional considerations also exist within the EV charging space. These range from important distinctions between diesel and electric vehicle fueling to trends fleets should pay attention to. These additional considerations are covered below.

## 13.1. Employee Safety

Many fleets we spoke with who were relatively unfamiliar with electric vehicle charging had concerns about safety, particularly involving high voltage lines. However, as touched on briefly in the Charger Basics section above and in NACFE's prior guidance reports on electric trucks,

---

<!-- page: 89 -->
March 1, 2019
Purchaser’s Internal Use Only
89

safety protocols have been deployed in hardware and software focused at minimizing risks. That said, high-powered chargers are still a work in progress, as evidenced by Electrify America’s January 2019 decision to shut down the majority of the high-powered chargers (150 kW-350 kW) in its network in order to investigate a potential safety issue with its liquid-cooled cables.144 Although the chargers came back online quickly, after being ruled safe a few days later, the incident demonstrated the relative adolescence of some of these technologies.145

Similarly, fleets are cautious about the physical demands on employees who are tasked with charging vehicles. For example, as chargers get faster and faster, power cords get larger and heavier, and fleets want to ensure that employees aren’t going to injure themselves lifting them. Luckily, thanks to the incorporation of liquid cooling systems, higher level charging rates are permitted with cabling that can still reasonably be maneuvered by an individual. Though NACFE expects cable ergonomics will continue to be a consideration for high power charging systems, particularly as the market starts to demand charging at megawatt levels.

## 13.2. Fueling Schedules and Operator Time Requirements

Fleets may also be weary of the more frequent fueling needs of electric vehicles compared to diesel vehicles. As mentioned in the first report in this series, most battery electric vehicles will need to charge once daily, whereas on comparable routes over a week’s time, a medium-duty diesel might only fill up once, and a heavy-duty vehicle might fill up twice. And while the time to actually fuel is indeed longer for electric vehicles than diesel ones, the net time requirements on the operator are not much different. For example, charging a CBEV at a charging station – putting the plug into the truck’s port – requires about the same level of effort as putting a fuel pump nozzle into a fuel tank. In a diesel scenario, a driver would wait for fueling to complete, then put the nozzle back on the pump. In electric charging, the driver or service tech would move to other tasks while the vehicle charged. The net labor time difference between a once-a-week fuel stop and a once nightly charging over a week seems minimal.

## 13.3. Scaling

Many fleets we spoke with had anxiety about scaling from a smaller pilot to a larger CBEV deployment. While the implementation process, described above, is undoubtedly more complex for larger deployments, fleets should not let this fact dissuade them. They should also not be paralyzed by the unknown of what the vehicle makeup of their fleets will be in 5, 10, or 20 years. Rather, fleets should jump in and get started at whatever level they’re at when it comes to electrification. NACFE expects that as more and more fleets do so, and utilities, regulators, and companies gain experience, new, improved, and expanded programs will be offered to help fleets scale electric vehicle deployments and associated charging infrastructure procurements based on lessons learned. This will give fleets a voice in guiding the rapidly evolving technologies and services tied to electric trucks.

NACFE expects energy management services to increase in popularity as electric truck deployments scale.

---

<!-- page: 90 -->
March 1, 2019
Purchaser’s Internal Use Only
90

## 13.4. Grid Services

Many utilities, think tanks, and academics across the country are interested in the ancillary grid services that electric vehicles may be able to provide the grid. Indeed, many “share the vision that a fleet of electric vehicles can act as a set of battery storage assets to help balance the power grid, which is especially relevant in a largely renewable energy-powered future.”146 As noted by Rocky Mountain Institute (RMI)’s report, *Electric Vehicles as Distributed Energy Resources*, “today’s fast-changing EV-charging market represents the beginnings of a demand-side opportunity like no other: intelligent, interactive electricity demand that is movable in time and space.”147 An electric truck with a 200 kWh battery stores as much electricity as an average U.S. residence consumes in a week.

As the RMI report goes on to say, “considered as a pooled resource, the growing number of electric vehicle batteries could provide a wide range of valuable grid services, from demand response and voltage regulation to distribution-level services, without compromising driving experience or capability. Electric utility companies can use new communications and control technologies, together with innovative tariffs and incentive structures, to tap the sizeable value potential of smart electric-vehicle charging to benefit utility customers, shareholders, vehicle owners, and society at large.”

> **Figure 82: Electric Vehicles as Distributed Energy Resources (Rocky Mountain Institute)**
> Type: photograph of report cover
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Cover of Rocky Mountain Institute report: Electric Vehicles as Distributed Energy Resources by Garrett Fitzgerald, Chris Nelder, and James Newcomb.

Demand response (DR), for example, allows a utility to remunerate EV customers for the ability to curtail vehicle charging during periods of peak power demand. Indeed, demand response programs are already being piloted with commercial vehicles. Because of their unique duty cycles, school buses are a prime target for pilot studies. For example, PG&E, through their Excess Supply Pilot (XSP), is working with the Pittsburg Unified School District in California to encourage school bus operators to charge their buses during peak solar hours due to the excess power being generated by California’s abundant solar panels.148

---

<!-- page: 91 -->
March 1, 2019
Purchaser’s Internal Use Only
91

> **Figure 83: PG&E School Bus Renewables Integration Pilot (Liberty PlugIns HYDRA-R)**
> Type: schematic / diagram
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Diagram illustrating PUSD On-site Solar and Wind Generation connected via PG&E Grid, XSP, Liberty HYDRA-RX, and ClipperCreek CS-100 EVSE to school buses at Pittsburg USD Administrative Offices and Central Yard.

Another school bus pilot is taking place in White Plains, New York, where the the local utility, Consolidated Edison, is working with the school district to help them purchase electric school buses in exchange for the right to use the buses during the summer when school is out and during which time they will be plugged into the local electric grid, where they’ll be used to store surplus electricity and discharge it when it’s needed most.149 This ability to allow truck batteries to provide capacity back to the grid at times of low power generation is known as “vehicle-to-grid” (V2G), and through this innovative pilot, it is expected to help the utility reduce costs and make it easier to build intermittent generation sources such as wind and solar. Combining renewables with battery storage lets the utility store extra power for moments when the wind dies down or the sun sets.

And while school buses clearly have more flexibility due to their limited operations throughout the day (and nonexistent operations during the summer), utilities are also interested in piloting DR and V2G programs with electric trucks. It’s worth noting some of the requirements in the fine print of SCE’s Charge Ready program, covered above in section 9. For example, all Level 2 charging stations funded by the program are required to be DR-capable (i.e., capable of receiving and executing real-time instructions to throttle, and/or modify the end-user pricing of EV charging load) and were encouraged to include additional load management features (e.g., EV charging sequencing or sharing).150 And program participants are contractually required to participate in future DR programs designed in connection with the program and approved by the Commission.

Utilities are also interested in further piloting and deploying V2G technologies, which can allow EVs to function as grid supply—serving the same functions as power generators—as well as being grid loads. Using bidirectional energy flows between the grid and the vehicle’s batteries, EVs could pump electricity back onto the grid at times of high demand and participate in the

---

<!-- page: 92 -->
March 1, 2019
Purchaser’s Internal Use Only
92

ancillary services markets, providing services like frequency and voltage regulation, reactive power for power factor correction, and reserve capacity.151 However, beyond a few small pilot projects – none of which involve commercial vehicles – V2G has yet to become a reality. And even if/when V2G programs are developed, they will likely target passenger vehicles first since shifting commercial fleet operations to align with V2G opportunities is much harder to do. That said, however, many EVSEs on the market are being manufactured with this bidirectional energy flow capability in order to “future proof” the hardware and avoid stranded assets, should V2G become a reality. CharIN e.V. has laid out a roadmap for full V2G integration by about 2025.152

> **Figure 84: CharIN’s V2G Roadmap (CharIN)**
> Type: chart / diagram (Grid Integration Levels)
> Axes: n/a vs n/a
> Series: Level 1 - V1G Controlled Charging, Level 2 - V1G/H Cooperative Charging, Level 3 - V2H Bidirectional Charging, Level 4 - V2G Aggregated (bidirectional) charging
> Values: Timeline intervals: today, till 2020, till 2025, around 2025.
> Notes: CharIN e.V. V2G Roadmap 2018-11-19 Version 4.

Communications technology for V2G capabilities also continues to progress, most notably with Ford recently announcing that it plans to move ahead with “C-V2X” communications technology in its vehicles, even as others such as GM and Toyota move forward with dedicated short-range radio communication (DSRC) devices, which are considered outdated by tech companies but were mandated by an Obama-era rule that is now on hold as federal regulations remain in limbo.153

---

<!-- page: 93 -->
March 1, 2019
Purchaser’s Internal Use Only
93

## 13.5. Clean Energy/Renewables Integration

As described above, using vehicle batteries as electricity storage can help utilities build intermittent generation sources such as wind and solar. However, utilities and fleets are also interested in integrating renewables – whether on site or onto the grid – to power electric vehicle charging.

In fact, achieving sustainability goals such as greenhouse gas emissions reductions was the primary motivator for going electric cited by fleets in the UPS/GreenBiz study.154 However, one common question NACFE hears from fleets is whether, depending on their local grid generation mix (i.e. coal vs natural gas vs renewables), if deploying electric vehicles and charging them from the grid is actually more sustainable. This is a common question particularly from fleets located in geographies that rely heavily on coal plants for electricity generation.

The short answer is yes, even in coal-heavy regions, charging an electric vehicle from the grid is still better from an emissions perspective than burning diesel or gas. In fact, according to a 2018 study by the Union of Concerned Scientists (UCS), commercial vehicles (in the study case, buses) have lower global warming emissions than diesel and natural gas buses everywhere in the country.155

> **Figure 85: Map of Global Warming Emissions Improvements when Comparing Battery Electric Vehicles to Diesel Throughout the U.S. (UCS)**
> Type: map
> Axes: n/a vs n/a
> Series: Good (less than 2x diesel); Better (2-3x diesel); Best (greater than 3x diesel)
> Values: Various MPG ratings shown across geographic regions on the U.S. map ranging from 6.8 MPG to 37.3 MPG. Diesel bus reference: 4.8 MPG. US electric bus average: 12.0 MPG.
> Notes: Map of Global Warming Emissions Improvements when Comparing Battery Electric Vehicles to Diesel Throughout the U.S. (UCS). Note regarding MPG data and GREET 2017 model.

New research from Bloomberg New Energy Finance confirms this, as they found that worldwide, carbon dioxide emissions from battery-powered vehicles were about 40% lower than for internal combustion engines.156 The superior emissions reductions even held in China, which is still heavily reliant on coal.

---

<!-- page: 94 -->
March 1, 2019
Purchaser’s Internal Use Only
94

> **Figure 86: Forecast Grid-Related Emissions from BEV Operations by Country (Bloomberg)**
> Type: line chart
> Axes: Year (2015 to 2040) vs gCO2/mile (0 to 300)
> Series: U.K., Germany, France, U.S., China, Japan, Average ICE
> Values: 
> - Average ICE: approx. 300 (2015) down to approx. 200 (2040)
> - China: approx. 230 (2015) down to approx. 100 (2040)
> - Germany: approx. 200 (2015) down to approx. 80 (2040)
> - U.S.: approx. 150 (2015) down to approx. 50 (2040)
> - Japan: approx. 150 (2015) down to approx. 40 (2040)
> - U.K.: approx. 100 (2015) down to approx. 10 (2040)
> - France: approx. 10 (2015) down to approx. 10 (2040)
> Notes: Source: BloombergNEF New Energy Outlook 2018. Note: The average ICE CO2 emissions are sales-weighted across all six countries

And electric vehicles and their associated charging infrastructure are one of the only investments a fleet can make that gets greener the longer you operate it. This is because the grid is steadily greening year to year as coal plants are retired and renewable generation is added. As Colin McKerracher, a transport analyst at BNEF noted, “When an internal combustion vehicle rolls off the line its emissions per km are set, but for an EV they keep falling every year as the grid gets cleaner.”

Though fleets with ambitious clean energy goals (or who service customers with clean energy goals that include their supply chain) may want to go above and beyond the renewables that are currently incorporated on their local grid. Luckily, corporate renewable energy procurement options are ever-evolving. NACFE recommends that fleets interested in these options review the many great resources offered by the Business Renewables Center.

---

<!-- page: 95 -->
March 1, 2019
Purchaser’s Internal Use Only
95

> **Figure 87: Buyers Roadmap (BRC)**
> Type: flowchart / diagram
> Axes: n/a vs n/a
> Series: n/a
> Values: n/a
> Notes: Buyers Roadmap (BRC) diagram showing phases: DETERMINE (electricity load & company goals), UNDERSTAND (the different options), BUILD (internal support for a transaction / a team (internal and/or external) / market approach / structure & refine options), IDENTIFY (your risk tolerances), RUN (RFP), OBTAIN (C-suite approval), NEGOTIATE (a contract), OBTAIN (Final Approval), FINALIZE (pre-COD details), ENSURE (post -COD follow-up).

## 13.6. Workforce

As mentioned briefly earlier in this report, there may be limitations to what a fleet’s workforce is able to do to support vehicle charging. For example, some union job descriptions may separate physically plugging in a vehicle from a driver's duties. This division of duties may exist today where vehicles are fueled by different staff than drivers. As mentioned above, wireless charging can be a potential solution to this challenge.

Automation in general can also be a solution, not just to minimize new tasks but also to ensure that charging is done effectively. That is, as the process for charging becomes more automated, the risk of user error decreases. Workforce training and clear procedures combined with computer automation has the potential to reduce risks such as failure to charge a vehicle overnight. Automated monitoring systems can allow preventive maintenance to spot failing equipment and allow timely repair so as not to impact deliveries. And since failures with respect to vehicle charging can translate to increases in downtime, operators need to minimize these sorts of failures as much as possible. For this reason, automation is something that is being built into EVSE systems more and more and something to consider when spec’ing a charger.

And while little is known about the impacts of vehicle electrification on driver attraction and retention, generally speaking, it is understood that a significant percentage of Millennials, the next generation of the workforce, care about issues like climate change and protecting the environment, though only 10% to 13% feel their organizations are doing something to address these challenges.158 This is a clear opportunity for improvement for companies looking to retain talent, given that Millennials intend to stay longer with employers that engage with social and environmental issues like these. Therefore, sustainable business practices like electrifying vehicles, may encourage the next generation of drivers to sign on with a certain fleet and help retain those drivers once hired.

---

<!-- page: 96 -->
March 1, 2019
Purchaser’s Internal Use Only
96

## 13.7. Ratepayer Benefits

The benefits of electric vehicles are much further-reaching than just the drivers who drive them or the fleets that manage them. In fact, due to the grid services mentioned above, as well as the environmental and public health benefits that come with reduced air pollution, all ratepayers within a utility’s service area benefit from the adoption of EV technology. Ratepayers also benefit financially because the fixed cost of the grid can be spread across a larger volume of sales, thereby driving down rates.159 Therefore, more and more regulators across the country are incentivizing utilities to expand infrastructure.160 And because most utility regulation is based on precedent, we expect these sorts of dockets and policies to continue to expand.

## 13.8. Utility Business Model Reform

Fleets should be aware that while their business models are changing and responsibilities expanding, so are those of utilities. For example, while utilities at one point just had to worry about electricity reliability (e.g. avoiding blackouts), affordability, and safety, the list of needs and expectations is now a much longer one, including new components like environmental performance, resilience, expanded choice, and innovation.161 That is, it is no longer enough to just “keep the lights on”. Rather, utilities, due to pressure from customers, regulators, and investors, are now also expected to lower emissions, quickly bounce back after disruptions, and ensure customer choice.

> **Figure 88: Expanding Utility Responsibilities and Expectations (Rocky Mountain Institute)**
> Type: diagram / flowchart
> Axes: n/a vs n/a
> Series: Long-Standing Responsibilities vs Modern Needs and Expectations
> Values: 
> - Long-Standing Responsibilities: Safety, Reliability, Affordability
> - Modern Needs and Expectations: Environmental Performance, Resilience, Expanded Choice, Innovation
> Notes: Expanding Utility Responsibilities and Expectations (Rocky Mountain Institute)

---

<!-- page: 97 -->
March 1, 2019
Purchaser’s Internal Use Only
97

And in order to meet these new expectations, utility business models are changing quite rapidly. Rocky Mountain Institute’s *Navigating Utility Business Model Reform* guide provides an excellent overview of these changes, both potential and realized (Figure 89), and the objectives they aim to achieve (Figure 90).162

> **Figure 89: Utility Business Model Reform Options (Rocky Mountain Institute)**
> Type: diagram / flowchart
> Axes: n/a vs n/a
> Series: I. Adjustments to the Cost-of-Service Model, II. Leveling the Playing Field, III. Retirement of Uneconomic Assets, IV. Reimagined Utility Business
> Values: 
> - I. Adjustments to the Cost-of-Service Model: I.a Revenue Decoupling, I.b Multiyear Rate Plans, I.c Shared Savings Mechanisms, I.d Performance Incentive Mechanisms (PIMs)
> - II. Leveling the Playing Field: II.a Changes to Treatment of CapEx/OpEx, II.b New Procurement Practices
> - III. Retirement of Uneconomic Assets: III.a Securitization, III.b Accelerated Depreciation
> - IV. Reimagined Utility Business: IV.a Platform Revenues, IV.b New Utility Value-Added Services
> Notes: Utility Business Model Reform Options (Rocky Mountain Institute)

---

<!-- page: 98 -->
March 1, 2019
Purchaser’s Internal Use Only
98

> **Figure 90: Utility Business Model Reform Objectives (Rocky Mountain Institute)**
> Type: table
> Axes: n/a vs n/a
> Series: n/a
> Values:
> | OBJECTIVES | DESCRIPTIONS |
> |---|---|
> | Remove utilities’ incentive to grow energy sales | By breaking the tie between utility revenues and energy sales, also called decoupling, utilities no longer have a disincentive to invest in solutions that decrease sales. |
> | Realign profit-making incentives | Earning opportunities for utilities can be structured so they are aligned with public policy goals consistent with a clean, affordable, and reliable energy future. Ratemaking reforms may be focused on encouraging energy conservation, reducing peak load, removing the utility capex bias, or addressing risk imbalances between shareholders and ratepayers. |
> | Develop new utility revenue and profit opportunities | As grid and customer needs evolve, utilities have the opportunity to provide new services and products to end-users and third parties compatible with a clean energy future. Changes to regulations and earning mechanisms may be necessary to ensure utilities can provide new offerings without damaging new competitive markets. |
> | Revise risk and value sharing | New utility financial incentives can create a better balance between ratepayers, utility managers, and shareholders as to who bears the brunt of business risks. Shared savings mechanisms between ratepayers and utilities can better balance the monetary value gained from achieving operational or programmatic goals. |
> | Encourage cost containment | Promote economic efficiency in investment decisions and business operations through realigning business practices and profit incentives in a manner that lessens cost overruns. Although cost containment has been a long-standing priority of utility regulations, new needs and opportunities force a reconsideration of how best to incentivize cost control. |
> Notes: Utility Business Model Reform Objectives (Rocky Mountain Institute)

Fleets should understand these trends impacting the utility industry and how they impact the feasibility of and enthusiasm for electric vehicles.

# 14. Summary Findings

NACFE’s research into charging infrastructure for commercial battery electric vehicles to date has revealed a number of findings, summarized here from this report:

*Private for Now, Public Later*
● The focus for the foreseeable future of electric truck charging will be on private, “depot-” or “return-to-base charging”. Public fast charging corridor networks capable of quickly charging commercial vehicle batteries will not be a reality in the immediate future.

*Allow Plenty of Lead Time*
● Planning and permitting for charging infrastructure can be a very time-intensive process, so fleets should appreciate lead times and start early. Indeed, infrastructure planning, negotiating, funding, permitting, installation, and certification can take much longer than procuring the truck itself.

*Collaborate with Utility and Others*
● Fleets planning to electrify some or all of their vehicles should work closely with their local utility from the planning stages through to implementation. In fact, fleets should also develop relationships with regulators, cities, neighbors, OEMs, charging system

---

<!-- page: 99 -->
March 1, 2019
Purchaser’s Internal Use Only
99

providers, etc. Information changes rapidly in this technology field, and having a network of partners communicating regularly can help keep you on track.

*Don’t Discount Software & Maintenance*
● Charging hardware is quickly becoming commoditized. Therefore, fleets should focus on differentiating products and companies based on their software, network, and maintenance offerings and should ensure that they are comparing apples to apples during the procurement process.

*See the Big Picture of the Grid*
● Fleets must develop a fairly sophisticated understanding of the broader charging ecosystem, including a basic understanding of the electric grid, so as to be able to see the forest for the trees of their individual site. Specifically, fleets should take the following considerations into account when planning for charging infrastructure:
  ○ Existing electric infrastructure and demand
    ■ physical electrical transmission and distribution infrastructure
    ■ other facility loads (building HVAC, offices, e-forklifts, etc.)
    ■ Location of transformer and panel on site
  ○ Type(s) of vehicles needing to be charged
    ■ Make, model, battery size, connector/port type, etc.
  ○ Number of vehicles
  ○ Anticipated daily duty cycle of vehicle(s) and expected impact on battery SOC
  ○ Time available for charging (dwell time between shifts, opportunity charging, etc.)
  ○ Electricity rates in utility service territory (including TOU and demand charges)

*Not One Size Fits All*
● Because of the nature of the considerations above, planning should be done on a site-by-site basis; however, planning techniques and best practices can be scaled to other additional locations.

*Mitigate Costs*
● Demand charges can make or break the economics of CBEV charging operations. NACFE expects charging to prosper most (and therefore fleet electrification to happen most) where special programs are implemented to help mitigate these costs, at least in the initial stages of technology adoption.

*Charge Smarter, Not Costlier*
● In order to help minimize both infrastructure capital and installation costs and operating expenses, fleets should consider investing in smart, networked charging software and services, particularly for deployments of multiple vehicles and/or vehicles with large battery capacities.

---

<!-- page: 100 -->
March 1, 2019
100
Purchaser’s Internal Use Only

*Make Demands on Tech Providers*
● Fleets should demand improvements from technology providers and inform them quickly of all dissatisfactions. Fleets are the proving ground for this technology, so they have the responsibility to be objective and clarify demands so technology providers can iterate.

*Moderate Expectations*
● Fleets are encouraged to moderate their initial expectations. All new technologies go through learning curves. Fleets – including technicians, drivers, customers, management, and investors – should not make rash conclusions in the first months or year of operation. Rather, all these stakeholders should educate themselves and go into the process optimistic but also with the expectation that solutions will be iterative as experience amasses.

*Act Short Term, Plan Long Term*
● No one has all the answers today. Get your feet wet while learning what's needed to be competitive in the future.

# 15. Recommendations

Fleets as well as utilities, regulators, and technology providers are constantly learning and iterating in this rapidly evolving space. And innovative utility programs and rate structures are allowing commercial battery electric vehicles to charge successfully and economically in growing areas of the country. Though much broader and faster design and approval of these sorts of programs by utilities and regulators will be needed to scale electric vehicle adoption across the nation. As much as possible, EV-friendly programs and rate structures should be standardized so that fleets with operations that span multiple utility service territories can scale their electrification efforts without having to reinvent the wheel in each new territory. It’s important to remember that utilities are relatively new to the EV charging space as well, and that although it will require a significant departure from their historical rate structures and business models, it is in their financial interest to support the build-out of charging infrastructure because it offers additional ratebasing investments and load growth opportunities in an otherwise plateauing market.

It is also imperative that utilities understand the important differences between passenger EVs and commercial EVs. Not only is the charging capacity much higher for CBEVs, but they have unique needs and constraints due to their mission-focused operations, which are much less flexible than personal vehicle usage and charging times. As such, CBEVs need to be looked at as a distinct market rather than an extension of the passenger EV market.

While the charger itself is the most visible piece of the charging infrastructure ecosystem, it in indeed just that – a piece. In order to be successful, fleets must understand this dynamic and must focus more on the big picture than on simply comparing EVSEs. We expect more and more innovative networking and maintenance options to arise. Software will be invaluable as

---

<!-- page: 101 -->
March 1, 2019
101
Purchaser’s Internal Use Only

smart charging will be key to minimizing costs while also ensuring mission critical uptime of vehicles. Many business models exist to help manage charging, and fleets will need to decide what trade-offs they’re comfortable making between risk management and price volatility. Fleets who develop expertise in smart charging will have a leg up on their peers, though innovative partnerships will allow even fleets new to the electrification space to be successful.

Smart charging and V2G capabilities may also enable new grid services that, if compensated for appropriately, may be a win-win for utilities, fleets, and ratepayers. That said, it is imperative that these services are piloted in the real world for further refinement, as they are mostly hypothetical today.

Last but certainly not least, charging infrastructure, though no doubt not sufficient today, should not be considered an insurmountable problem. As Rick Mihelic and Mike Roeth point out in *Electric Trucks: Where They Make Sense*, it is relevant to note that Thomas Edison’s first patent for the light bulb was filed in 1879 well before there was a North American power grid. Light bulb and electric motor technology ignited national development of new infrastructure to adapt society to the new technology rather than forcing the technology to fit poorly into the existing infrastructure. Figure 91 diagrams that the power grid infrastructure was demand driven based on success of the electric devices that needed it. This lag between product introduction and infrastructure investment has been repeated many times, and there’s no reason to think it won’t be repeated for CBEV charging infrastructure as well.

> **Figure 91: Infrastructure Follows Market Adoption of Revolutionary Technologies (NACFE)**
> Type: diagram / timeline with images
> Axes: n/a vs n/a
> Series: n/a
> Values: 
> - 1879 Lightbulb
> - 1888 Electric Motor
> - 1893 Redlands Power Plant 250 kW
> - 1896 Niagra Falls Powers Buffalo 11,185 kW
> - 1926 Wilson Dam 83,500 kW
> - 1939 Hoover Dam 343 MW
> - 1961 Hoover Dam 1,458 MW
> - 2017 Hoover Dam 2,236 MW
> Notes: Infrastructure Follows Market Adoption of Revolutionary Technologies (NACFE)

---

<!-- page: 102 -->
March 1, 2019
102
Purchaser’s Internal Use Only

# 16. Bibliography

1 Mihelic, R., Roeth, M., “Electric Trucks – Where They Make Sense,” Guidance Report, North American Council for Freight Efficiency (NACFE), May 1, 2018, https://nacfe.org/future-technology/electric-trucks
2 Mihelic, R., Roeth, M., “Medium-Duty Electric Trucks – Cost of Ownership,” Guidance Report, North American Council for Freight Efficiency (NACFE), October 1, 2018, https://nacfe.org/future-technology/medium-duty-electric-trucks-cost-of-ownership/
3 UPS, GreenBiz, “Curve Ahead: The Future of Commercial Fleet Electrification,” October 1, 2018, https://sustainability.ups.com/media/UPS_GreenBiz_Whitepaper_v2.pdf
4 EV100, “Business Driving Demand For Electric Vehicles: EV100 Progress and Insights Annual Report, February 2019”, The Climate Group, February 4, 2019, https://www.theclimategroup.org/sites/default/files/downloads/ev100_progress_and_insights_annual_report_2019.pdf
5 Mihelic, R., Roeth, M., “Electric Trucks – Where They Make Sense,” Guidance Report, North American Council for Freight Efficiency (NACFE), May 1, 2018, https://nacfe.org/future-technology/electric-trucks
6 “Blink Level 2 Pedestal EV Charger”, Blink Charging, https://www.blinknetwork.com/chargers-commercial-l2-pedestal.html
7 International Electrotechnical Commission, “Electric Vehicle Conductive Charging System,” IEC 61851-1:2017 and related parts -21, -23, -24, https://webstore.iec.ch/publication/33644
8 International Electrotechnical Commission,” Plugs, socket-outlets, vehicle connectors and vehicle inlets - Conductive charging of electric vehicles - Part 1: General requirements,” IEC 62196-1:2014 and related parts -2 & -3, https://webstore.iec.ch/publication/6582
9 Herron, D., “Range Confidence: Charge Fast, Drive Far, with your Electric Car,” Greentransportation.info, 2016, https://greentransportation.info/ev-charging/range-confidence/chap8-tech/ev-dc-fast-charging-standards-chademo-ccs-sae-combo-tesla-supercharger-etc.html
10 TÜV SÜD, “Progress on EV fast charging standards harmonization,” TÜV SÜD, https://www.tuv-sud.com/home-com/resource-centre/publications/e-ssentials-newsletter/e-mobility-e-ssentials/e-mobility-e-ssentials-vol.-1/progress-on-ev-fast-charging-standards-harmonization
11 Holland, M., “CCS Becoming Dominant DC Charging Standard In Europe — Will Nissan Drop CHAdeMO?”, Clean Technica, November 22, 2018, “https://cleantechnica.com/2018/11/22/ccs-becoming-dominant-dc-charging-standard-in-europe-will-nissan-drop-chademo/
12 CharIN, “Vision and Mission”, https://www.charinev.org/about-us/vision-mission/
13 “High Power Charging for Commercial Vehicles (HPCCV) request for submission”, CharIN, https://www.charinev.org/hpccv/
14 Mihelic, R., Roeth, M., “Electric Trucks – Where They Make Sense,” Guidance Report, North American Council for Freight Efficiency (NACFE), May 1, 2018, https://nacfe.org/future-technology/electric-trucks
15 “Developing Infrastructure to Charge Plug-In Electric Vehicles”, Alternative Fuels Data Center, https://afdc.energy.gov/fuels/electricity_infrastructure.html#dc
16 “Greenlots to collaborate with Volvo Trucks to Provide EV Charging Infrastructure for New Zero-Emissions Warehouses and Electrified Fleets in Southern California”, Greenlots, November 7, 2018, https://greenlots.com/greenlots-to-collaborate-with-volvo-trucks-to-provide-ev-charging-infrastructure-for-new-zero-emissions-warehouses-and-electrified-fleets-in-southern-california/
17 Gene, “Close-up look at the Tesla Semi “Megacharger” charging port”, Teslarati, November 17, 2017, https://www.teslarati.com/tesla-semi-megacharger-charging-port-close-up-look/
18 “Electric Vehicle Charging Guide,” ChargeHub website, https://chargehub.com/en/electric-car-charging-guide.html
19 Sladen, P. “Chademo-combo2-iec-type-2-connectors-side-by-side”, Wikimedia Commons, August 24, 2017, https://commons.wikimedia.org/wiki/File:Chademo-combo2-iec-type-2-connectors-side-by-side.jpg
20 “eCanter,” Mitsubishi Fuso product specifications brochure, http://www.mitfuso.com/enus/models/ecanter

---

<!-- page: 103 -->
March 1, 2019
103
Purchaser’s Internal Use Only

21 Transpower Connecting/Disconnecting Instructional Video, hosted on Vimeo, https://vimeo.com/254955118
22 “TransPower Electric Yard Tractor Standard Specifications Based on the Ottawa 4x2 Off Road Yard Tractor,” Transpower spec sheet, http://www.transpowerusa.com/downloads/Spec-sheet-TransPower-Ottawa-4x2-Off-Road-Yard-Tractor.pdf
23 “Liquid Cooled CCS1 & CCS2 High Power Charging Solutions,” ITT Cannon product brochure, https://www.ittcannon.com/Core/medialibrary/ITTCannon/website/Literature/Catalogs-Brochures/ITTCannon-EVC-DC-Liquid-Cooled-Brochure.pdf?ext=.pdf
24 “High Power Charging”, Phoenix Contact, 2017, https://www.phoenixcontact.com/assets/downloads_ed/global/web_dwl_promotion/52007586_EN_HQ_E-Mobility_LoRes.pdf
25 “DC charging cable - EV-T2HPCC-DC500A-5,0M50ECBK11 - 1085638,” Phoenix Contact, https://www.phoenixcontact.com/online/portal/us/?uri=pxc-oc-itemdetail:pid=1085638&library=usen&pcck=P-29-03-01-01&tab=2&selectedCategory=ALL#Cable
26 Afridi, K., “Wireless Charging of Electric Vehicles,” University of Colorado Boulder, published in “Frontiers of Engineering: Reports on Leading-Edge Engineering from the 2017 Symposium,” National Academy of Engineering, 2018, Washington, DC: The National Academies Press. https://doi.org/10.17226/24906.
27 “Highly Resonant Wireless Power Transfer: Safe, Efficient, and over Distance,” Wi Tricity Corporation white paper, 2017, http://witricity.com/wp-content/uploads/2016/12/White_Paper_20161218.pdf
28 “Wireless Power Transfer for Light-Duty Plug-In/Electric Vehicles and Alignment Methodology J2954_201711,” SAE International, Nov. 27, 2017, https://doi.org/10.4271/J2954_201711
29 “The Next Wireless Revolution: Electric Vehicle Wireless Charging - Power and Efficiency,” Wi Tricity Corporation white paper, 2018, http://witricity.com/wpcontent/uploads/2018/03/WIT_White_Paper_PE_20180321.pdf
30 “Developing Infrastructure to Charge Plug-In Electric Vehicles”, Alternative Fuels Data Center, https://afdc.energy.gov/fuels/electricity_infrastructure.html#dc
31 Lambert, F., “The first 200-kW wireless charging system for electric buses is deployed,” electrek, Apr 19, 2018, https://electrek.co/2018/04/19/200-kw-wireless-charging-system-electric-buses/
32 Momentum Dynamics website, https://www.momentumdynamics.com/
33 Adler, A., “Volvo Invests in Wireless Charging Supplier Momentum Dynamics”, Trucks, January 17, 2019, https://www.trucks.com/2019/01/17/volvo-invests-wireless-charging-supplier-momentum-dynamics/
34 “How eHighway works,” Siemens, https://www.siemens.com/global/en/home/products/mobility/roadsolutions/ electromobility/ehighway.html
35 “First electric road in Sweden inaugurated,” Trafikverket, Swedish Transport Administration, https://www.trafikverket.se/en/startpage/about-us/news/2016/2016-06/first-electric-road-in-swedeninaugurated/
36 “Electrified roads – a sustainable transport solution of the future,” eRoadArlanda website, https://eroadarlanda.com/globally-unique-electrified-roadenables-fossil-free-road-transport/
37 Ryan, J., “Zero Emission, All-Electric Buses Take to Louisville Streets,” WFPL 89.3 News, http://wfpl.org/zero-emission-electric-buses-take-louisville-streets/
38 Gilpin, L., “These City Bus Routes are Going Electric – and Saving Money,” Inside Climate News, Oct. 23, 2017, https://insideclimatenews.org/news/18102017/these-city-bus-routes-are-going-all-electric
39 “Fast-Charging, all electric ZeroBus,” U.S. Department of Energy (DOE), video, https://vimeo.com/180798047
40 “King Country Metro All-Electric Bus - from Proterra,” Proterra, video, September 1, 2015, https://www.youtube.com/watch?v=b1YW1LV6S_w
41 “ABB powers 450kW fast charging for electric buses in Gothenburg,” ABB website, Nov. 22, 2017, http://www.abb.com/cawp/seitp202/be24f0a9ca998b59c12581e00031630b.aspx
42 “Overhead Fast-Chargers”, Proterra, https://www.proterra.com/technology/chargers/

---

<!-- page: 104 -->
March 1, 2019
104
Purchaser’s Internal Use Only

43 Pelletier, S., et al, “Battery Electric Vehicles for Goods Distribution: A Survey of Vehicle Technology, Market Penetration, Incentives and Practices,” Interuniversity Research Centre on Enterprise Networks, Logistics and Transportation (CIRRELT), CIRRELT-2014-43, Sep 2014, https://www.cirrelt.ca/DocumentsTravail/CIRRELT-2014-43.pdf
44 Lambert, F., “Nio Builds a Battery Swap Station Right Next to a Tesla Supercharger”, Electrek, November 1, 2018, https://electrek.co/2018/11/01/nio-battery-swap-station-next-tesla-supercharger/amp/
45 Loveday, E., “Navistar begins commercial production of eStar electric cargo van, first four going to FedEx,” Autoblog, May 15, 2010, https://www.autoblog.com/2010/05/15/navistar-begins-commercialproduction-of-estar-electric-cargo-va/
46 Gallo, J., Tomic, J., “Battery Electric Parcel Delivery Truck Testing and Demonstration,” CALHEAT Research Center/ CALSTART Inc., Aug 2013, http://www.calstart.org/Libraries/CalHEAT_2013_Documents_Presentations/Battery_Electric_Parcel_Delivery_Truck_Testing_and_Demonstration.sflb.ashx
47 Bakke, N., Linhoj, J., Rask, M., “Better Place - Transforming the global auto industry?,” Department of Business Administration, Business and Social Sciences, Aarhus University, Denmark, Nov. 20, 2015, https://www.researchgate.net/publication/260046588_Better_Place_Transforming_the_Global_Auto_Industry
48 “A Broken Place: The Spectacular Failure Of The Startup That Was Going To Change The World,” Innovation Agents, April 7, 2014, https://www.fastcompany.com/3028159/a-broken-place-better-place
49 Pelletier, S., et al, “Battery Electric Vehicles for Goods Distribution: A Survey of Vehicle Technology, Market Penetration, Incentives and Practices,” Interuniversity Research Centre on Enterprise Networks, Logistics and Transportation (CIRRELT), CIRRELT-2014-43, Sep 2014, https://www.cirrelt.ca/DocumentsTravail/CIRRELT-2014-43.pdf
50 Mihelic, R., “Envisioning Future Commercial Vehicles,” presentation for Stifel Transportation Equipment Equity Research, Mihelic Vehicle Consulting, Aug. 30, 2017. (available from author, or through Stifel at https://stifel2.bluematrix.com/sellside/EmailDocViewer?encrypt=8fe7415e-801a-48c0-ade6-3a8e7418dddc&mime=pdf&co=Stifel&id=null&source=mail ) http://www.stifel.com/
51 https://store.clippercreek.com/level2/level2-40-to-80/cs-100-70-amp-80-amp-ev-charging-station
52 https://www.blinknetwork.com/chargers-commercial-l2-pedestal.html
53 HDT Staff, “Worldwide Annual Electric Truck Sales May Reach 332,000 by 2026,” Heavy Duty Trucking, Jan. 3, 2017, http://www.truckinginfo.com/channel/fuel-smarts/news/story/2017/01/worldwide-annual-electric-truck-sales-may-reach-332-000-by-2026.aspx
54 https://www.tesla.com/support/supercharging
55 https://www.tritium.com.au/veefillpk
56 Nicholas, M., Hall, D.,” Lessons Learned on Early Electric Vehicle Fast-Charging Deployments,” International Council on Clean Transportation (ICCT), Jul 2018, https://www.theicct.org/publications/fast-charging-lessons-learned
57 Howell, D, et al, “Enabling Fast Charging: A Technology Gap Assessment,” U.S. Department of Energy Office of Energy Efficiency & Renewable Energy, Oct 2017, https://energy.gov/sites/prod/files/2017/10/f38/XFC%20Technology%20Gap%20Assessment%20Report_FINAL_10202017.pdf
58 Fitzgerald, G., Nelder, C., “From Gas To Grid,” Rocky Mountain Institute (RMI), https://www.rmi.org/insight/from_gas_to_grid/ or https://www.rmi.org/wpcontent/uploads/2017/10/RMI-From-Gas-To-Grid.pdf
59 Long, M., “Charging All-Electric Trucks,” Transport Topics (TT), May 11, 2018, https://www.ttnews.com/articles/charging-all-electric-trucks
60 https://www.tesla.com/models
61 https://freightliner.com/e-mobility/
62 https://chanje.com/wp-content/uploads/2018/08/Chanje-V8070-2018-brochure.pdf
63 https://freightliner.com/e-mobility/

---

<!-- page: 105 -->
March 1, 2019
105
Purchaser’s Internal Use Only

64 “Express Plus,” Specifications for Power Block, Power Modules and Station, ChargePoint brochure, https://www.chargepoint.com/files/datasheets/ds-expressplus.pdf
65 “ChargePoint Express Plus“ ChargePoint website, https://www.chargepoint.com/products/commercial/express-plus/
66 “ChargePoint Reveals New Concept Design for High-Powered Charging of Electric Aircraft and Semi-Trucks”, ChargePoint, May 9, 2018, https://www.chargepoint.com/about/news/chargepoint-reveals-new-concept-design-high-powered-charging-electric-aircraft-and-semi/
67 Gene, “Close-up look at the Tesla Semi “Megacharger” charging port”, Teslarati, November 17, 2017, https://www.teslarati.com/tesla-semi-megacharger-charging-port-close-up-look/
68 Alvarez, S., “Tesla Semi’s temporary ‘Megacharger’ system glimpsed in Madonna Inn sighting”, Teslarati, November 8, 2018, https://www.teslarati.com/tesla-semi-temporary-megacharger-system-madonna-inn-sighting/
69 “BU-409: Charging Lithium-ion,” Battery University website, https://batteryuniversity.com/index.php/learn/article/charging_lithium_ion_batteries
70 Mihelic, R., Roeth, M., Ballschmidt, B., Lund, J., “Medium-Duty Electric Trucks: Cost Of Ownership”, NACFE, October 7, 2018, https://nacfe.org/future-technology/medium-duty-electric-trucks-cost-of-ownership/
71 Gordon-Bloomfield, N., “What Is EVSE And Why Does Your Electric Car Charger Need It?”, Green Car Reports, October 28, 2010, https://www.greencarreports.com/news/1050948_what-is-evse-and-why-does-your-electric-car-charger-need-it
72 Edison Electric Institute, “Transportation Electrification”, June 2014, http://www.eei.org/issuesandpolicy/electrictransportation/FleetVehicles/Documents/EEI_UtilityFleetsLeadingTheCharge.pdf
73 “ABB Smart Charging solution for bus depots”, ABB EV Charging, October 24, 2017, https://www.youtube.com/watch?v=qxDFS87K_QU&feature=youtu.be
74 Email with Muffi Ghadiali, Founder and CEO of electriphi Inc., January 21, 2019, more info available at info@electriphi.ai
75 “Open vs. Closed Charging Stations: Advantages and Disadvantages”, Greenlots, https://greenlots.com/wp-content/uploads/2018/10/Open-Standards-White-Paper-compressed.pdf
76 “Open vs. Closed Charging Stations: Advantages and Disadvantages”, Greenlots, https://greenlots.com/wp-content/uploads/2018/10/Open-Standards-White-Paper-compressed.pdf
77 “Charging at Home”, U.S. Office Of Energy Efficiency & Renewable Energy, https://www.energy.gov/eere/electricvehicles/charging-home
78 Prohaska, R., et al, “Field Evaluation of Medium-Duty Plug-in Electric Delivery Trucks,” U.S. National Renewable Energy Laboratory (NREL), 2016, doi:10.2172/1337010, https://doi.org/10.2172/1337010
79 Heid, B., Hensley, R., Knupfer, S., and Tschiesner, A., “What’s sparking electric-vehicle adoption in the truck industry?”, McKinsey & Company, September 2017, https://www.mckinsey.com/industries/automotive-and-assembly/our-insights/whats-sparking-electric-vehicle-adoption-in-the-truck-industry
80 Alternative Fuels Data Center (AFDC), www.afdc.energy.gov/fuels/stations_counts.html
81 https://afdc.energy.gov/stations/states
82 AFDC Alternative Fueling Station Locator Data, https://afdc.energy.gov/locator/stations/
83 https://chargehub.com/en/charging-stations-map.html
84 “FAQs”, Electrify America, https://www.electrifyamerica.com/faqs
85 “Locate a Charger”, Electrify America, https://www.electrifyamerica.com/locate-charger
86 “Tesla Semi & Roadster Unveil”, Tesla, November 16, 2017, https://www.tesla.com/semi
87 Johnson, E., “Exclusive: How Tesla's first truck charging stations will be built“, Reuters, February 1, 2018, https://www.reuters.com/article/us-tesla-trucks-charging/exclusive-how-teslas-first-truck-charging-stations-will-be-built-idUSKBN1FM0I9
88 Business Renewables Center, “Buyer Basics 101”, Rocky Mountain Institute

---

<!-- page: 106 -->
March 1, 2019
106
Purchaser’s Internal Use Only

89 https://www.instagram.com/clrkrd/
90 https://www.businessinsider.com/china-panda-shaped-solar-energy-farms-project-2018-6#here-is-a-rendering-of-what-yan-tung-imagined-the-farms-could-look-like-2
91 https://en.wikipedia.org/wiki/Roscoe_Wind_Farm
92 https://nacfe.org/technology/solar-panels-2/
93 https://www.homedepot.com/p/Cuisinart-2-Slice-Compact-Stainless-Toaster-CPT-320/302860899
94 https://www.homedepot.com/p/Whirlpool-25-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-with-Internal-Water-Dispenser-WRF535SWHZ/301688547
95 https://www.eia.gov/tools/faqs/faq.php?id=97&t=3
96 Table 5.b https://www.eia.gov/electricity/sales_revenue_price/
97 https://www.fueleconomy.gov/feg/Find.do?action=sbs&id=40812
98 https://www.fhwa.dot.gov/ohim/onh00/bar8.htm
99 https://www.usbr.gov/lc/hooverdam/faqs/powerfaq.html
100 Otto, K., Lund, J., Roeth, M., “Solar for Trucks and Trailers,” Confidence Report, North American Council for Freight Efficiency (NACFE), June 28, 2018, https://nacfe.org/technology/solar-panels-2/
101 Schefter, K., “Electric Vehicle Infrastructure,” Energy Lawyers and Contracting Officers Forum presentation, Edison Electric Institute, Nov. 16, 2017, https://energy.gov/sites/prod/files/2017/11/f46/33-fupwgfall2017_schefter.pdf
102 Cross-Call, D., Gold, R., Goldenberg, C., Guccione, L., and O’Boyle, M., “Navigating Utility Business Model Reform: A Practical Guide to Regulatory Design”, Rocky Mountain Institute, 2018, www.rmi.org/insight/navigating-utility-business-model-reform
103 Jamison, M. “Rate of Return: Regulation”, Public Utility Research Center, https://bear.warrington.ufl.edu/centers/purc/docs//papers/0528_Jamison_Rate_of_Return.pdf
104 Chowdhury, J., “US Utility Industry and Regulatory Landscape”, SmarterUtility.com, February 14, 2012, https://www.slideshare.net/johnchowdhury/electric-utility-primer-john-chowdhury-2012-final
105 Edison Electric Institute, “Transportation Electrification”, June 2014, http://www.eei.org/issuesandpolicy/electrictransportation/FleetVehicles/Documents/EEI_UtilityFleetsLeadingTheCharge.pdf
106 Xcel Energy Inc., “Time of Use Pricing - How it Works”, https://www.xcelenergy.com/billing_and_payment/understanding_your_bill/residential_rate_plans/time_of_use_pricing/time_of_use_pricing_how_it_works
107 https://www.edmunds.com/fuel-economy/the-true-cost-of-powering-an-electric-car.html
108 https://www.stem.com/resources/demandcharges/
109 “Electric Truck & Bus Grid Integration: Opportunities, Challenges & Recommendations”, CALSTART, September 2015, http://calstart.org/wp-content/uploads/2018/10/Electric-Truck-and-Bus-Grid-Integration_-Opportunities-Challenges-and-Recommendations.pdf
110 http://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=201520160SB350
111 Southern California Edison, “Testimony of Southern California Edison Company in Support of its Application of Southern California Edison Company (U 338-E) For Approval of its 2017 Transportation Electrification Proposals,” January 20, 2017. http://on.sce.com/2kXeu1X
112 “Charge Ready”, Southern California Edison, https://www.sce.com/business/electric-cars/Charge-Ready
113 Griffo, Paul. “SCE Gets Thumbs-Up for Program to Electrify Thousands of Industrial Vehicles”, Edison International, https://energized.edison.com/stories/sce-gets-thumbs-up-for-program-to-electrify-thousands-of-industrial-vehicles?utm_source=scehomepage
114 “Fleet”, Southern California Edison, https://www.sce.com/business/electric-cars/charging-stations-for-fleet
115 https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-30
116 “SB 535 Maps of Disadvantaged Communities”, https://calepa.ca.gov/EnvJustice/GHGInvest/

---

<!-- page: 107 -->
March 1, 2019
107
Purchaser’s Internal Use Only

117 Southern California Edison, “Testimony of Southern California Edison Company in Support of its Application of Southern California Edison Company (U 338-E) For Approval of its 2017 Transportation Electrification Proposals,” January 20, 2017. http://on.sce.com/2kXeu1X
118 “Program Overview: PG&E EV Fleet Ready”, PG&E, January 2019.
119 Pyper, J., “PG&E Proposes Ditching Demand Charges for Commercial EV Charging”, Greentech Media, November 7, 2018, https://www.greentechmedia.com/articles/read/pge-ditch-demand-charges-for-commercial-ev-charging#gs.muckATiM
120 “PG&E Proposes to Establish New Commercial Electric Vehicle Rate Class”, PG&E, November 5, 2018, https://www.pge.com/en/about/newsroom/newsdetails/index.page?title=20181105_pge_proposes_to_establish_new_commercial_electric_vehicle_rate_class
121 “Six projects to facilitate adoption of electric semi-trucks, other electric medium/heavy-duty equipment and cars”, San Diego Gas and Electric, January 19, 2018, http://newsroom.sdge.com/clean/sdge-expand-electric-vehicle-charging-network-support-clean-transportation-and-clean-air
122 “Transportation Electrification Movement”, San Diego Gas and Electric, https://www.sdge.com/residential/electric-vehicles/power-your-drive
123 Everett, M., Fang, C., Ramirez, A., and Schneider, J., “A Modern Rate Architecture for California's Future: Enabling the IOUs to be Effective Change Agents”, Fortnightly Magazine, November 1, 2018, https://www.fortnightly.com/fortnightly/2018/11/modern-rate-architecture-californias-future
124 St. John, J. “Michigan Approves Its First Utility EV Charging Infrastructure Pilot”, Greentech Media, January 10, 2019, https://www.greentechmedia.com/articles/read/michigan-approves-its-first-utility-ev-charging-infrastructure-pilot#gs.kI3lxzba
125 “Michigan approves first electric vehicle charging infrastructure program”, Fox 47, January 9, 2019, https://www.fox47news.com/news/local-news/michigan-approves-first-electric-vehicle-charging-infrastructure-program
126 Schefter, K., “Electric Vehicle Infrastructure,” Energy Lawyers and Contracting Officers Forum presentation, Edison Electric Institute, Nov. 16, 2017, https://energy.gov/sites/prod/files/2017/11/f46/33-fupwgfall2017_schefter.pdf
127 “Amply Power EV Fleet Solution”, Amply, https://www.amplypower.com/
128 “AMPLY Update on Port of Long Beach Electrification Efforts and Amply Power”, Webinar, Amply, December 5, 2018, recording available at https://vimeo.com/305584873
129 https://www.tesla.com/energy
130 “Energy as a Service”, Chanje, https://chanje.com/energy-solutions/
131 Sachgau, Oliver. “ Volkswagen Plans Green-Power Alternative to Tesla's Energy Push.” Bloomberg, January 8, 2019, https://www.bloomberg.com/news/articles/2019-01-08/vw-to-become-electricity-charging-provider-in-battery-car-push
132 https://www.elli.eco/en/
133 https://www.act-news.com/funding-incentives-news/
134 “Charge Ahead Colorado”, Clean Air Fleets, https://cleanairfleets.org/programs/charge-ahead-colorado
135 “About the Settlement”, VW Settlement Clearinghouse, https://vwclearinghouse.org/about-the-settlement/
136 “State Progress and Contacts Dashboard”, VW Settlement Clearinghouse, https://vwclearinghouse.org/#settlement_info
137 Beneficiary Mitigation Plan, “Tennessee Department of Environment and Conservation”, https://www.tn.gov/environment/program-areas/energy/state-energy-office--seo-/tennessee-and-the-volkswagen-diesel-settlement/beneficiary-mitigation-plan.html
138 “Driving PA Forward”, Pennsylvania Department Of Environmental Protection, http://www.depgis.state.pa.us/DrivingPAForward/
139 Annotti, J., “The First VW Settlement Projects Have Been Funded”, ACT News, October 11. 2018, https://www.act-news.com/news/vw-funds-now-hitting-streets/

---

<!-- page: 108 -->
March 1, 2019
108
Purchaser’s Internal Use Only

140 http://www.eei.org/about/members/uselectriccompanies/Documents/EEIMemCoTerrMap.pdf
141 “Transportation Electrification”, Southern California Edison, https://www.sce.com/sites/default/files/inline-files/Business%2BEV-TE%2BInfrastructure%2BFS%2Br3_1_WCAG_1.pdf
142 Edison Electric Institute, “Transportation Electrification”, June 2014, http://www.eei.org/issuesandpolicy/electrictransportation/FleetVehicles/Documents/EEI_UtilityFleetsLeadingTheCharge.pdf
143 Quimby, T., “Ford & DHL electric van offers more than you think & it may be going global”, Hard Working Trucks, January 31, 2018, https://www.hardworkingtrucks.com/ford-dhl-electric-van-offers-more-than-you-think-it-may-be-going-global/
144 “Electrify America Partially Shuts Down Its Network As Its Global Supplier Investigates Safety Issue With High-powered Charging Cables”, Electrify America, January 25, 2019, https://elam-cms-assets.s3.amazonaws.com/inline-files/Electrify%20America%20Announcement%20on%20Partial%20Network%20Shutdown%201.25.2019.pdf
145 Lambert, F., “All high-powered EV charging stations are coming back online after shutdown”, electrek, January 30, 2019, https://electrek.co/2019/01/30/high-powered-ev-charging-stations-online-after-shutdown/
146 UPS, GreenBiz, “Curve Ahead: The Future of Commercial Fleet Electrification,” October 1, 2018, https://sustainability.ups.com/media/UPS_GreenBiz_Whitepaper_v2.pdf
147 Nelder, C., Newcomb, J., and Fitzgerald, G., “Electric Vehicles as Distributed Energy Resources”, Rocky Mountain Institute, 2016, http://www.rmi.org/pdf_evs_as_DERs
148 Williams, F., “PG&E’s School Bus Renewables Integration Pilot”, Liberty PlugIns HYDRA-R, July 10, 2018, PowerPoint deck available by contacting Mr. Williams directly
149 Thompson, A., “The Big Yellow School Bus Is Going Green”, Popular Mechanics, November 13, 2018, https://www.popularmechanics.com/science/green-tech/a25025285/electric-school-bus-white-plains-coned/
150 “Charge Ready Pilot Program Report”, Southern California Edison, April 2, 2018, http://www3.sce.com/sscc/law/dis/dbattach5e.nsf/0/AB93EF695A92832A88258263006AF4A7/$FILE/A1410014-SCE%20Charge%20Ready%20Pilot%20Program%20Report.pdf
151 Nelder, C., Newcomb, J., and Fitzgerald, G., “Electric Vehicles as Distributed Energy Resources”, Rocky Mountain Institute, 2016, http://www.rmi.org/pdf_evs_as_DERs
152 Kane, M., “CCS will offer all the bi-directional charging features”, Inside EVs, January 23, 2019, https://insideevs.com/ccs-combo-standard-v2g-2025/
153 Abuelsamid, S., “Ford Breaks With Auto Rivals By Committing To C-V2X Vehicle Communications Tech”, Forbes, January 7, 2019, https://www.forbes.com/sites/samabuelsamid/2019/01/07/ford-becomes-first-automaker-to-commit-production-c-v2x-communications/#2ae3d4da788f
154 UPS, GreenBiz, “Curve Ahead: The Future of Commercial Fleet Electrification,” October 1, 2018, https://sustainability.ups.com/media/UPS_GreenBiz_Whitepaper_v2.pdf
155 https://blog.ucsusa.org/jimmy-odea/electric-vs-diesel-vs-natural-gas-which-bus-is-best-for-the-climate
156 Hodges, J. “Electric Cars Are Cleaner Even When Powered by Coal”, Bloomberg, January 14, 2019, https://www.bloomberg.com/news/articles/2019-01-15/electric-cars-seen-getting-cleaner-even-where-grids-rely-on-coal
157 “BRC Buyers Roadmap”, Business Renewables Center, http://businessrenewables.org/buyers-roadmap/
158 “The 2017 Deloitte Millennial Survey”, Deloitte, 2017, https://www2.deloitte.com/content/dam/Deloitte/global/Documents/About-Deloitte/gx-deloitte-millennial-survey-2017-executive-summary.pdf
159 McMahon, J., “Electric Vehicles Benefit All Utility Ratepayers”, Forbes, February 1, 2019, “https://www.forbes.com/sites/jeffmcmahon/2019/02/01/electric-vehicles-benefit-all-utility-customers-as-much-as-their-owners/#4de8a52e44fe

---

<!-- page: 109 -->
March 1, 2019
109
Purchaser’s Internal Use Only

160 “Episode 33: Transportation – EV Charging Infrastructure And Utilities”, The Energy Policy Podcast, February 2, 2017, http://energypodcast.colostate.edu/2017/02/02/episode-33-transportation-ev-charging-infrastructure-and-utilities/
161 Cross-Call, D., Gold, R., Goldenberg, C., Guccione, L., and O’Boyle, M., “Navigating Utility Business Model Reform: A Practical Guide to Regulatory Design”, Rocky Mountain Institute, 2018, www.rmi.org/insight/navigating-utility-business-model-reform
162 Cross-Call, D., Gold, R., Goldenberg, C., Guccione, L., and O’Boyle, M., “Navigating Utility Business Model Reform: A Practical Guide to Regulatory Design”, Rocky Mountain Institute, 2018, www.rmi.org/insight/navigating-utility-business-model-reform

---

<!-- page: 110 -->
March 1, 2019
110
Purchaser’s Internal Use Only

# 17. Appendix A: Charging Suppliers

The following suppliers have been identified as relevant to past and present commercial truck applications, so are singled out here for additional discussion regarding their electric vehicle supply equipment (EVSE). However, the list is not exhaustive, and there are a considerable number of competing offerings that may be applicable to trucks due to commonality in connectors and charging levels.

## 17.1. ABB

ABB offers HVC-overnight charging products capable of charging larger fleets of electric buses and trucks during the night. This EVSE solution is scalable and can help enable smaller and cheaper grid connections. The chargers also offer remote diagnostics and management through ABB Ability. These chargers are compliant with CCS and OCPP standards and have a flexible design to allow for roof and floor mounting. ABB has years of experience in creating, installing and maintaining charging infrastructure, including several nationwide charger networks.

## 17.2. Amply Power

Amply Power provides a modular, fully automated, EV charging solution that flattens out the facility’s demand curve by optimizing the charge rate and vehicle process flow, thereby minimizing costs. Through its turnkey solution, Amply provides the bulk of the upfront capital expense for the charging infrastructure and offers a per-electric-mile-driven usage rate to customers. Amply typically acts as the account holder for the utility meter used for EV charging

## 17.3. BTCPower

BTCPower (Broadband TelCom Power, Inc.) offers charging stations for passenger EVs, electric school buses, electric transit buses, electric delivery vans, and Class 8 heavy duty EVs. They offer both Level 2 and DCFC chargers for commercial customers. OCPP is supported by all BTCPower charging stations, which makes them easily integrated with most existing charger network software. They offer DC fast chargers with max power ratings as high as 350 kW. Their fast chargers have both SAE CCS1 Liquid Cooled & CHAdeMO connectors.

## 17.4. ChargePoint

ChargePoint began by focusing on the light-duty passenger vehicle market and has made a name for themselves as one of the most popular EVSE suppliers, with a built-out nationwide network of electric car chargers. They have since broadened their scope to include charging solutions for medium- and heavy-duty fleets as well. ChargePoint stations include 24/7 driver support, cloud-based software with features and plans specific to various industries, and service and maintenance. ChargePoint’s team is able to work with customers to help them through the infrastructure process, from beginning to end. All of their chargers are UL listed and some are ENERGY STAR® certified. Their DC fast chargers all use CHAdeMO connections and are available with CCS1 (SAE J1772™ Combo) and CCS2 (IEC 61851-23) connectors. Their DC ultra-fast charging solution is modular and easily scalable so that it can grow with demand and accommodate the battery technologies of today’s and tomorrow's CBEVs.

---

<!-- page: 111 -->
March 1, 2019
111
Purchaser’s Internal Use Only

## 17.5. Chateau Energy Solutions

Chateau Energy Solutions provides customized, turnkey electric vehicle infrastructure charging solutions for commercial, military, federal fleets, as well as state and local municipalities. This includes the design, installation, project management, and maintenance services of electric vehicle charging stations. Their EV experts will provide strategic insight into how the additional load will impact your facility and how to best integrate into your system. In addition to EV infrastructure deployment, Chateau also offers EVSE maintenance as well as EV controls, metering integration, and analytics.

## 17.6. ClipperCreek

ClipperCreek chargers are popular for residential and workplace charging, but they also offer commercial charging stations for fleets at a variety of power levels. Their stations are capable of electrical load management, and optional data tracking is available.

## 17.7. Eaton

Eaton used to supply both Level 2 and DCFC stations, however in 2015 they announced they would focus on producing other components for CBEVs. That said, there are still Eaton chargers available in the marketplace. Their Pow-R-Stations and other charging products may still be available in the market or encountered in the field.

## 17.8. electriphi, Inc.

electriphi, Inc. provides software for energy management and charging control of electric vehicle fleets. Their software is based on open industry standards and designed to interoperate with any charging infrastructure or electric vehicle type. The key benefits of their platform are the management of energy costs, reliable fleet charging and to manage the transition from conventional to electric fleets.

## 17.9. eMotorWerks

eMotorWerks offers a line of SmartGrid-ready EV charging stations with a max power level of up to 18 kW with world-class grid management and user-facing control features, managed through their cloud-based platform. eMotorWerks provides grid management services such as demand response, frequency regulation, peak shaving, and local balancing that help utilities and ISOs better manage the grid volatility and prepare for and fully leverage accelerating EV adoption. They offer a fleet management software called JuiceNet Enterprise and a professional installation service called JuiceReady. They also offer JuiceNet Green, an emission-minimizing software upgrade that synchronizes with grid generation sources and enables fleets to charge their vehicles when the cleanest energy is available on the grid.

## 17.10. EV Connect

EV Connect offers scalable, flexible, and comprehensive industry-specific solutions for EV charging, including charge stations, software, and 24/7 support.

---

<!-- page: 112 -->
March 1, 2019
112
Purchaser’s Internal Use Only

## 17.11. EVgo

In addition to owning, operating, and maintaining the largest public fast charging network, EVgo builds dedicated EV fast charging stations for their fleet partners. They handle the entire process, from site acquisition, construction, procurement, network management, and customer service.

## 17.12. EVoCharge

EVoCharge offers a Level 2 charging station that uses an SAE J1772 connector and is rated for 7.2 kW max power output. However, they are best known for their retractable reel cable management solution products, which are marketed under the brand name “EVoReel®”. EVoReel locks the charge cordset into place without tension, then once charging is complete, the charge cordset reels back easily, removing unsightly cords from off the ground, reducing tripping hazards (liability concern), and helping mitigate common damage that may result from the cordset lying on the ground (O&M concern).

## 17.13. EVSE LLC

EVSE LLC, a subsidiary of Control Module Inc., is a designer and manufacturer of smart EV chargers for the workplace, parking facilities, public locations, fleets and multi-dwelling residential units. They offer both Level 1 and 2 chargers, both of which are OCPP compliant. They also offer AutoCoil, an automatic electronic cable management feature.

## 17.14. GE

ChargePoint acquired GE’s EV charging network in June 2017 with support for GE’s Durastation and Wattstation chargers. These devices may still be available on the open market under the GE brand and may be encountered in the field.

## 17.15. Greenlots

Greenlots offers turnkey EV charging stations. They offer multiple brands of stations and 24/7 support. Their Greenlots SKY™ EV Charging Network Software enables customers to deploy and manage their own network of smart EV charging stations at scale. They also offer smart charging optimization and grid balancing services. Greenlots was acquired by a subsidiary of Royal Dutch Shell in February 2019.

## 17.16. PlemCo

Pacific Lighting & Energy Management Company (PLEMCo) offers a variety of products geared toward energy and/or water savings, including EV charging systems. They offer chargers from a variety of manufacturers.

## 17.17. PowerFlex Systems

PowerFlex Systems offers adaptive load management software for large scale electric vehicle charging. The software allows fleet managers to schedule charging of each EV based on energy and time requirements, and then optimizes for peak demand.

---

<!-- page: 113 -->
March 1, 2019
113
Purchaser’s Internal Use Only

## 17.18. Rhombus Energy Solutions

Rhombus Energy Solutions’ charging station portfolio has thus far been focused on electric bus charging (they supply chargers for Proterra), but their chargers are capable of charging electric trucks as well. Rhombus’ chargers are smart grid ready, offering bi-directional power flow for future vehicle-to-grid (V2G) capability and meet UL requirement 1741 SA. They utilize a standard J1772-CCS plug-in system.

## 17.19. Siemens

Siemens supplies electric vehicle charging stations for municipalities, corporations, fleets and utilities. Their VersiCharge line of Level 2 chargers use an SAE J1772 connector.

## 17.20. Tellus Power

Tellus Power offerings focus on charging solutions for electric buses, though they are compatible with some electric trucks as well. Their chargers range from 7.2 - 200 kW and are all OCPP compliant.

## 17.21. Tritium

Australian-based company, Tritium offers award-winning DC fast charging stations and has partnered with ChargePoint to install these Veefil stations across the U.S. The fast charging stations are able to charge all vehicles equipped for DC fast charging, using the included SAE-Combo connector or a CHAdeMO connector.

## 17.22. Webasto

Webasto is an OEM charging partner that offers Level 2 chargers that have a SAE J1772 connector. They also offer a wireless EV charger with a max power output of up to 6.6 kW.

---

<!-- page: 114 -->
March 1, 2019
114
Purchaser’s Internal Use Only

# 18. Appendix B: Utilities with Electric Truck Charging Programs

Many utilities now have special programs to support electric vehicle charging. A quick search of your utility’s website should help you find their EV programs and rates. However, many of these programs focus on electric passenger vehicles, and only a handful have programs specially designed for commercial vehicles and their unique characteristics. The following utilities have programs targeted toward commercial battery electric vehicle (CBEV) customers.

## 18.1 Pacific Gas and Electric (PG&E)

PG&E recently launched their new EV Fleet Program, through which it plans to spend $236 million over the next five years to support 6,500 new EVs at over 700 sites. Through the program, PG&E will cover the costs of “make-ready” infrastructure, such as design, permitting, and construction from power pole to charger. They are also planning to offer additional incentives for disadvantaged communities, school buses, and transit buses, such as a rebate on charger equipment and installation costs.

## 18.2 San Diego Gas and Electric (SDG&E)

SDG&E has been working with fleets to help support charging of electric vehicles by building on the success of its Power Your Drive Program. The utility took on new transportation electrification projects in 2018, including adding charging stations at the Port of San Diego, San Diego International Airport, delivery fleet hubs and shuttle hubs to support electric vehicles, including semi-trucks, forklifts, and other medium/heavy-duty equipment. According to its website, it plans to install charging stations for approximately 90 fleet delivery vehicles at multiple locations.

## 18.3 Southern California Edison (SCE)

Building off the success of its passenger vehicle-focused Charge Ready Program, SCE now also offers support for medium- and heavy-duty vehicle charging throughout their service territory via their fleet Transportation Electrification program. The program helps customers design a site plan for charging and then works with them to install the charging system on a new, dedicated circuit with its own panel, meter, and service, separate from any existing electrical service infrastructure on site. The program covers the cost of the new electric infrastructure, including any transformers, trenching, conduit, conductors, or stub-outs, and offers rebates to help reduce the cost of the charger equipment and installation, though vendors and charging stations must be selected from the approved list and chargers must be either Level 1 or 2. (SCE is also offering a DCFC pilot program available to customers who provide parking to the public.) Participants must also agree to select an eligible TOU rate for EV charging.
