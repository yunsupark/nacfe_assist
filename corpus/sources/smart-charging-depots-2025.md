<!-- page: 1 (unnumbered) -->
Cover page: Smart Charging for Electric Trucking Depots report by Ampcontrol and NACFE.

<!-- page: 2 -->
# Table of Contents

- Foreword 2
- Executive Summary 3
- Challenges in Electrifying Truck Fleets 4
  - Grid Capacity Constraints 4
  - Energy Costs and Demand Management 6
  - Operational Uncertainties 8
- Smart Charging for Electric Trucking Depots 10
  - Definition and Overview of Smart Charging 10
  - Core Concepts and Technologies 11
  - Key Functionalities: Load Balancing, Demand Response, Scheduling, and Microgrids 12
  - Benefits and Challenges of Smart Charging 13
- Implementation Strategies for Smart Charging 15
  - Cloud vs. Local Power Management: Pros & Cons 15
  - Microgrid solutions for large fleet sites 19
  - Integration with Depot Management and Telematics 21
  - Real-world Examples of Smart Charging 21
- The Future of Smart Charging for Electric Trucks 24
  - Megawatt Charging Systems (MCS) 24
  - Vehicle-to-Grid (V2G) Integration 24
  - Certification and Regulatory Trends 25
- Conclusions 26

<!-- page: 3 -->
# Foreword

**Joachim Lohse**
CEO at Ampcontrol

At Ampcontrol, we’ve spent the past seven years advancing smart charging—starting with a few vehicles and simple use cases, and now supporting large truck depots with semi-trucks, on-site batteries, and megawatt-scale infrastructure. We’ve worked with nearly every major truck brand, from Volvo and Daimler to BYD, Nikola, and Isuzu.

While our platform has grown, I believe smart charging remains the most critical piece to making electric fleets scalable and truly decarbonizing trucking. The industry has made great strides, but we’re still at the beginning of a long journey. That’s what excites us and drives our work every day.

Together with NACFE, we hope this report helps fleets and partners navigate the path ahead with confidence and clarity.

**Mike Roeth**
Executive Director at NACFE

For 16 years, NACFE has helped fleets and the ecosystem of supporters who help them improve their freight efficiency while beginning to take advantage of many alternative fuels, even those that can produce zero emissions in their operations.

Having completed four Run on Less efforts, focusing on the successes of early pioneers, NACFE showcases the best of the best in trucking leaning into new technologies and operations. It has always been our hope, even expectation, that the industry would use our work to further its understanding to accelerate adoption.

This analysis and report with our friends at Ampcontrol is exactly one of those efforts. Take advantage of the information contained and reach out to us with comments or questions.

<!-- page: 4 -->
# Executive Summary

This report, developed collaboratively by the **North American Council for Freight Efficiency (NACFE)** and **Ampcontrol**, examines the most pressing challenges in electrifying commercial truck fleets and provides a practical framework for overcoming them. It builds on insights from *Run on Less – Electric DEPOT*, a groundbreaking study that tracked over 250 electric trucks across multiple fleet operations. The study confirms that battery-electric trucks are not only operationally viable across many applications but also deliver strong driver satisfaction and performance outcomes.

Despite this progress, the report identifies **grid infrastructure and energy delivery** as critical barriers to scaling electrification. Large fleet depots require substantial power—often up to 5 MW—but most existing sites were never designed for such high energy demands. Grid capacity constraints, slow utility coordination, and lengthy regulatory approvals regularly delay projects by one to five years, stalling fleet deployment plans.

In addition to power availability, fleets face growing complexity in charging infrastructure, energy cost volatility from time-of-use (TOU) rates and demand charges, and the need for high charger reliability to maintain fleet uptime. The transition from diesel to electric also introduces operational uncertainties that must be carefully managed, particularly in terms of vehicle readiness and scheduling.

To address these issues, the report introduces **smart charging and energy management solutions** that allow fleets to better control when and how their vehicles charge. These systems reduce infrastructure costs, optimize energy usage, mitigate demand charges, and ensure operational efficiency. Smart charging also enables seamless integration of solar, battery storage, and telematics, making depots more resilient and cost-effective.

With practical guidance, real-world examples, and technology recommendations, this report provides a clear path forward for fleets, utilities, and policymakers committed to accelerating the adoption of electric trucks while minimizing costs, risks, and delays.

<!-- page: 5 -->
# Challenges in Electrifying Truck Fleets

NACFE’s *Run On Less – Electric DEPOT* report shows that production-level electric vehicles are becoming available for many market segments. After analyzing 291 EV trucks during this first-ever research project, the experience for drivers and operators was overwhelmingly positive. The NACFE team also reported on the advancements in charging infrastructure, showing the many pathways available to companies looking to begin their electrification journeys.

While there were many encouraging findings, a challenge highlighted in the report is that power delivery and infrastructure are still taking too long to install. Major contributors to this challenge are constraints on grid capacity and infrastructure. Ampcontrol, a leading EV Charging Solution Company, identifies that transitioning to electric trucking requires substantial power availability at depots, often reaching capacities as high as 5 MW for a single charging site.

Installing the required charging infrastructure for truck depots is challenging since the existing grid infrastructure is not necessarily designed to support high-capacity charging needs. This often results in significant substation, transformer, and transmission infrastructure investments. These grid limitations pose a major hurdle for fleet operators that aim to scale up their electric vehicle deployments and require careful planning and coordination with utilities and regulators.

## Grid Capacity Constraints

### High Power Demand and Limited Grid Availability

With the increasing size of vehicle batteries, fleet depots have significant energy requirements, with some needing megawatt-scale capacity to support high-power charging operations. Grid limitations often restrict the ability to meet these demands, particularly in urban areas, ports, and warehouse districts where electrical capacity is already constrained. When originally planned and designed for ports, warehouses, and vehicle parking lots, the electrification of vehicles could not have been anticipated. And upgrading grid infrastructure to accommodate EV charging is a costly and time-consuming process, requiring utilities to expand transmission lines, reinforce substations, and deploy advanced grid management systems.

As an example, a warehouse might require 300 kW of power to supply energy for machines, cooling, and other systems. But to install 10 DCFC chargers for the fleet, the site will require at least an additional 1,500 kW of power. And while a warehouse might have some unused capacity, it is often insufficient for the demands of fast chargers and EV trucks.

<!-- page: 6 -->
> **Figure 1 — Additional load from EV chargers is not supported by the site**
> Type: bar chart
> Axes: [none] vs Power (kW)
> Series: Existing Base Load, New Charger Load, New Power Demand
> Values: Existing Base Load: approx. 300, New Charger Load: approx. 1500, New Power Demand: approx. 1800
> Notes: A dashed horizontal line indicates "Site capacity".

## Utility Coordination and Regulatory Delays

Fleet operators often face multi-year delays when seeking grid upgrades due to complex utility approval processes and regulatory bottlenecks. Obtaining the necessary permits and approvals for new grid connections can take anywhere from one to five years, significantly impacting fleet electrification timelines. These delays stem from multiple factors, including limited utility resources, evolving regulatory frameworks, and competing infrastructure demands.

To accelerate the process, fleets and utilities must establish long-term agreements and strategic partnerships that can help streamline approvals and reduce delays. Proactive planning, precise demand forecasting, and early engagement with utility providers are essential. Additionally, regulatory bodies need to adopt policies that prioritize grid readiness for EV deployment, ensuring that utility planning aligns with the rapid growth of electric trucking.

## Charging Infrastructure Complexity

The electrification of truck depots requires diverse charging solutions to accommodate different fleet needs, which is why a typical depot may utilize a mix of Combined Charging System (CCS) fast chargers, Megawatt Charging System (MCS) chargers (once they become commercially available), on-site solar generation, and battery storage. Each of these systems and their components adds value but also complexity to the infrastructure requirements and necessitates an integrated energy management approach.

<!-- page: 7 -->
The adoption of MCS, in particular, is a promising solution for long-haul battery-electric vehicles (BEVs), but deployment remains limited due to ongoing development and standardization efforts. Until MCS technology becomes widely available, fleets must rely on existing DC fast charging solutions, which require careful load balancing and demand response strategies to optimize energy use. As a result, a unified charging management system is essential to coordinate these different charging technologies effectively and ensure maximum efficiency.

> **Figure 2 — WattEV site in California**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photograph showing a WattEV charging site with trucks and charging infrastructure.

## Energy Costs and Demand Management

The cost of electricity is a major factor in the financial viability of electric truck fleets. While diesel prices also fluctuate throughout the year, their hourly volatility is minimal—it makes little difference whether a truck fuels up at 6 PM or 10 PM. In contrast, electricity rates can vary by 40% or more depending on the time of day. In California, peak rates might be around $0.40 per kWh, while off-peak rates could be as low as $0.15 per kWh. These fluctuations are driven by factors such as utility rate structures, demand charges, and market conditions. Effective energy management is crucial for fleet operators to control costs and maintain profitability.

The following section describes the two essential components for energy cost and demand management.

<!-- page: 8 -->
## Time-Of-Use (TOU) Rate Sensitivity

Electricity rates often vary based on time-of-use (TOU), meaning fleets that charge during peak hours can face significantly higher costs. In some cases, peak-hour charging can double electricity expenses compared to off-peak charging. Fleet operators who don’t optimize their charging sites can spend up to 50% more on monthly electricity bills.

If a semi-truck operates five days per week, it consumes around 90,000 kWh; this can mean up to $22,000 in lost savings per year per vehicle.

> **Figure 3 — Exemplary TOU rates in California**
> Type: line chart
> Axes: Time of the day vs USD/kwh
> Series: TOU rate
> Values: off-peak (12am): 0.17$/kwh, mid-peak: 0.20$/kwh, on-peak: 0.42$/kwh, off-peak: 0.17$/kwh
> Notes: Shows rate tiers throughout the day.

## Demand Charges and Cost Control Strategies

Demand charges are based on the highest level of power consumption in a billing period and account for a substantial portion of fleet electricity costs. Even with small sites, demand charges can become expensive since they are not calculated by energy consumption (kWh) but by power (kW). Charging one vehicle at 300 kW will cost you the same demand charges as charging multiple vehicles sequentially at 300 kW.

Demand charges are used by utilities to address the stress to the electrical grid that high demand surges cause. Handling those surges requires significant investment on the part of the utilities in power plants and grid infrastructure. They must be sized to meet these peak demands, even if they are only used at full capacity for a few hours each year. Demand charges allow the utilities to pass these high infrastructure costs on to the consumer.

<!-- page: 9 -->
Below, we discuss how these charges can be mitigated through strategic energy management practices such as load balancing, staggered charging schedules, and the use of on-site energy storage.

> **Figure 4 — Demand charges vs. energy costs**
> Type: line chart
> Axes: Time of the day vs Energy Charge ($/kwh) / Demand Charge ($/kw)
> Series: Demand Charge, Energy Charge
> Values: [visual trend showing electricity and demand cost curves across 24 hours]
> Notes: Illustrates the difference between energy charges and demand charges over the course of a day.

## Long-Term Electricity Price Trends and Industry-wide Electrification

Historically, inflation-adjusted electricity prices have shown a downward trend, offering long-term stability for electric fleet operators. However, the broader shift toward electrification across multiple industries—including manufacturing, construction, and heating—is expected to increase overall electricity demand. This rising demand may put additional pressure on energy markets, influencing electricity pricing and availability for trucking fleets.

## Operational Uncertainties

Uptime and availability of fleet vehicles are key to the success of companies. Logistics companies have spent decades optimizing uptime for diesel vehicles and need the same reliability with electric truck operations. Maintaining the same high level of operational reliability when transitioning to electric vehicles requires overcoming potential grid outages and emergencies, such as wildfires or extreme weather events that can disrupt electricity availability.

<!-- page: 10 -->
## Charger Reliability

Charger downtime is a significant concern for fleets, as any disruption can lead to missed delivery schedules and operational inefficiencies. High failure rates of EV chargers necessitate continuous monitoring and maintenance. Predictive maintenance systems and remote fault detection technologies are increasingly being used to identify potential failures before they occur, reducing downtime and lowering maintenance costs.

Remote troubleshooting capabilities enable fleet operators to diagnose and resolve issues without requiring on-site service, thereby further improving reliability. Additionally, maintaining a diverse charging infrastructure, including backup chargers and energy storage, can help mitigate potential disruptions.

> **Figure 5 — Uptime monitoring for EV chargers and charging depots (Ampcontrol)**
> Type: line chart / dashboard screenshot
> Axes: Time Range (Last 48 hrs) vs Uptime (%)
> Series: Uptime (%)
> Values: Average Uptime: 94.54%
> Notes: Shows charger uptime fluctuating over a 48-hour period with a horizontal reference line around 94-95%.

## EV Readiness and Fleet Optimization

Ensuring that electric trucks are fully charged and ready for dispatch is a persistent challenge. Unlike traditional fueling, which takes minutes, EV charging requires careful scheduling to guarantee vehicle availability. Fleet managers must implement optimized charging schedules to ensure each truck reaches a full state of charge (SoC) before its next route.

Real-time tracking of SoC and automated alerts help fleet operators monitor vehicle readiness, reducing the risk of delays. Integrating telematics with charging management systems provides greater visibility into fleet status, allowing for more effective energy planning and operational coordination.

<!-- page: 11 -->
# Smart Charging for Electric Trucking Depots

## Definition and Overview of Smart Charging

In the 2023 Infrastructure Report, NACFE defined that in almost any circumstance involving more than a single vehicle, smart or managed charging, as opposed to simple unmanaged charging, will enable a fleet to save on operational and electrical infrastructure costs. Smart charging is a method of optimizing charging schedules and power distribution to reduce costs while ensuring that electric vehicles are charged and ready for operation. By integrating charger data with fleet management software, smart charging balances energy demand and prevents excessive peak loads that can strain the grid. This approach allows fleets to manage electricity consumption effectively and align charging times with lower-cost periods, reducing operational expenses. Additionally, smart charging systems facilitate the integration of renewable energy sources such as solar and battery storage, contributing to a lower carbon footprint.

While the implementation of smart charging can be complex, the principles are usually the following:

- Reduce the peak power demand of the fleet
- Shift charging to periods with lower energy prices
- Ensure that all vehicles are charged on time at the lowest possible costs
- Incorporate onsite generation, battery energy storage, and solar.

In the following sections, we will discuss the technology, key functionalities, benefits, and challenges of smart charging.

> **Figure 6 — Data-driven smart charging decision for fleets**
> Type: schematic diagram
> Axes: none
> Series: none
> Values: none
> Notes: Shows inputs (Utility Data, Price Data, Vehicle & Route Data, Charger Data, Solar Data, Battery Storage Data) flowing into "Smart Charging Decisions".

<!-- page: 12 -->
## Core Concepts and Technologies

One of the fundamental components of smart charging is a set of optimization algorithms, which dynamically optimize charging schedules to ensure energy is used efficiently. Algorithms calculate charging rates and prioritize vehicles based on their operational needs, thereby reducing unnecessary energy consumption and peak demand costs. Those charging rates are then sent to each individual charging station as a “command”, and the charging station executes these commands automatically.

To simplify integration efforts, smart charging systems utilize the Open Charge Point Protocol (OCPP) to communicate between EV charging stations and smart charging systems. OCPP is used by nearly all hardware and software brands, allowing real-time meter reading access and providing a standardized way to send charging commands. Very often, the smart charging system is cloud-based software that operates through the chargers' internet connection.

If required, the smart charging system also integrates renewable energy generation, battery storage, buildings, or other systems that are colocated with the charging stations at the depot. Therefore, it is possible to install the smart charging system as a local hardware appliance, such as an AmpEdge controller, to simplify the integration and ensure uninterrupted optimization between the different assets.

> **Figure 7 — Smart charging system sends commands to EV chargers**
> Type: schematic diagram
> Axes: none
> Series: none
> Values: none
> Notes: Illustrates the "Smart Charging System" interacting with Vehicles, Charging Stations, Solar Energy, Energy Storage, and Other External Data.

<!-- page: 13 -->
# Key Functionalities: Load Balancing, Demand Response, Scheduling, and Microgrids

## Dynamic Load Management

The best way to overcome grid constraints and install more chargers than a site might otherwise support is through load management. Load management dynamically distributes power across charging stations, ensuring efficient energy use. When only a few chargers are active, they receive higher power levels, while a greater number of active chargers results in automatic power throttling. By adjusting charging power in real-time based on grid conditions and energy availability, these systems prevent grid overload and reduce peak demand charges. Advanced algorithms anticipate power demand spikes and optimize load balancing across multiple chargers. Additionally, charging prioritization mechanisms ensure that high-priority fleet vehicles receive sufficient charge first, while lower-priority vehicles can be scheduled for off-peak hours.

> **Figure 8 — Load Management monitoring on Ampcontrol**
> Type: dashboard / line chart
> Axes: Time (12:00 to 15:00 next day) vs Power (kW) (0 to 300)
> Series: Max Power (kW), Optimized limit (kW), Power (kW), DR/V2G event
> Values: Live Power: 171.1 kW, Live Utilization: 63%
> Notes: Ampcontrol UI screenshot showing real-time power limits and consumption curves.

## Cost Optimization

When a site has energy contracts with TOU rates, cost optimization algorithms help avoid peak price periods. Smart charging reduces electricity costs by aligning charging events with off-peak utility rates and time-of-use pricing structures. Optimization strategies also enable fleets to participate in demand response programs, adjusting energy consumption based on utility signals to earn financial incentives while supporting grid stability. When implementing these optimizations,

<!-- page: 14 -->
careful planning of vehicle departure times is essential. Avoiding peak price periods without a proper setup can lead to delayed departures.

## Real-time Data Integration and Monitoring

As mentioned above, smart charging is about more than just power and energy costs. It is crucial to consider the state of vehicles upon arrival at the depot and their planned routes. Ensuring on-time departures always takes priority over cost reduction. The integration of telematics and charging management software enables continuous data collection and analysis, allowing fleet operators to monitor energy consumption, battery SoC, charging rates, and vehicle readiness. These capabilities help optimize energy usage, predict charging needs, and enhance overall operational efficiency. Remote monitoring features also support proactive troubleshooting, reducing downtime and improving fleet reliability. While this is less critical for small sites, its importance grows with the scale of operations.

## Grid and Renewable Energy Integration

The larger the vehicles and fleet, the more challenging it becomes to overcome grid constraints solely through load management. If the gap becomes too large, fleets must install additional energy resources such as solar power and energy storage. Smart charging enables the seamless integration of distributed energy resources, dynamically balancing power between on-site generation, stored energy, and grid supply. Advanced energy management systems optimize costs while enhancing sustainability. This capability is especially valuable for fleet depots facing grid constraints, as microgrid technology allows them to operate independently during peak demand periods, reducing reliance on costly grid electricity.

# Benefits and Challenges of Smart Charging

## Financial and Operational Benefits for Truck Operators

As illustrated above, smart charging for trucks leads to significant financial savings both during the planning phase of new charging infrastructure projects and during the operation of charging sites. This, in turn, reduces the total cost of ownership (TCO) for the fleet. The key benefits include:

- **Reduction of infrastructure investments for grid upgrades:** When fleets need to upgrade a grid connection, transformers, or similar infrastructure, smart charging can help minimize these costs. This leads to lower initial project expenses and shorter project timelines. Grid upgrades can cost several million dollars and take up to several years to finish.

<!-- page: 15 -->
- **Lower monthly energy costs:** By optimizing charging schedules through load management and cost-based charging strategies, fleets can significantly reduce their average cost per kilowatt-hour (kWh) and kilowatt (kW) demand charges. At Ampcontrol, we have observed savings of up to 45%. These savings are most significant for fleets operating under TOU rates and demand charges, which are common in commercial electricity contracts.
- **Improved vehicle uptime and operational efficiency:** Smart charging software typically includes alert systems and asset management tools, helping fleets maintain higher uptime. This, in turn, allows fleets to install fewer charging stations while ensuring on-time vehicle availability. Over the long term, this reliability enhances customer satisfaction and strengthens business growth.

## Challenges in the Adoption of Smart Charging

Technology always comes with potential challenges. Since the EV market—especially EV trucking—is still in its early stages, operators need to be aware of the challenges they may encounter.

The success of smart charging often depends on the technology choices companies make when purchasing charging hardware and software. If the selected software is unreliable or does not meet market requirements, fleet operators may face significant difficulties. However, even when the best-fit partners are selected, issues can still arise. Below are some of the most common challenges:

- **Interoperability between chargers and software:** Connecting charging stations to a smart charging system requires real-time communication and data exchange between hardware and software. The market has made significant progress in this area by adopting the OCPP protocol, which enables seamless integration of any charger brand with any smart charging system. However, if chargers or software do not fully support OCPP or have an incomplete implementation, site reliability decreases and failures may occur. The best way to mitigate this risk is to check for OCA certifications, conduct small-scale pilots or hardware tests, sometimes over several weeks or months, to verify interoperability and reduce deployment risks.
- **Interoperability between chargers and vehicles:** When drivers plug a charging cable into a vehicle, both sides must communicate. The vehicle must inform the charger about its readiness to start charging, the allowable power input, and other parameters to ensure safe and efficient charging. However, this communication is not always consistent across vehicles, and when chargers adjust power output dynamically, some vehicles may misinterpret the signals and shut down unexpectedly. It's been observed that new vehicle models with limited field testing are more likely to experience these issues. As a result, fleets

<!-- page: 16 -->
may find that charging sessions are interrupted and vehicles are not fully charged by morning. The best way to prevent this is to conduct interoperability tests before making large vehicle purchases and to implement an alert system within smart charging tools. These alerts notify operators if a session stops unexpectedly. Since vehicles and chargers receive regular software updates, new issues may emerge over time, but many can be resolved through software updates rather than costly hardware replacements.
- **Smart charging uptime and reliability:** Smart charging systems are complex, as they must integrate multiple devices, run real-time algorithms, and anticipate potential challenges such as internet downtime. If a system is unreliable, it can frustrate fleet managers more than it helps. In this case, quality and experience matter more than the number of features. Ensuring 99.995%–99.999% smart charging uptime is critical, and selecting systems that have undergone external certifications—such as OCA, UL 60730-1 or UL 3142, if required—can improve reliability. Additionally, using an on-site controller (as discussed in the following section) can further enhance system stability.

# Implementation Strategies for Smart Charging

## Cloud vs. Local Power Management: Pros & Cons

When installing smart charging, fleets can select two primary approaches: cloud-based smart charging and local onsite smart charging. Each has unique advantages and trade-offs, and many fleets benefit from a hybrid approach that combines both solutions.

### Cloud-Based Smart Charging

Cloud-based smart charging refers to a system in which EV charging stations are directly connected to the internet and continuously communicate with a remote central server, typically hosted in a data center. Charging decisions—such as scheduling, load balancing, and energy optimization—are processed in the cloud and then transmitted back to the charging stations over the network. While sites may have external routers and Ethernet cables, they do not rely on on-site controllers or local algorithms.

Cloud-based systems enable remote monitoring and management, providing fleet operators with real-time insights into energy consumption and charger performance. These platforms seamlessly integrate with telematics systems, fleet dispatching software, and demand response programs, facilitating advanced energy strategies such as predictive maintenance and dynamic pricing optimization.

<!-- page: 17 -->
> **Figure 9 — Cloud-based smart charging without local hardware**
> Type: schematic diagram
> Axes: none
> Series: none
> Values: none
> Notes: Shows cloud-based optimization platform communicating via OCPP directly with EV chargers.

### Key Advantages of Cloud-Based Smart Charging

- **Computational power:** Cloud systems can execute advanced optimization algorithms, including artificial intelligence (AI) and machine learning (ML) models, to maximize efficiency and reduce costs. These algorithms continuously analyze historical data, learning from charging patterns to refine future energy management decisions.
- **Scalability and flexibility:** As fleets expand, cloud-based platforms can easily accommodate additional chargers and vehicles without requiring extensive on-site infrastructure upgrades. Additionally, cloud systems support seamless software updates and remote system enhancements, minimizing the need for manual interventions.
- **Enhanced data accessibility and interoperability:** They can integrate with third-party software, such as fleet management platforms, to optimize vehicle charging based on schedules, driver routes, and vehicle SOC. This interconnected approach ensures that vehicles are charged and ready for deployment when needed.

### Challenges of Cloud-Based Smart Charging

Despite these advantages, cloud-based systems have some limitations.

- **Reliance on an internet connection** is a key challenge—if connectivity issues arise, real-time decision-making may be impacted, potentially disrupting charging operations.
- **Sites that include power plants, energy storage systems, or energy meters** can face integration difficulties without on-site hardware. As a general rule, the more complex the site, the harder it is to rely solely on a cloud-based system. In such cases, a hybrid approach—combining cloud-based intelligence with local controllers—can offer greater reliability and operational efficiency.

<!-- page: 18 -->
## Local (on-site) Smart Charging

On-site energy management systems, such as AmpEdge, operate independently of the cloud or internet, making real-time energy decisions through hardwired communication with chargers. Instead of relying on remote servers, local hardware controllers process data and execute load management directly at the depot. This ensures uninterrupted operations, even in areas with poor or unreliable internet connectivity.

> **Figure 10 — AmpEdge controller for local (on-site) smart charging**
> Type: photograph
> Axes: none
> Series: none
> Values: none
> Notes: Photograph of physical hardware controller installed on a DIN rail.

### Key Advantages of Local Energy Management

- **Resilience and Reliability:** Since all decisions are made on-site, fleet operations remain unaffected by network failures or latency issues. This is particularly beneficial for mission-critical applications where uninterrupted smart charging is essential.
- **Seamless Integration with Microgrids and Renewables:** Local systems work efficiently with on-site solar panels, battery storage, and building energy management systems. These controllers dynamically manage energy flows in real time, making them particularly effective for microgrid applications, where low-latency decision-making is required.

<!-- page: 19 -->
### Challenges of Local Systems

- **Limited Computational Power and Optimization Complexity:** Compared to cloud-based solutions, local controllers generally have lower processing capacity, which limits the complexity of optimization algorithms they can execute.
- **Restricted Remote Monitoring and Data Access:** Local systems typically offer limited API support, remote monitoring capabilities, and data backups. While cloud-based platforms provide extensive data storage and automated backups, on-site controllers rely on simpler, more basic backup mechanisms.

For fleets that require high reliability and local energy management, on-site controllers are an ideal choice. However, for advanced analytics, large-scale optimization, and remote access, a hybrid approach that combines local control with cloud intelligence can offer the best of both worlds.

## Hybrid Approach: Combining Cloud with On-site Controllers

For many fleet operators, the best solution lies in a hybrid approach that integrates both cloud and local power management. The cloud serves as the central data platform, providing real-time alerts, charging reports, and integration with telematics systems like Geotab, Samsara, and Webfleet. This enables fleet managers to remotely optimize energy usage while maintaining complete visibility over operations.

At the same time, the on-site controller ensures local reliability, allowing depots to continue functioning even during network disruptions. By managing power distribution locally and prioritizing energy flows between chargers, solar panels, and battery storage, a local controller like the AmpEdge maximizes efficiency and reduces electricity costs.

While it is often possible to use a cloud-only system if the site does not require very complex smart charging, it is usually not recommended to use an on-site-only system. Systems like AmpEdge always operate in hybrid mode with the cloud, providing advanced remote functionalities and greater flexibility.

<!-- page: 20 -->
> **Figure 11 — Hybrid Smart Charging System (Cloud and On-site)**
> Type: schematic diagram
> Axes: none
> Series: none
> Values: none
> Notes: Shows Cloud System connected to Local Controller, which manages Charging Stations, Solar Energy, Building Load, and Energy Storage.

A hybrid system offers several key benefits:

- **Resilience and Redundancy:** Ensures uninterrupted operations even during internet outages, keeping charging schedules consistent.
- **Optimized Energy Management:** Combines advanced cloud-based optimization with on-site energy balancing to improve efficiency.
- **Seamless Integration with Renewables:** Supports microgrid operations, solar power, and battery storage, reducing dependence on the utility grid.
- **Scalability for Complex Sites:** Ideal for depots with intricate power setups that require both remote oversight and localized control.

# Microgrid solutions for large fleet sites

As fleet electrification scales, large fleets face increasing challenges in securing a reliable and affordable energy supply. The growing size of battery-electric trucks significantly increases energy demand per vehicle, while grid availability continues to decline.

<!-- page: 21 -->
Over the past 12 months, a growing number of companies have been exploring microgrid installations. In contrast to traditional grid-dependent charging sites, microgrids generate a portion of their energy on-site and often store energy temporarily in battery storage systems. By installing on-site power generation and storage, fleets can improve energy reliability while reducing dependency on the grid.

Microgrid solutions typically consist of EV chargers, battery energy storage systems, and solar panels, while maintaining a grid connection for additional flexibility. This configuration enables depots to increase available power capacity without requiring costly and time-consuming utility upgrades.

> **Figure 12 — Possible microgrid setup with EV charging**
> Type: schematic diagram
> Axes: none
> Series: none
> Values: none
> Notes: Shows Optimization block connected to Charging Stations, Solar Energy, Utility Data, Energy Storage, Building Load, and Price Data.

For microgrids to function effectively, charging must be intelligently controlled using generation forecasts and real-time battery storage data. Smart energy management platforms coordinate power distribution between solar generation, stored energy, and direct grid consumption, ensuring fleets maximize efficiency while maintaining reliability. By leveraging predictive analytics and real-time monitoring, fleets can optimize charging schedules, lower energy costs, and ensure a resilient power supply.

While microgrid integration is particularly beneficial for fleet depots in areas with weak grid infrastructure or high energy costs, it can come with large upfront costs and integration complexity. However, as grid infrastructure becomes increasingly constrained, microgrids provide a viable alternative. Since it increases fleet energy independence, reduces exposure to grid outages, and lowers CO₂ emissions, it is likely that the number of microgrid projects will grow significantly.

<!-- page: 22 -->
# Integration with Depot Management and Telematics

The departure time of trucks is the most important factor in fleet operations, but is often overlooked when discussing smart charging. One of the key KPIs for fleets is on-time delivery and vehicle readiness. If a truck is not fully charged, drivers may need to switch to a combustion vehicle or wait until charging is complete, causing operational delays.

The integration of telematics and route data with smart charging is essential to preventing such issues. In other words, smart charging algorithms use planned departure times, live vehicle locations, and SOC data to optimize charging. Once a vehicle enters the depot, a geofence detects its arrival, identifies the need for charging, and can notify the operator. The smart charging algorithm then uses route data to prioritize the right vehicles, increasing power input as needed to ensure they are ready on time.

Fleet and charging operators utilize this approach effectively to minimize vehicle downtime and receive early notifications if a vehicle may not be ready on time. This provides enough time to find alternatives or adjust the energy management setup to ensure smooth operations.

> **Figure 13 — Smart charging software integrated with vehicle and route data**
> Type: schematic diagram
> Axes: none
> Series: none
> Values: none
> Notes: Shows Vehicle Telematics and Route Planning feeding into Software, which communicates via OCPP with EV Charger and Vehicle.

# Real-world Examples of Smart Charging

Smart charging is well beyond theory, and over the past years, many projects have been deployed. To demonstrate the success of charging, we present real-world data and examples of sites supported globally, including the United States, Canada, Europe, Africa, and Latin America.

The two selected examples are fleet depots, each with more than 20 charge point connections (DCFC).

<!-- page: 23 -->
## Depot 1: Overcoming grid constraints and reducing demand charges

The fleet depot experiences significant charging activity both during the day and at night, resulting in high peak power demand if left unmanaged. Most of the depot’s DC fast chargers (DCFC) operate at 100–150 kW. Without route planning, the system relies solely on charging station data for energy management. Given the site’s lack of additional buildings or battery storage and the presence of a stable internet connection, a cloud-based smart charging solution was the most cost-effective approach.

To mitigate demand charges ($15–$25/kW, varying by time of day and season) and TOU pricing, a real-time smart charging algorithm was deployed. This system dynamically distributes available power across the chargers, capping total site demand at approximately 1 MW. Without smart charging, unmanaged charging would push peak demand beyond 1.5 MW, significantly increasing costs.

By implementing smart charging, the fleet reduces peak demand and associated demand charges by 50%, resulting in annual savings of approximately $150,000 per depot—equivalent to $7,500 per charge point connector. Additionally, by avoiding high TOU pricing, the fleet saves an additional $100,000 annually in energy supply costs. In total, the 20-charger site achieves annual savings of $250,000 through intelligent charging optimization.

> **Figure 14 — Load management example - depot 1**
> Type: line chart / dashboard screenshot
> Axes: Time (18:00 to 18:00 next day) vs Power (kW) (0 to 300)
> Series: Max Power (kW), Optimized limit (kW), Power (kW), DR/V2G event
> Values: Live Power: 146.2 kW, Live Utilization: 54%
> Notes: Chart showing managed power profile for Depot 1.

<!-- page: 24 -->
## Depot 2: Avoid peak TOU rates pricing

This depot operates 40 DCFC dispensers at a single site. While there are no grid capacity constraints, the facility is subject to a TOU utility contract with the following rates:

- Peak: $0.45/kWh
- Off-Peak: $0.25/kWh
- Super Off-Peak: $0.20/kWh

To maintain a low total TCO, the fleet operator must minimize charging during peak and off-peak periods—particularly between 4 PM and 9 PM. However, many vehicles return during this window, and without smart charging, they would immediately begin charging at high-cost rates.

By implementing smart charging, the system can curtail charger power until 9 PM, shifting the majority of charging to lower-cost super off-peak hours. This optimization reduces monthly energy costs by 15%–55%, translating to annual savings of $60,000–$217,800 for the fleet.

> **Figure 15 — Peak shifting example - depot 2**
> Type: line chart / dashboard screenshot
> Axes: Time (16:00 to 08:00) vs Power (kW) (0 to 1,600)
> Series: Max Power (kW), Optimized limit (kW), Power (kW), DR/V2G event
> Values: [visual power curtailment curve dropping to near zero between 16:00 and 21:00]
> Notes: Chart showing power shifted away from peak hours for Depot 2.

<!-- page: 25 -->
# The Future of Smart Charging for Electric Trucks

As the electrification of trucking continues to expand, advances in charging technology, regulatory frameworks, and grid integration will shape the future of smart charging. Key developments will address the need for faster charging, vehicle-to-grid (V2G) capabilities, and increased utility oversight.

## Megawatt Charging Systems (MCS)

Megawatt Charging Systems (MCS) will be essential for high-power depot applications, offering charging speeds of up to 1 MW per port. Currently, most chargers are limited to 360 kW, while most vehicles charge at speeds below 250 kW. As charger and vehicle technologies advance, charging times could be cut in half, improving operational efficiency for long-haul electric trucks.

However, MCS adoption presents challenges. High-speed charging generates significant heat, which can accelerate battery degradation. To address this, advancements in battery technology will be required to manage thermal loads effectively. Additionally, the power demand for MCS chargers is substantial, increasing the complexity and cost of installation. Many depots may opt for a hybrid setup, with only a few MCS chargers alongside standard DC fast chargers to balance cost and grid impact.

## Vehicle-to-Grid (V2G) Integration

V2G technology allows electric vehicles to discharge energy back into the grid, providing demand-side flexibility and potential revenue streams for fleet operators. While promising in theory, V2G has yet to reach scale due to technological, regulatory, and economic barriers.

In the U.S., only a handful of V2G projects exist, and most chargers lack universal compatibility with different vehicle types. Furthermore, truck manufacturers often restrict V2G usage due to concerns over battery degradation and warranty implications. The high installation costs of V2G infrastructure also pose a significant challenge, with limited financial incentives currently available. However, pilot projects with electric school buses have demonstrated promising results due to their long dwell times, potentially paving the way for future V2G adoption in specialized fleet applications.

<!-- page: 26 -->
## Certification and Regulatory Trends

As utilities seek greater control over energy demand, they are increasingly requiring charging sites to meet certification standards. This is particularly relevant for depots planning to oversubscribe their charging infrastructure, as mismanagement could impact grid reliability. Certifications such as UL 60730-1, UL 3141, and IEEE 2030.5 are being considered to standardize safety, interoperability, and power management practices.

While no unified national standard exists, utilities are tightening requirements, mandating deeper integration of charging data into their operational systems. As regulatory frameworks evolve, fleets will need to ensure compliance with utility requirements to avoid service disruptions and optimize grid interactions.

<!-- page: 27 -->
# Conclusions

The electrification of truck fleets is no longer a distant vision—it is an active transformation underway. This report, developed by NACFE and Ampcontrol, highlights both the remarkable progress made and the substantial work that lies ahead. The deployment of electric trucks at scale is technically feasible, operationally proven, and increasingly necessary to meet sustainability and decarbonization goals. Yet scaling this transition demands more than just new vehicles—it requires a rethinking of how to design, power, and manage fleet infrastructure.

From grid capacity constraints and utility delays to charging complexity and rising energy costs, fleets face real and persistent barriers. However, the path forward is clear: **smart charging and energy management must become foundational elements of every electric fleet strategy.** These tools are no longer optional add-ons—they are instead essential to unlocking cost savings, improving reliability, and ensuring long-term scalability.

The insights and case studies in this report demonstrate that, with the right planning, technology, and partnerships, fleets can successfully navigate the transition. Real-world results demonstrate that intelligent energy management reduces costs, improves uptime, and avoids grid bottlenecks, making electric fleets not only possible but also practical.

As the industry evolves, we encourage all stakeholders—fleet operators, utilities, regulators, and technology providers—to collaborate, innovate, and act with a sense of urgency. The road ahead is complex, but it is also full of opportunity. By working together, we can build the infrastructure and systems needed to power the next generation of freight and move one step closer to a cleaner, more resilient transportation future.

<!-- page: 28 -->
# About Ampcontrol

Ampcontrol is a leading provider of energy management and EV charging optimization solutions, designed to streamline the deployment and operation of charging infrastructure. Its innovative software and hardware solutions cater to diverse requirements, enabling seamless integration, real-time monitoring, and intelligent management of EV charging networks. Ampcontrol’s Energy Management system optimizes energy usage across diverse sites, accommodating unique depot constraints such as transformers, grid connections, energy tariffs, and vehicle departure schedules. The system enables real-time monitoring and optimization of both chargers and vehicles, integrating seamlessly with OEM telematics systems or third-party telematics devices, requiring no additional hardware installation.

[www.ampcontrol.io](https://www.ampcontrol.io)

# About NACFE

The North American Council for Freight Efficiency (NACFE) works to drive the development and adoption of efficiency enhancing, environmentally beneficial, and cost-effective technologies, services, and operational practices in the movement of goods across North America. NACFE provides independent, unbiased research, including Confidence Reports on available technologies and Guidance Reports on emerging ones, which highlight the benefits and consequences of each, and deliver decision-making tools for fleets, manufacturers, and others. NACFE partners with RMI on a variety of projects including the Run on Less demonstration series, electric trucks, emissions reductions, and low-carbon supply chains.

[nacfe.org](https://nacfe.org)

# About Run on Less by NACFE

Run on Less is a biennial demonstration showcasing advancements in freight efficiency. The event takes place over the course of three weeks and has fleets across North America moving freight in their normal operations to show how it is possible to operate efficiently with today’s technologies. Each truck in the Run is outfitted with a telematics device allowing NACFE to track key metrics. The first Run took place in 2017 and featured seven fleets that averaged 10 MPG in long-haul routes over the course of the Run. Run on Less Regional, held in 2019, featured 10 fleets that averaged 8.4 MPG in regional haul routes during the Run. Run on Less – Electric, held in 2021, featured 13 trucks operating electric vehicles on real routes carrying real freight. Run on Less – Electric DEPOT featured 10 fleets with 15 or more electric trucks at a depot.

[runonless.com](https://runonless.com)

<!-- page: 29 -->
www.ampcontrol.io
contact@ampcontrol.io
Ampcontrol Technology, Inc.
New York City

www.nacfe.org
www.runonless.com

Copyright © 2025 Ampcontrol Technologies, Inc. and NACFE. All rights reserved.
