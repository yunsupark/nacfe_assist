<!-- page: 1 -->
Cover page / Title page of the report: "TRUCKING EFFICIENCY CONFIDENCE REPORT: PROGRAMMABLE ENGINE PARAMETERS"

<!-- page: 2 -->
Acknowledgements page.

<!-- page: 3 -->
Table of Contents (Part 1).

<!-- page: 4 -->
Table of Contents (Part 2).

<!-- page: 5 -->
Executive Summary page 1.

<!-- page: 6 -->
Executive Summary page 2.

A sample of programmable parameters related to idle reduction
| Feature/Parameter | Range | Default |
| --- | --- | --- |
| Idle Engine Speed — Parameter | 500 – 800 RPM | 600 RPM |
| Idle Shutdown — Feature Option | Enable/Disable | Disable |
| Idle Shutdown Timer — Parameter | 2 – 1,440 minutes | 60 minutes |
| Idle Shutdown Manual Override — Feature Option | Enable/Disable | Disable |
| Idle Shutdown in PTO — Feature Option | Enable/Disable | Disable |
| Idle Shutdown PTO Load Override — Parameter | 0 – 100% | 10% |
| Idle Shutdown Ambient Air Temperature Override — Feature Option | Enable/Disable | Disable |
| Idle Shutdown Intermediate Ambient Air Temperature — Parameter | 0 – 120⁰ F | 60⁰ F |
| Idle Shutdown Hot Ambient Air Temperature — Parameter | 0 – 120⁰ F | 85⁰ F |
| Idle Shutdown Cold Ambient Air Temperature — Parameter | 0 – 120⁰ F | 30⁰ F |
| Idle Shutdown Hot Ambient Automatic Override — Feature Option | Enable/Disable | Disable |
| Idle Shutdown Manual Override Inhibit Zone — Feature Option | Enable/Disable | Disable |

<!-- page: 7 -->
Executive Summary page 3.

<!-- page: 8 -->
Executive Summary page 4.

> **Figure — Confidence Matrix for Electronic Engine Parameters**
> Type: Matrix / Scatter plot
> Axes: Confidence Rating (Low to High) vs Payback in Years (1 to 4)
> Series: Investment cases
> Values: Invest in Testing: low/1, medium/2; Quickly Invest in Testing: medium/1; Great case for adoption! No testing required: high/1; Consider Testing: low/2, medium/3; Invest in technology: high/2; Wait for Next-Gen Products: low/3; Share Data with Industry: high/3
> Notes: Programmable Parameters highlighted in green in the top right quadrant (High Confidence / 1 Year Payback).

<!-- page: 9 -->
Executive Summary page 5.

<!-- page: 10 -->
# 1 Introduction

> **Figure 1 — US Annual Diesel Fuel Prices**
> Type: Line chart
> Axes: Year (2003 to 2014) vs Annual Avg Diesel ($) ($0.00 to $4.50)
> Series: Annual Avg Diesel
> Values: 2003: $1.51, 2004: $1.81, 2005: $2.40, 2006: $2.71, 2007: $2.89, 2008: $3.80, 2009: $2.47, 2010: $2.99, 2011: $3.83, 2012: $3.97, 2013: $3.92, 2014: $3.82
> Notes: Source: US EIA Jan 8, 2015

<!-- page: 11 -->
> **Figure 2 — Trucking Operational Cost**
> Type: Stacked bar chart
> Axes: Year (2011, 2012, 2013) vs Cost per mile ($)
> Series: Fuel, Truck/Trailer, Repair & Maint, Other Operational, Driver Wages & Benefits
> Values: 2011 - Fuel: 0.59, Truck/Trailer: 0.19, Repair & Maint: 0.15, Other Operational: 0.16, Driver Wages & Benefits: 0.61; 2012 - Fuel: 0.64, Truck/Trailer: 0.17, Repair & Maint: 0.14, Other Operational: 0.15, Driver Wages & Benefits: 0.54; 2013 - Fuel: 0.65, Truck/Trailer: 0.16, Repair & Maint: 0.15, Other Operational: 0.15, Driver Wages & Benefits: 0.57
> Notes: Source: American Transportation Research Institute 2014.

<!-- page: 12 -->
> **Figure 3 — Fleet Fuel Study Participants**
> Type: Photograph / Logo collage
> Axes: None
> Series: None
> Values: None
> Notes: Logos include Con-way, Bison Transport, C.R. England, Ryder, Werner Enterprises, UPS, Frito Lay, Paper Transport, Challenger Motor Freight, Schneider National, NACFE.

## 1.1 Trucking Efficiency’s Confidence Reports

<!-- page: 13 -->
## 1.2 Methodology

<!-- page: 14 -->
### 1.2.1 Preliminary study questions used in study team interviews

<!-- page: 15 -->
# 2 Overview of Electronic Engine Parameters

<!-- page: 16 -->
> **Figure 4 — Location of Engine Control Modules**
> Type: Photograph
> Axes: None
> Series: None
> Values: None
> Notes: Highlights ECMs on cutaway display engines.

> **Figure 5 — Programmable Features on Personal Electronics**
> Type: Table / Photograph collage
> Axes: None
> Series: Smart Phones, Laptops, Car audio
> Values: Smart Phones: Ring Tones, Screen locks, Wallpaper, Language, Screen Brightness, Camera flash on/off/auto, Block call times...; Laptops: Mouse Button (Right handed/left handed), Wallpaper, Screen Savers...; Car audio: Loudness on/off, Speed sensitive volume, Radio presets, Equalizer selection...
> Notes: None

<!-- page: 17 -->
## 2.1 Categories of Fuel Economy Parameters

<!-- page: 18 -->
| PARAMETER GROUP | SAMPLE PARAMETERS IN THIS GROUP |
| --- | --- |
| Vehicle Speed Limits | • Accelerator Maximum Vehicle Speed<br>• Maximum Cruise Control<br>• Road speed governor droop<br>• Gear down protection |
| Vehicle Configuration Information | • Tire Revolutions Per Mile<br>• Rear Axle Ratio<br>• Transmission Top Gear Ratio(s)<br>• Vehicle Speed Sensor Pulses Per Revolution |
| Engine Speed (& Torque) Limits | • Progressive Shifting (RPM allowed by vehicle MPH ranges)<br>• Load Based Speed Control<br>• Vehicle Acceleration Control |
| Idle Reduction | • Number of minutes before Idle Shutdown Timer turns off the engine<br>• Hot Ambient Air Temperature<br>• Cold Ambient Air Temperature<br>• PTO override |
| Driver Rewards | • MPG Thresholds For Rewards<br>• Idle Time Thresholds For Rewards<br>• Additional MPH Allotted For Attaining Thresholds |
| Miscellaneous MPG Related Features | • Service Brake Enable Brake Activation<br>• Engine Brake Cruise Control Activation<br>• Upshift Recommended Gauge Cluster Light Enable (Manual Transmissions) |

> **Figure 6 — Sample Parameters from each Category**
> Type: Table (transcribed above)
> Axes: None
> Series: None
> Values: None
> Notes: None

<!-- page: 19 -->
### 2.1.1 Vehicle Speed

<!-- page: 20 -->
*(Continuation of section 2.1.1)*

<!-- page: 21 -->
> **Figure 7 — Vehicle Speed**
> Type: Diagram (Speedometer gauge illustration)
> Axes: Miles Per Hour (0 to 75)
> Series: Speed settings
> Values: 62 = Accelerator Maximum Vehicle Speed, 63 = Cruise Control Maximum Vehicle Speed, 65 = Accelerator Maximum Vehicle Speed 62 + Accelerator Lower Droop 3, 67 = Pass Smart Maximum Vehicle Speed, 68 = Accelerator Maximum Vehicle Speed 62 + Driver Reward Best Speed Reward 6, 70 = Global Maximum Vehicle Speed
> Notes: NOTE: This is an example for illustrative purposes only: Many variations on this theme can be programmed

### 2.1.2 Vehicle Configuration Parameters

<!-- page: 22 -->
> **Figure 8 — HD Diesel Engine "Sweet Spot" for Fuel Economy**
> Type: Engine fuel map contour chart / Diagram
> Axes: Spd_Engine (rpm) (800 to 2000) vs % LOAD (25 to 100)
> Series: Typical Diesel Engine Fuel Map, Volvo XE13 & Mack Super Econodyne
> Values: None (contour map)
> Notes: 1. Cruise operating point 2010 Baseline; Volvo XE13 & Mack Super Econodyne — Down-speeded engine, enabled by integrated AMT & high torque yields 2-3% FE; 1-2: chassis & trailer improvements reduce load; 2-3 downspeeding improves efficiency; 3-4 downsizing increases percent load; RESULT: Major improvement in vehicle fuel consumption with same engine efficiency.

<!-- page: 23 -->
> **Figure 9 — Move to Faster Rear Axles**
> Type: Bar chart
> Axes: Axle Ratio (2.47 to 7.17) vs Percent of Total Model Sales (0% to 25%)
> Series: 14x (Today), 14x (2011)
> Values: 2.47: approx. 3% (Today) / 0% (2011), 2.64: approx. 19% (Today) / approx. 3% (2011), 2.79: approx. 5% / 0%, 2.93: approx. 5% / 0%, 3.07/3.08: approx. 8% / approx. 4%, 3.25/3.21: approx. 9% / approx. 7%, 3.36: approx. 2% / approx. 1%, 3.42: approx. 17% / approx. 13%, 3.55/3.58: approx. 22% / approx. 18%, 3.70/3.73: approx. 11% / approx. 11%, 3.9: approx. 2% / approx. 3%, 4.11: approx. 1% / approx. 1%, 4.33: approx. 0.5% / approx. 1%, 4.63: approx. 0.5% / approx. 2%, 4.88: approx. 0.5% / approx. 2%, 5.29: approx. 0.5% / approx. 2%, 5.86: approx. 0.5% / approx. 1%, 6.14: 0% / 0%, 6.43: 0% / 0%, 6.83: 0% / 0%, 7.17: 0% / 0%
> Notes: Meritor Highway Tandem Axle Sales by Ratio - Current vs. 2011

### 2.1.3 Engine Speed (and/or Torque) Parameters

<!-- page: 24 -->
> **Figure 10 — Progressive Shift Points**
> Type: Line chart (Progressive Shift Sawtooth)
> Axes: Vehicle Speed (MPH) (0 to 90) vs Engine Speed (RPM) (0 to 1800)
> Series: High Load, Low Load, RSL
> Values: None (stylized sawtooth schematic)
> Notes: RSL indicates Road Speed Limit.

### 2.1.4 Idle Reduction Parameters

<!-- page: 25 -->
| Feature/Parameter | Range | Default |
| --- | --- | --- |
| Idle Engine Speed — Parameter | 500 – 800 RPM | 600 RPM |
| Idle Shutdown — Feature Option | Enable/Disable | Disable |
| Idle Shutdown Timer — Parameter | 2 – 1,440 minutes | 60 minutes |
| Idle Shutdown Manual Override — Feature Option | Enable/Disable | Disable |
| Idle Shutdown in PTO — Feature Option | Enable/Disable | Disable |
| Idle Shutdown PTO Load Override — Parameter | 0 – 100% | 10% |
| Idle Shutdown Ambient Air Temperature Override — Feature Option | Enable/Disable | Disable |
| Idle Shutdown Intermediate Ambient Air Temperature — Parameter | 0 – 120⁰ F | 60⁰ F |
| Idle Shutdown Hot Ambient Air Temperature — Parameter | 0 – 120⁰ F | 85⁰ F |
| Idle Shutdown Cold Ambient Air Temperature — Parameter | 0 – 120⁰ F | 30⁰ F |
| Idle Shutdown Hot Ambient Automatic Override — Feature Option | Enable/Disable | Disable |
| Idle Shutdown Manual Override Inhibit Zone — Feature Option | Enable/Disable | Disable |

> **Figure 11 — A Partial List of Idle Reduction Parameters**
> Type: Table (transcribed above)
> Axes: None
> Series: None
> Values: None
> Notes: None

<!-- page: 26 -->
> **Figure 12 — Idle Shutdown Ambient Air Temperatures**
> Type: Diagram (Thermometer illustration)
> Axes: Temperature scale (0 to 100)
> Series: Hot Ambient Air Temperature, Cold Ambient Air Temperature, Idle Shutdown Timer
> Values: Hot Ambient Air Temperature: Above the Hot Ambient Air Temperature, the engine can idle to provide air conditioning (H at 70); Cold Ambient Air Temperature: Below the Cold Ambient Air Temperature, the engine can idle to provide heat (C at 20); Idle Shutdown Timer: Between the 2 temperature set points the Idle Shutdown Timer (if enabled) will stop the engine after the selected time period.
> Notes: None

### 2.1.5 Driver Reward Parameters

<!-- page: 27 -->
### 2.1.6 Miscellaneous MPG Parameters

## 2.2 Programming Methods and Tools

<!-- page: 28 -->
> **Figure 13 — Participants in Engine Parameter Optimization**
> Type: Diagram (Process flowchart with character illustrations)
> Axes: None
> Series: Fleet Purchasing, Fleet Maintenance, Drivers, Dealer Service, Dealer Sales, OEM Assembly Plant
> Values: None
> Notes: Continuous loop engine parameter optimization.

<!-- page: 29 -->
### 2.2.1 OEM Software Tools

### 2.2.2 Manufacturer Parameter Name Comparison Tool

<!-- page: 30 -->
## 2.3 Legislation and Programmable Parameters

### 2.3.1 Idle Reduction

### 2.3.2 Vehicle Speed

## 2.4 Perspectives on Future Programmable Parameters

<!-- page: 31 -->
> **Figure 14 — GPS Technologies for Vehicle Speed**
> Type: Diagram / Illustration (Truck on winding highway with terrain controls overlay)
> Axes: None
> Series: Speed, Brake Speed, Set Speed, Intelligent Controls, Conventional cruise, Roll, Reduce Speed, Hold Gear, Accelerate, Terrain
> Values: None
> Notes: Intelligent Controls leverage vehicle improvements to achieve further fuel efficiency gains. Auxiliary Integration maximizes use of free energy. Powertrain Management minimizes fuel use.

<!-- page: 32 -->
# 3 Fuel Economy Benefits

<!-- page: 33 -->
> **Figure 15 — Fuel Economy via Parameter Setting**
> Type: Pyramid / Tiered diagram
> Axes: None
> Series: OPTIMIZED (5-8%), PARAMETERS SET (3-5%), USE FACTORY SETTINGS ONLY
> Values: None
> Notes: MPG scale on left.

## 3.1 Additional Benefits of Optimized Parameters

# 4 Challenge of Complexity

<!-- page: 34 -->
## 4.1 Understanding Parameters

### 4.1.1 The large number of parameters

### 4.1.2 Interrelations between multiple parameters, and/or between parameters and other systems on the truck

<!-- page: 35 -->
## 4.2 Selecting and Ordering Parameters

### 4.2.1 Variation in OEM terminology and precise functionality

### 4.2.2 Variation in ordering tools

<!-- page: 36 -->
### 4.2.3 Incorrect initial parameters settings

## 4.3 Maintaining Parameters

### 4.3.1 Lack of telematics and variation in service tools

<!-- page: 37 -->
### 4.3.2 Negative reactions from drivers

<!-- page: 38 -->
### 4.3.3 Parameter records maintenance

> **Figure 16 — Sample Template of Parameters by Categories**
> Type: Schematic / UI representation
> Axes: None
> Series: Engine Brand A (Template A1)
> Values: Vehicle Speed Limits, Vehicle Configuration, Engine Speed Limits, Idle Reduction, Driver Rewards, Miscellaneous MPG
> Notes: None

<!-- page: 39 -->
tractors. This would likely require them to alter the parameters within the Idle Reduction category, doing things like raising the Hot Ambient Air Temperature parameter lowering the Cold Ambient Air Temperature one, in order to ensure that more of the HVAC burden is shunted to the new APU rather than placed on the main engine. Another example would be for a fleet transitioning to Automated Manual Transmissions, which would necessitate changing their parameters from the Engine Speed category, in order to place more responsibility for optimized shifting onto the AMT. Figure 17 is an example of what the full suite of templates for a fleet who owns two or three models of engines from three different OEMs could look like, though an actual template would go into much more detail regarding specific parameters in each of these general categories and the settings that they should be assigned.

> **Figure 17 — Multiple Templates When Adding a Few Features**
> Type: schematic
> Axes: none
> Series: none
> Values: none
> Notes: Displays a matrix of templates across Engine Brand A (Templates A1, A2, A3), Engine Brand B (Templates B1, B2, B3), and Engine Brand C (Templates C1, C2, C3), each containing categories such as Vehicle Speed Limits, Vehicle Config, Engine Speed Limits, Idle Reduction (+ APU / + Bt HVAC), Driver Rewards, and Misc. MPG+.

One very large fleet reported that they now have 50 templates for one engine OEM and 30 templates for a second. Additionally, one of the engine OEMs reported that they had recently worked with a large fleet to consolidate down to just 28 templates for their engines, with significant differences still required from one generation of their engines to another.

An oft-overlooked challenge occurs when fleet acquisitions and/or mergers require the integration of another fleet’s programmable parameters. Given the sheer number of programmable parameters it is unlikely that any two fleets have many of their parameters set to the same values. Accelerator Maximum Vehicle Speed is obviously one of the first parameters to standardize in such a situation, but beyond that fleet’s will need to fully understand the specifications of these new trucks, from vehicle ages to rear axle

<!-- page: 40 -->
ratios to idle reduction systems, to make sure that parameters from progressive shifting to gear down protection to minimum and maximum temperatures and more can be set correctly.

Another reason why good records keeping on parameters is so critical is that although instances of complete or even partial ECM failure are uncommon, when they do occur, all of that engine’s programmable parameters must be reentered. In a best case scenario, it may have been possible to download the parameters from the ECM before replacement. In a worst case scenario, the parameters will need to be reset from scratch, which will require either good fleet records on how the parameters are to be set, or an identical vintage vehicle from which to copy parameter settings. Fleets that implement strategies for recording their parameter settings as they change over time will avoid problems in these instances. While many fleets retain vehicle production records, using these as a reference to replace failed ECM data will only work if none of the truck’s parameter settings have been modified in the interim.

Finally, good parameters record keeping is a guard against tampering. Fleets need to also be vigilant for other tampering beyond the programmed parameter values. It is possible to tamper with programmable parameters by changing the sensors which engine electronics use to monitor the performance and external environment of the vehicle. It is possible to move the ambient air temperature sensors into a position, say under the hood or near the exhaust, where excessive heat will trick the engine controls to allow unlimited idling. It is also possible to tamper with a vehicle’s speed sensors to alter the signal the engine uses to determine the road speed of the vehicle.

Advancements in password protection for programmable parameters have largely put a stop to tampering, but some fleets reported that they still struggle with it. Parameters governing the Accelerator Maximum Vehicle Speed are most commonly changes, as drivers may wish to go faster than the fleet allows. Several fleets reported spot checking this setting to ensure it had not been altered. Fleets should be aware though that there are other ways to subvert these limits which would not be detected by a simple spot check of the parameter itself. For instance, it may be possible to alter other parameters, such as the Driver Rewards parameters, to enable the vehicle to go faster than the fleet’s maximum speed. Therefore changes to any of the parameter settings should require both a programming tool and the fleet’s password, which should be kept as confidential as possible.

## 5 Interview Perspectives: Original Equipment Manufacturers

The study team’s conversations with the engine OEMs provided substantial insight into why programmable parameters are so complex to create and maintain. First of all, heavy trucks can last for a long time, meaning there are still trucks from the 1980s equipped with the very first sets of electronic controls in operation and requiring occasional support today. Volvo determined that they currently have 12 variations of ECMs in the field, Cummins reported 10 and Detroit and Navistar both reported seven generations/variations. Available engine parameters have not only grown in number in the interim years, but in some cases have also changed names or evolved entirely, as truck systems become more

<!-- page: 41 -->
interrelated. Within the engine OEMs, it appears that engineers and marketing personnel have the most control over the names given to new parameters.

The challenge for an engine OEM goes far beyond the development and validation of the parameters within the ECM itself. They must also create the sales and marketing materials that explain the parameters to dealers and fleets, the software tools used to enter parameters when a truck is sold, and the service/diagnostic tools and supporting diagnostic troubleshooting materials for the life of the truck. They may even need to play some role in driver training, as this is tightly intertwined with the operational functionality of some of the programmable parameters.

Starting on the front end with sales, the study team found some very different approaches to handling programmable parameters among the engine and vehicle OEMs. The Cummins PowerSpec software is available at no cost for anyone to download to their computer, and this software has detailed definitions of the parameters for all current Cummins engines as well as some of the more recent past generations of Cummins engines. When a PowerSpec installation is licensed via Cummins, the computer on which it is downloaded becomes capable of programming a Cummins engine via an additional adaptor cable. At this point in time, however, none of the vehicle OEMs can download parameters directly from the PowerSpec tool into their vehicle sales ordering tool. And the vertically OEM integrated sales software systems are not publically available so little could be gleaned on these systems for this report.

The study team asked Engine OEMs to gauge which actors along the value chain of a truck’s sale and operation seemed to possess a detailed knowledge of programmable parameters. They reported that company distribution centers and national account managers appear to be the last of highly trained parameter experts on the sales side. They also reported that detailed knowledge of parameters is frequently stronger on the service channel side, since those personnel are more likely to have the actual equipment to program a vehicle and therefore experience programming parameters. Obviously there are variations in understanding throughout all companies, so each fleet will need to seek the best support they can find. Each OEM appears to have at least someone in an “Applications Engineering” role who can make customer visits to optimize parameter settings post-purchase. But such teams were minimally staffed and really only have the bandwidth to support a small number of customer fleets. As more fleets investigate and ultimately demand the optimization of their engine parameters, it is the study team’s belief the OEMs will respond by increasing the staff resources devoted to this.

One OEM reported they offer live on site training classes in parameter settings and service tools but require at least eight participants to cover travel costs. This typically requires the dealer to invite fleet personnel and others to get enough students.

Overall, each of the OEMs that agreed to discuss parameters with the study team talked about their success with customers in the field, and some of the results cited in Chapter 3’s section on “Fuel Economy Testing” may be seen as proof of their success.

<!-- page: 42 -->
## 6 Interview Perspectives: Dealership Sales Staff

This is the first time that Trucking Efficiency has consulted directly with dealer sales personnel to get their insights for a Confidence Report.

The salesperson at the dealership is an often missed player in conversations on the adoption of new technologies, as they do in fact play a role in the process. Salespeople must understand new technologies well enough to explain their operation to fleets, and more importantly to communicate their value. They must also learn how to correctly specify the technology for inclusion on a new truck order.

A common thread heard in the interviews with deal sales personnel was that programmable parameters are a particularly challenging technology for which to steer fleets in the right direction due the sheer numbers of terms given to those settings by the various OEMs. It was also brought up several times that fleet customers have been known to work with the dealer salesperson to select the right vehicle drivetrain for a given maximum operating speed that they have decided to program into their engines, but then upon delivery will set the maximum vehicle speed higher than what was previously discussed, potentially decreasing the overall performance of their truck.

The dealer sales personnel all reported to the study team that the most common parameters such as Accelerator Maximum Vehicle Speed, Cruise Control Maximum Vehicle Speed, and Idle Shutdown Timer were understood by their customers already. On the other hand, they find fleets to be generally confused in conversions around newer or more obscure parameters, such as Droop Settings, among many others.

When asked how they set the programmable engine parameters in a new vehicle order for an existing fleet customer, half of the sales people responded that would contact the customer to make sure they had the most recent settings. Other responses included that they would use the settings from the fleet’s previous order, or that they would just using the default settings due to the limited functionality of their sales tools. In this last case, the responsibility for optimizing the parameters would fall on the service department. Two of the sales people reported that they had different levels of programming capabilities depending on whether the engine they were selling was their own vertically integrated engine (in which case they had more parameters they could set) or was the Cummins engines (which has fewer parameters for them to set). Interestingly, in both cases the dealership ordering software could not take a customer’s password information and set the password at the factory. This shortcoming places responsibility to set the password at the dealership for all new vehicles.

Only one of the sales people could actually change parameter settings on a real truck themselves. All of the others relied completely on their service department teammates to handle all customer parameter programming needs.

Three-fourths of the sales people have worked with a customer on optimizing the parameter settings on a vehicle and being aware of the performance results. Anecdotally, one interview respondent told of a fleet vehicle that was only getting 4 MPG due to the driver not being in top gear at highway speeds, and achieving and improvement to 6.5 MPG simply by programming the Gear Down Protection

<!-- page: 43 -->
parameter. Another related having programmed the Progressive Shifting parameters for a fleet and thereby improving fuel economy by 0.2 MPG. And a third reported experiencing 0.5 MPG improvements simply optimizing the parameters.

## 7 Interview Perspectives: Fleets

Trucking Efficiency conducted confidential, over-the-phone interviews with nine NACFE-affiliated large fleets, all of whom had significant experience with the optimization of programmable engine parameters. Insights from medium and smaller fleets were garnered via an online collaboration with the Michelin Fleet Forum; forty-five fleets participated in that survey, of which forty-one had first-hand knowledge of parameter settings and processes.

All surveyed fleets were asked about their respective use of fuel economy related programmable engine parameters. Fleets in the Michelin Fleet Forum with less than 50 vehicles were considered to be the “small fleets” and any fleet with 50 or more vehicles was classified as a “large fleet.” Fleets with NACFE affiliation who granted a live phone interview, all of which were large fleets, are referred to as “NACFE fleets.” Figure 18 displays the percentage of each group that is using the various programmable parameters related to fuel economy.

> **Figure 18 — Parameter Use**
> Type: bar chart
> Axes: Parameter Name vs Percentage Use (%)
> Series: NACFE (Phone), Large (50+), Small (0-49)
> Values: 
> Maximum Road Speed: NACFE approx. 100%, Large approx. 80%, Small approx. 40%
> Maximum Cruise Speed: NACFE approx. 100%, Large approx. 90%, Small approx. 40%
> Road Speed Droop: NACFE approx. 78%, Large approx. 25%, Small approx. 10%
> Cruise Control Droop: NACFE approx. 68%, Large approx. 30%, Small approx. 13%
> Gear Down Protection: NACFE approx. 90%, Large approx. 50%, Small approx. 10%
> Progressive Shifting: NACFE approx. 78%, Large approx. 50%, Small approx. 26%
> Load Based Speed Control: NACFE approx. 55%, Large approx. 20%, Small approx. 5%
> Idle Shutdown Timer: NACFE approx. 100%, Large approx. 75%, Small approx. 40%
> Unlimited Idling Ambient Temperature Limits: NACFE approx. 88%, Large approx. 10%, Small approx. 5%
> Engine Brake Upon Service Brake Activation: NACFE approx. 45%, Large approx. 20%, Small approx. 10%
> Vehicle Acceleration Control: NACFE approx. 34%, Large approx. 36%, Small approx. 5%
> Driver Rewards: NACFE 0%, Large approx. 21%, Small approx. 19%
> Notes: Source notes printed on figure or legend indicating categories.

<!-- page: 44 -->
As shown, all three groups self-reported using the Accelerator Maximum Vehicle Speed, Cruise Control Maximum Vehicle Speed, and Idle Shutdown Timer parameters at the high rates, though the NACFE fleets use these three parameter groups more than twice as often as the small fleets.

The NACFE fleets in fact implement the majority of these parameters at very high percentages. Interviews reveled that their reasons for reporting lower adoption rates of a few of the parameters were generally either:
1. The parameter not being available from all OEMs (Load Based Speed Control and Engine Brake Upon Service Brake Activation both fit this category)
2. There are other features that minimize the need for the parameters (for examples AMTs can supplant Progressive Shifting)

The Driver Rewards parameters were not utilized by any of the NACFE fleets spoken to for this report. The general consensus among these fleets was that offering even the best drivers higher speeds was counterproductive to an overall pursuit of optimal fuel economy and fuel efficient driving techniques.

The features with the highest deviation between the NACFE fleets and the smaller fleets are the Unlimited Idling Ambient Temperature Limits. Unfortunately, since the Michelin Fleet Forum participants reported via an on-line survey there was no chance to ask follow-up questions to understand why their usage of these parameters is so low.

### 7.1 NACFE Fleet Interviews

Nine large fleets, several of whom are NACFE members, committed their time to participate in a personal phone interview with the study team on programmable parameters.

These fleets were asked about the mix of each of the different class 8 engines within their fleet, so that if anything irregular or outstanding surfaced in the findings the study team could check to see if a certain brand of engine might behind that outlier. The question about engine mix revealed that nearly all of the fleets were operating at least three different brands of engines simultaneously, and that some had four or more – one fleet reported having a full seven brands of engines in operation. It follows, and was confirmed in the interviews, that these fleets are experiencing a good deal of confusion around the different programmable parameter features and terminology offered by each of the engine OEMs, as no two are alike. One fleet reported specifically that they do not use a particular feature because not all of their engines have that feature, and they do not want a discrepancy in a function, such as how the engine brake engages in one truck compared to the other, to cause confusion for any of their drivers.

Seven of the nine fleets stated that making any changes to their programmable parameter settings was either difficult or extremely difficult. One fleet that relies on contract maintenance support reported that it took 12 months to get all of their vehicles programmed with a new maximum vehicle speed limit. Other challenges that were noted included:
* It is extremely difficult to physically connect with thousands of trucks
* Not all service locations have a service person that is software knowledgeable

<!-- page: 45 -->
Only one of the fleets reported wanting to have all of their trucks, whatever the model or engine, set to the same parameters. This would not be a good choice for most fleets, especially those with a range of truck model years in their fleet, as newer trucks tend to have different optimal engine speeds and therefore different transmission gear ratios and rear axle ratios.

Some fleets reported developing different parameter templates to meet other differences among their trucks, such as:
* Setting team vehicles with a higher maximum road speed than single driver tractors
* Setting idle shutdown temperature limits differently, to match the performance capabilities of different types of idle reduction systems (fuel operated heater only, APU, Battery HVAC, etc.)

When asked about the challenges that can arise after fleet mergers and acquisitions, there was general agreement that it was extremely difficult to get any newly acquired vehicles programmed to match the rest of the fleet. Not only do all of the challenges cited in the last two paragraphs arise, but new vehicles may have features such as gearing combination not seen in the rest of the fleet, so brand new templates will need to be developed which will allow those trucks to meet fleet fuel economy goals. Two fleets admitted that this situation was found to be so difficult that they had either sold the newly acquired vehicles out of the fleet or just left them alone without any changes.

When asked about their fleet’s procedures for maintaining records of their electronic engine parameters the answers varied widely. One fleet keeps a 3-ring binder on each and every vehicle which includes the programmable parameter settings. Others had spreadsheets, and others had nothing documented at all. Many of the fleets do attempt to monitor that the settings on vehicles are as they should be. Some spot check upon new vehicle delivery, after finding that at least two vehicle OEMs completely missed programming the parameters at the assembly plant. Others spot check parameters during service work or preventative maintenance. Fleets also use telematics and driving reports from the engines to spot drivers that have been operating outside of their parameter settings. Most fleets state that they now change these fuel economy parameters less than once per year.

All of the fleets expressed a strong desire for the industry to develop a telematics system that could at least report back on parameter settings, ideally be able to reprogram the parameters remotely, as needed. According to fleets, such technologies would offer benefits including:
1. Enabling spot checks and facilitating of wholesale changes
2. Saving shop bay time as well as technician time, both of which are more expensive compared to a business analyst in an office using telematics
3. Making changes to parameters would be much easier and faster, with no shop time required
4. Ability to adjust settings as the driver gains experience, and possibly also based on the truck’s location, neither of which are currently possible
5. Ensure all vehicles are operating with optimized parameters, and obtain higher quality data about various parameters' effectiveness

The only concern expressed was that telematics providers might charge high prices for such programming.

When asked about tampering several fleets reported some incidents with maximum speed settings or vehicle sensors being altered, but by and large found that tampering occurs at low rates today.

<!-- page: 46 -->
Several fleets mentioned the terminology challenge when asked for any open ended comments. One sees this as a strong need at vocational training schools. Another commented that he now has a request for common terms on all Request for Proposals on all new truck orders. Fleets also requested a downloadable file and a comparison tool to know if any settings do not match the desired settings. They complained that the dealerships are not well trained when it comes to parameters, and in any case that dealerships lack a master file saved in their system with version controls.

### 7.2 Fleet Forum Internet Survey

In conjunction with Michelin, NACFE and the study team surveyed the 200-plus members of Michelin’s Fleet Forum to learn about how these fleets manage their parameter settings. The survey was available online to members of the Fleet Forum from November 6, 2014 until November 20, 2014. A total of 45 fleets responded to the survey, though not every question was answered on every survey.

> **Figure 19 — Frequency of Help**
> Type: pie chart
> Axes: none
> Series: Quarterly, Yearly, Every new vehicle order, Occasionally with a new vehicle order, Never
> Values: Quarterly: approx. 5%, Yearly: approx. 10%, Every new vehicle order: approx. 12%, Occasionally with a new vehicle order: approx. 18%, Never: approx. 55%
> Notes: Figure 19: Frequency of Help

One of the most striking findings regards when fleets receive support with optimizing their parameters. More than half of the fleets responded that they had never been given any help from their dealerships or their engine or vehicle OEMs in optimization. Roughly a quarter of the fleets received support at least occasionally when placing new orders. The most startling response was “bought a new truck in May and have not been informed of any of the values discussed in this survey unfortunately.”

<!-- page: 47 -->
> **Figure 20 — Help in Setting Parameters**
> Type: bar chart
> Axes: Help Source vs Number of Fleets
> Series: Engine OEM, Dealership, Another Fleet
> Values: Engine OEM: 10, Dealership: 10, Another Fleet: 2
> Notes: Annotations in chart: "4 Used OEM +Dealership", "One Fleet Used All 3 Methods". Figure 20: Help in Setting Parameters

A follow-up question found that the source of any optimization support is evenly split between truck dealerships and engine OEMs, with several fleets using both. A pair of fleets also reported using a peer from another fleet to help optimize their settings.

> **Figure 21 — Percent with Trucks Identically Set**
> Type: pie chart
> Axes: none
> Series: Identical, Differently But Want Identical, Set Differently By Design
> Values: Identical: approx. 50%, Differently But Want Identical: approx. 10%, Set Differently By Design: approx. 40%
> Notes: Figure 21: Percent with Trucks Identically Set

The majority of surveyed Michelin Fleet Forum fleets have their parameters set identically. This may be more common in fleets that utilizer fewer of the available parameters to manage their tractors. For instance, if only the vehicle’s maximum speed, maximum cruise speed, and idle shutdown timer are set at all, differences between the engine sweet spots, drivetrain gearing, and idle reduction systems of the various trucks in a fleet will not need to be accounted for.

<!-- page: 48 -->
Fleets were also asked how often they changed any of the parameters, and the results in the following chart show that the parameters are rarely if ever changed.

> **Figure 22 — Frequency of Changes**
> Type: pie chart
> Axes: none
> Series: Once Per Quarter, Twice A Year, Once A Year, Never
> Values: Once Per Quarter: approx. 10%, Twice A Year: approx. 5%, Once A Year: approx. 20%, Never: approx. 65%
> Notes: Figure 22: Frequency of Changes

When asked about how difficult it is to change parameters, the Michelin fleets reported finding it less than the NACFE fleets. This may be a result of fleet size, as the NACFE fleets will need to touch much larger numbers of individual vehicles to implement any changes. One Michelin fleet responded that “some engines are much easier to set parameters than others - some engines are much more time consuming and difficult.”

> **Figure 23 — Difficulty in Changing**
> Type: bar chart
> Axes: Difficulty Level vs Number of Fleets
> Series: none
> Values: Very Easy: 9, Somewhat Easy: 8, Neither Easy Nor Difficult: 7, Somewhat Difficult: 15
> Notes: Figure 23: Difficulty in Changing

<!-- page: 49 -->
Given that there is presently a debate within the industry around whether the parameter for maximum cruise control speed should be set to a higher, lower, or the same value as the parameter for maximum accelerator speed, the fleets Michelin were asked where they set those values. Most of the Michelin Forum fleets had the two parameters set to identical speeds, but a few set them differently.

> **Figure 24 — Maximum Speed Settings**
> Type: pie chart
> Axes: none
> Series: CC Max = Rd Spd Max, CC Max Higher, Rd Spd Max Higher
> Values: CC Max = Rd Spd Max: approx. 70%, CC Max Higher: approx. 20%, Rd Spd Max Higher: approx. 10%
> Notes: Figure 24: Maximum Speed Settings

Finally, when asked about whether they would benefit from the introduction of telematics to manage engine parameters, some of these fleets indicated that they had little need for such technologies as they either do not use telematics generally, or else never change any of the parameters. One fleet stated that “Security is a serious issue I have with this option.” But other fleets saw significant opportunities in telematics systems. One stated “Being able to modify parameters according to driver behavior real time, whether that is with reward or discipline, would help our fuel economy.”

## 8 Conclusions and Recommendations

By the close of this research one of Trucking Efficiency’s initial hypotheses about engine parameters had been resoundingly confirmed, namely, that many people in the industry understand the basic concept of using this technology to control certain aspects of vehicle operation, but that it is much harder to find people who are comfortable truly optimizing those parameters to obtain their full potential impact on fuel economy.

Along with this observation, three top level conclusions are evident in this body of research:

<!-- page: 50 -->
## 8.1 Conclusions

### 8.1.1 This Is More Complex Than It Needs To Be.

The optimization of fuel economy parameters is made difficult for fleets today by the sheer plethora of parameters available, the wide variation in the terminology and brand names used by the various OEMs, and the need to tailor parameters to the overall specifications of a truck, including its drivetrains, rear axle ratios, and additional installed technologies.

This report found that fuel-economy-related parameters can be broken down into six categories according to the aspect of fuel use the parameters address, though there is some overlap between them.

It is understandable that the OEMs did not and could not work together on their confidential software development. On the other hand when Accelerator Maximum Vehicle Speed (Cummins), Max Road Speed (Detroit), Customer Vehicle Limiting Speed (Mack), Max Accelerator Vehicle Speed (Navistar), and Maximum Accelerator Pedal Vehicle Speed (PACCAR) all mean the same thing there is now unnecessary complexity in the marketplace. The time has come to help fleets and dealership personnel alike by standardizing terminology whenever possible. Two of the OEMs made comments to this nature from our discussions.

### 8.1.2 Programmable Parameters Enhance Fuel Economy.

The study team believes that a fleet which is not currently utilizing programmable parameters at all, and rather is simply leaving their trucks set with OEM defaults, could obtain fuel efficiency improvements on the order of 5-8% by tailoring their parameters to their operations. Meanwhile, fleets may have improvements of 3-5% simply by setting parameters in a few key areas such as vehicle speed limiting and idle reduction. But it appears only large fleets have worked to optimize their settings.

Survey feedback indicates that large fleets appear to have achieved more optimization in their parameter settings. This may well be a result of more sensitivity to fuel costs as well as more support from the OEM representatives. Smaller fleets could well benefit from similar approaches whether supported by the OEMs or some well-trained dealership representatives.

### 8.1.3 Processes must be in place to manage these parameters.

Its key for fleets to have processes around the management of their programmable parameters to ensure they are operating as expected, and OEMs and dealers can each play a role in this. Fleets create templates, but they might have up to like 80 of them for different engines, plus different model years, plus different features a truck might have like APUs, transmissions etc.

<!-- page: 51 -->
With these conclusions in mind, the following are some recommendations for each of three groups involved; fleets, suppliers (OEMs), and dealerships.

## 8.2 Recommendations for Fleets

If you are not currently optimizing your programmable engine parameters for fuel economy, do so. The interviews, surveys, and Trucking Efficiency Workshops conducted for this Confidence Report provided numerous insights into how fleets handle their programmable parameters. Certain best practices clearly emerged:

### 8.2.1 Record keeping of parameter settings

One fleet interviewed has a three ring binder for every vehicle and every binder includes the list of parameter settings for that vehicle. Another fleet keeps their parameter settings on a thumb drive stored in their vault. If either of these fleets experienced an engine ECM that was destroyed, they could replace it with the same parameters it previously was running. If there was a lawsuit and the fleet was asked to show their settings, they would have records to do so.

Determine how parameter settings are to be documented and make sure the system is working in day to day operations. Does it capture changes recommended when new orders are made? Does it force thought about whether existing vehicles should be reprogrammed to have the same settings? Does the dealer always discuss these settings before ordering more vehicles to insure any improvements the fleet has found are rolled into the specifications for future vehicles? If the engine ECM suffers a complete failure can a replacement module be set to the exact same settings as the failed unit? The answer might not be templates as discussed in this report, but some method of control is highly recommended.

### 8.2.2 Create “parameter Templates” that cover a group of similarly specified trucks

One major fleet recently worked with their engine supplier and created over two dozen templates that cover all of their vehicles. Groups may differ by age, powertrain gearing and idle reduction system at a minimum. Make sure the parameter templates clearly call out the different aspects of the vehicle specifications that make the templates different from each other. Keep these files in a safe place electronically where they can be accessed as required.

### 8.2.3 Pilot review (or spec review) time with OEM to discuss parameter settings

At a workshop, one of the major fleets stated they have purposefully changed the agenda whenever meeting with OEM representatives at pilot or spec reviews. Now much more time is spent reviewing all parameter settings and less time is spent on the not nearly as critical items such as interior trim details. The parameters can make a large difference in fuel economy and operations, so focus on them whenever experts on parameters are available in the room.

<!-- page: 52 -->
If a fleet finds parameters to be confusing or challenging in any way they should be spending more time in conversation with their dealer and OEM. When OEM experts are present, such as order reviews or pilot reviews, it is time to have these discussions. When an OEM representative at a NACFE workshop was asked about the typical length of time fleets wanted to talk about parameters during a pilot review, the response was “anywhere from no discussion at all to a 45 minute discussion.”

For fleets that desire more training on parameters, it is highly recommended to contact both the appropriate dealership as well as the engine OEM for support. As an interim step, the Cummins PowerSpec system is available to anyone to download from the web. It provided detailed information on parameters.

### 8.2.4 Parameter verification checks to ensure vehicles are set as desired

Many of the fleets stated during the survey that they spot check the parameter settings upon delivery as well as periodically throughout the vehicle’s lifetime, especially at PM intervals.

### 8.2.5 Read or write tools to use with parameters

Some fleets are very well equipped with proper service tools and trained technicians to validate programmable parameter settings as desired, as well as reprogramming them if tampering has occurred or the fleet operational strategy has forced vehicles to be reprogrammed.

### 8.2.6 Protect passwords

Both fleet managers and dealer sales people reported that vehicles returned for trade-in frequently still had the fleet password active in the truck. While it is possible for at least some brands of engines to be erased via special passwords; that is not always what is happening. In many cases someone (dealership or used truck center) will call a friend at the fleet to ask for their parameter password to reprogram the engine for the next customer. Trust is great, but this is also an opportunity for your fleet’s password to get into the wrong hands. Fleets with the best procedures erase the password before the truck leaves their lot. Trade-ins should not have your password on them. Passwords should not be given out widely over the phone.

## 8.3 Recommendations for Suppliers

Fleet frustrations with parameter names and terminology was expressed loudly, clearly, and frequently in the study teams’ interviews. Degrees of differences due to intellectual property concerns are understandable and may be unavoidable, but greater commonality in nomenclature and naming conventions than what exists today would be very beneficial to the customer base. Engine manufacturers could provide an improved level of understanding and communications in the field if the names for common parameters between the different engine manufacturers were the same. Granted, given the software infrastructure already in place at each OEM (on-board vehicle/engine software, sales software, order management data, service tool software and training materials), streamlining terms

<!-- page: 53 -->
would not be a simple task, nor will it be simple to convince executives to undertake such an overhaul, given that the returns on this investment will be seen by the fleets who are thereby more easily able to optimize their parameters. Nonetheless, the study team recommends that creation of an industry-wide group to create recommended practices, before too many more parameters are created and implemented without guidelines. One fleet stated that making the change to common names would be very similar to the industry’s conversion to HD-OBD (common Heavy Duty On-Board Diagnostics) since “you will have to relearn a few things but you will only have to do it once.”

The development of telematics service tools by the OEMs was desired by many of the fleets surveyed. At a minimum, such telematics should be able to read existing parameters to validate settings. The ideal telematics-based tool would give fleets the ability to program parameters via wirelessly, removing the need to touch the trucks in a service bay, which requires time to coordinate the work and the time of the service technicians. Several hints were dropped that this capability may not be far off for some manufacturers.

Dealer sales personnel also appear to need additional training and technical support from OEMs on programmable parameters, especially if the number of parameters continues to increase. OEMs could perhaps accomplish this with additional training or more support engineers, or give them additional time to support dealers in providing fleets with the greatest available fuel economy opportunities from optimized programmable parameters.

Fleets can use additional help from OEMs in managing their parameters over a span of engine generations, variety of engine makes, and generations of vehicles. A software tool to manage templates would be a marketplace advantage for any OEM willing to develop such data. It would have far more value if it could handle parameters from other OEMs as well. Additionally, a feature that allows entire vehicle parameter sets to be compared would be extremely valuable to fleets.

Although not investigated as part of this report, it appears that all of the OEM systems have one generic set of defaults for programmable parameters. It may be helpful to both dealer sales personnel and to fleets if different sets of default values were made available to differentiate between different vehicle applications.

## 8.4 Recommendations for Dealers

Dealerships must provide their sales staff with strong training on programmable parameters as this is key to helping providing the best guidance to your customer base. Dealer guidelines on parameters should be available on-demand, online, and quite possibly also shared directly with customers. The effectiveness of this training will increase with strong support from other sales and service personnel at the dealership, as well as support-as-required from the engine OEM.

Dealers should ensure that all parameter optimization processes start with educated discussions of gearing. The right gearing is determined by the fleet’s desired operations, and encompasses engine sweet spots, transmission ratios, rear axle ratios, tire sizes, and programmable parameters. A failure to consider *any* one of these areas can and will result in vehicles that do not achieve optimal fuel economy. Customers that don’t fully comprehend the implications of the interrelations between vehicle and

<!-- page: 54 -->
engine configurations, programmable parameters, and fuel economy, need all of the visual aids, training materials, and software simulations that can be provided.

An OEM representative shared that the very best of dealer sales people will actually ride in a customer’s truck for a while to understand how that customer wants their vehicle to operate, as you first have to know everything that the customer already knows if you want to become even more knowledgeable than your customers.

## 8.5 Confidence Matrix

A Confidence Matrix (Figure 25) is a diagram used to inform fleets of Trucking Efficiency’s overall confidence in the technology being studied and the currently available performance data of that technology as compared to the payback a fleet should expect to receive from the technology.

This report finds that programmable engine parameters are proven to provide payback to those fleets that use them wisely – that is, who make a concentrated effort to optimize them on new trucks and then to manage them over the life of the truck Depending on where the fleet is in its current operations, paybacks can be rapid and significant for those that have not previously focused on this area. This report, unlike previous Confidence Report, does not provide a payback calculator, since programmable parameters have no direct upfront cost. The only “cost” of parameter optimization is the time and effort required to understand today’s complex parameter options and tools, and to maintain good records of fleet activities relating to parameters.

<!-- page: 55 -->
> **Figure 25 — Confidence Rating on Optimizing Engine Parameters**
> Type: scatter
> Axes: Payback in Years (1, 2, 3, 4) vs Confidence Rating (Low, Medium, High)
> Series: none
> Values: Invest in Testing (Low, 1), Quickly Invest in Testing (Medium, 1), Programmable Parameters / Great case for adoption! No testing required (High, 1), Consider Testing (Low, 2), Invest in Testing (Medium, 2), Invest in technology (High, 2), Wait for Next-Gen Products (Low, 3), Consider Testing (Medium, 3), Share Data with Industry (High, 3)
> Notes: Figure 25: Confidence Rating on Optimizing Engine Parameters. Legend shows color codes for Low, Medium, High confidence rating.

<!-- page: 56 -->
## Appendix A: References

Baxter, John, “A Primer on the Concept of Downspeeding Heavy Duty Trucks,” http://www.vehicleservicepros.com/article/11586930/a-primer-on-the-concept-of-downspeeding-heavy-duty-trucks, August 13, 2014.

Crissey, Jeff, “CCJ Innovator: J&R Schugel’s slow-go campaign boosts safety, fuel economy,” Commercial Carrier Journal, May 5, 2014.

Cummins Engine Company, “Feature Description, Cummins PowerSpec Version 5.4.1.17,” January 28, 2014.

Kilcarr, Sean, “Speed limiters: Perspective from Ontario,” Fleet Owner, April 24, 2014.

Leiper, James, “Ontario’s truck speed limiter law making the roads safer for us all,” TheRecord.com, April 15, 2013.

McKenna, David, “MAC IV Technical Sales Manual for US2010 For MACKTRAQ,” Mack Trucks, 2011.

Park, Jim, “Green in 2014: The New Greenhouse Gas Regulations,” Heavy Duty Trucking, Truckinginfo.com, January 2013.

Transport Canada, “Technical Considerations: Questionnaire responses from Truck and Engine Manufacturers Associations Regarding Heavy Truck Speed Limiters,” https://www.tc.gc.ca/eng/motorvehiclesafety/tp-tp14809-menu-400.htm, 2014.

<!-- page: 57 -->
## Appendix B: Manufacturer Summaries

| OEM | Ordering Tools | Service Tools | Contact |
| :--- | :--- | :--- | :--- |
| **CUMMINS** | PowerSpec | Insite (also PowerSpec if licensed & data cable equipped) | Jason Owens, Customer Performance Technical Manager – jason.owens@cummins.com |
| **FREIGHTLINER** | Spec Pro & Spec Manager | Diagnostic Link | Victor Meloche, Manager, Technical Sales - victor.meloche@daimler.com |
| **KENWORTH** | Prospector | DAVIE & ESA | Applications Support: 425-828-5999 |
| **MACK** | MACKTRAQ | VCADS PC | Joe Scarnecchia, National Accounts Powertrain Sales Manager - joseph.scarnecchia@macktrucks.com |
| **NAVISTAR** | Sales Tools & TCAPE | Service Maxx | Aaron Peterson, Chief Performance Engineer - aaron.peterson@navistar.com |
| **PETERBILT** | Prospector | DAVIE & ESA | Applications Support: PBDivision.Applications@paccar.com, 940-591-4096 |
| **VOLVO** | TM2 | PPT | Customer Support: 1-800-525-6586 |

<!-- page: 58 -->
Note: This is an initial NACFE comparison of parameter names, additional in depth analysis with the OEMs is required to know that these parameters match. 9-Feb-15

| | Cummins | Detroit | Mack | Navistar | PACCAR | Volvo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Vehicle Speed Limits | Accelerator Maximum Vehicle Speed | Max Road Speed | Customer Vehicle Limiting Speed | Max Accelerator Vehicle Speed | Maximum Accelerator Pedal Vehicle Speed | |
| Vehicle Speed Limits | Global Maximum Vehicle Speed | Absolute Max Veh Speed | | | Maximum Vehicle Speed – vehicle speed limiter | |
| Vehicle Speed Limits | Gear Down Protection - Feature Option | | Lower Gear Vehicle Limiting Speed Feature Activation | Gear Down Protection | SCM: Gear Down Protection | |
| Vehicle Speed Limits | | | Lower Gear Vehicle Limiting Speed | | | |
| Vehicle Speed Limits | Gear Down Protection Heavy Load Vehicle Speed | | | Top Gear Minus 1 - Heavy Load Speed | | |
| Vehicle Speed Limits | | | | Top Gear Minus 2 - Heavy Load Speed | | |
| Vehicle Speed Limits | Gear Down Protection Light Load Vehicle Speed | | | Top Gear Minus 1 - Light Load Speed | | |
| Vehicle Speed Limits | | | | Top Gear Minus 2 - Light Load Speed | | |
| Vehicle Speed Limits | | Torque Factor Gear Down Protect | | | | |
| Vehicle Speed Limits | | Torque Factor High Gear Power | | | | |
| Vehicle Speed Limits | PowerSpec Gear Down Protection Auto-Calculate - Feature Option | | | | | |
| Vehicle Speed Limits | Cruise Control Enable - Feature Option | | | Cruise Control Enable | | |
| Vehicle Speed Limits | Cruise Control Maximum Vehicle Speed | Max Cruise Set Speed | CC Max Set Speed | Max Cruise Control Vehicle Speed | Maximum CC set speed | |
| Vehicle Speed Limits | Min Cruise Set Speed low | Min Cruise Set Speed low | CC Min Set Speed | Min Cruise Control Vehicle Speed | Minimum speed that CC may be enabled | |
| Vehicle Speed Limits | | | | | Min Speed to Automatically Turn Cruise Control On | |
| Vehicle Speed Limits | Cruise Control Auto Resume | Enable Cruise Auto Resume | CC Autoresume with Clutch | | | |
| Vehicle Speed Limits | | | Cruise n' Brake Engagement Delay | | Delay in Engine Brake Activation with Brake Pedal Depressed | |
| Vehicle Speed Limits | Cruise Control Save Set Speed - Feature Option | | | | | |
| Vehicle Speed Limits | Cruise Control Lower Droop | | | | | |

<!-- page: 59 -->
| | Cummins | Detroit | Mack | Navistar | PACCAR | Volvo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Vehicle Speed Limits | Cruise Control Upper Droop | | | | | |
| Vehicle Speed Limits | | Pass Smart | | Vehicle Speed Limiter Override | | |
| Vehicle Speed Limits | Reserve Speed Increase Delta | PS Pass Speed Increment | | VSLO Speed Increment | | |
| Vehicle Speed Limits | | PS Pass Speed Duration | | VSLO Maximum Activation Time | | |
| Vehicle Speed Limits | Reserve Speed Maximum Distance | | | | | |
| Vehicle Speed Limits | | PS Pass Speed Interval | | Time Interval To Reset VSLO | | |
| Vehicle Speed Limits | | | | VLSO Time Duration Source | | |
| Vehicle Configuration | ECM Master Password | | | Customer Password | | |
| Vehicle Configuration | ECM Reset Password | | | | | |
| Vehicle Configuration | ECM Adjustment Password | | | | | |
| Vehicle Configuration | Vehicle Setup Rear Axle Ratio | Axle Ratio | | Rear Axle Ratio | | |
| Vehicle Configuration | Vehicle Setup Transmission Top Gear Ratio | Top Gear Ratio | | Top Gear Ratio | | |
| Vehicle Configuration | Vehicle Setup Transmission One Gear Down Ratio | Second Highest Gear Ratio | | Gear Ratio of Top Gear Minus 1 | | |
| Vehicle Configuration | | | | Gear Ratio of Top Gear Minus 2 | | |
| Vehicle Configuration | Vehicle Setup Tire Revolutions Per Distance | Tire Revs per Unit Distance | | Tire Revs per Mile | | |
| Vehicle Configuration | Vehicle Setup Transmission # of Tailshaft Teeth | Number of Output Shaft Teeth | | | | |
| Vehicle Configuration | Vehicle Speed Sensor(VSS) Type | Vehicle Speed Sensor | | | | |
| Vehicle Configuration | Vehicle Setup Application Type | | | | | |
| Engine Speed / Torque Limits | | Progressive Shift Enable | | Progressive shift enable | SCM: Progressive Shift | |
| Engine Speed / Torque Limits | | | | Low gear ratio break point | | |
| Engine Speed / Torque Limits | | | | High gear ratio break point | | |

<!-- page: 60 -->
| | Cummins | Detroit | Mack | Navistar | PACCAR | Volvo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Engine Speed / Torque Limits | | PS Low Gear 1 Max RPM Limit | | Low gear engine speed limit | | |
| Engine Speed / Torque Limits | | PS Low Gear 1 Max Vehicle Spd | | Low gear speed limit | | |
| Engine Speed / Torque Limits | | PS Low Gear 1 RPM Limit | | | | |
| Engine Speed / Torque Limits | | PS High Gear RPM Limit | | High gear engine speed limit | | |
| Engine Speed / Torque Limits | | PS High Gear On Vehicle Spd | | | | |
| Engine Speed / Torque Limits | | | | High gear engine speed range | | |
| Engine Speed / Torque Limits | | PS Low Gear 2 Max RPM Limit | | | | |
| Engine Speed / Torque Limits | | PS Low Gear 2 Max Vehicle Spd | | | | |
| Engine Speed / Torque Limits | | PS Low Gear 2 RPM Limit | | | | |
| Engine Speed / Torque Limits | Load Based Speed Control | | | RPM Minimum Progressive Shift Gear Ratio | | |
| Engine Speed / Torque Limits | High Engine Speed Breakpoint | | | | | |
| Engine Speed / Torque Limits | Low Engine Speed Breakpoint | | | | | |
| Engine Speed / Torque Limits | Vehicle Acceleration Management | | | | | |
| Engine Speed / Torque Limits | Acceleration Limit #1 | | | | | |
| Engine Speed / Torque Limits | Acceleration Limit #2 | | | | | |
| Engine Speed / Torque Limits | Speed threshold #1 | | | | | |
| Engine Speed / Torque Limits | Speed threshold#2 | | | | | |
| Engine Speed / Torque Limits | Smart Torque 2 enable | | | | | |
| Engine Speed / Torque Limits | Powertrain protection | | | | | |
| Engine Speed / Torque Limits | PTP Max torque at 0 VSS | | | | | |
| Engine Speed / Torque Limits | PTP Max allowable driveshaft / axle torque | | | | | |

<!-- page: 61 -->
| | Cummins | Detroit | Mack | Navistar | PACCAR | Volvo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Idle Reduction | Idle Engine Speed | | | Idle Engine Speed | Engine idle speed | |
| Idle Reduction | Idle Shutdown - Feature Option | Enable Idle Shutdown | Idle Shutdown Feature Activation | Idle Shutdown Mode | Engine Idle Shutdown Timer Enabled | |
| Idle Reduction | Idle Shutdown Timer | Idle Shutdown Time | Idle Shutdown Time | Idle Shutdown Time with Park Brake Set | Timer Setting Non-PTO Mode w/Park Brake Set | |
| Idle Reduction | | Enable Idle Shutdown | | Idle Shutdown Time with Park Brake Released | Timer Setting Non-PTO Mode wo/Park Brake Set | |
| Idle Reduction | | | Idle Shutdown Warning Time | EIST Time for Shutdown Warning | | |
| Idle Reduction | | | Idle Shutdown Warning Temp | | | |
| Idle Reduction | | | Idle Shutdown Warmup Timer | | | |
| Idle Reduction | Idle Shutdown Manual Override - Feature Option | | | Latched Driver Override Enable | | |
| Idle Reduction | | | | | Enable Accelerator Pedal Reset | |
| Idle Reduction | | | | | Enable Clutch Pedal Reset | |
| Idle Reduction | | | | | Enable Park Brake Reset | |
| Idle Reduction | | | | | Enable Service Brake Reset | |
| Idle Reduction | Idle Shutdown in PTO | Enable PTO Shutdown | | | EIST - PTO Mode Overrule | |
| Idle Reduction | Idle Shutdown Percentage PTO Load Override | | | | | |
| Idle Reduction | Idle Shutdown Ambient Air Temperature Override - Feature Option | Ambient Air Temp Sensor Enable | | | EIST - Ambient Temperature Overrule | |
| Idle Reduction | Idle Shutdown Intermediate Ambient Air Temperature | | | Intermediate Ambient Air Temperature | | |
| Idle Reduction | Idle Shutdown Hot Ambient Air Temperature | Hi Amb Air Override Temp | | Max Ambient Air Temp for Idle Shutdown | High Temperature Ambient Overrule | |
| Idle Reduction | Idle Shutdown Cold Ambient Air Temperature - Parameter | Lo Amb Air Override Temp | | | Low Ambient Temperature Overrule | |
| Idle Reduction | Idle Shutdown Hot Ambient Automatic Override - Feature Option | | | | | |
| Idle Reduction | Idle Shutdown Manual Override Inhibit Zone - Feature Option | | | | | |
| Driver Rewards | Driver Reward Enable | Fuel Economy Incentive Enable | Fuel Economy Incentive Mode Selection (enable) | Driver Reward Enable | | |
| Driver Rewards | Driver Reward Expected Fuel Economy Standard | FEI Minimum Fuel Economy | FEI Penalty Target Fuel Economy | Fuel Economy - Expected Level | | |

<!-- page: 62 -->
| | Cummins | Detroit | Mack | Navistar | PACCAR | Volvo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Driver Rewards | Driver Reward Penalty Speed | | FEI Penalty Vehicle Speed Decrease | VS Limit Decrement - Penalty Level | | |
| Driver Rewards | Driver Reward Good Fuel Economy Standard | | FEI Reward Target Fuel Economy | Fuel Economy - Good Level | | |
| Driver Rewards | Driver Reward Best Fuel Economy Standard | | | Fuel Economy - Excellent Level | | |
| Driver Rewards | Driver Reward Expected Idle Standard | | | Percent Idle Time - Expected Level | | |
| Driver Rewards | Driver Reward Good Idle Standard | | | Percent Idle Time - Good Level | | |
| Driver Rewards | Driver Reward Best Idle Standard | | | Percent Idle Time - Excellent Level | | |
| Driver Rewards | Driver Reward Speed Reward Mode | | | | | |
| Driver Rewards | Driver Reward Expected Speed Reward | | | VS Limit Increment - Expected Level | | |
| Driver Rewards | Driver Reward Good Speed Reward | FEI Max Vehicle Speed Reward | FEI Reward Vehicle Speed Increase | VS Limit Increment - Good Level | | |
| Driver Rewards | Driver Reward Best Speed Reward | | | VS Limit Increment - Excellent Level | | |
| Driver Rewards | FEI Use Trip Mileage | | FEI Distance Calculation Interval | | | |
| Driver Rewards | FEI Conversion Factor | | | | | |
| Driver Rewards | CDR mode | | | | | |
| Driver Rewards | CDR Reset Frequency | | | | | |
| Driver Rewards | Top Gear Max CDR incentive | | | | | |
| Driver Rewards | Max CDR incentive for CC | | | | | |
| Miscellaneous | Engine Brake Cruise Control Activation | Cruise Control Enable Eng Brk | | Retarder Mode | MX Retarder State Cruise Control On | |
| Miscellaneous | Cruise Control Speed Delta for Minimum Engine Brake | Low Eng Brk Max Cruise RSL Spd | | Minimum Vehicle Speed For Retarder | | |
| Miscellaneous | Cruise Control Speed Delta for Maximum Engine Brake | Hi Eng Brk Max Cruise RSL Spd | | Cruise Control Retarder High Speed | | |
| Miscellaneous | | | | | | |
| Miscellaneous | | | | | | |
| Miscellaneous | | Eng Brk Stage On Service Brake | | | | |

<!-- page: 63 -->
| | Cummins | Detroit | Mack | Navistar | PACCAR | Volvo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Miscellaneous | | Service Brk Enable Eng Brakes | | | | |
| Miscellaneous | | | | Delay After Throttle Pedal Released | | |
| Miscellaneous | | | | Delay After Brake Pedal Pressed | | |
| Miscellaneous | | Enable eCoast (Detroit Transmission) | | | | |
| Miscellaneous | Engine brake delay (up to 3 seconds) | | | | Allow Multi-Torque Only When Cruise is Active | |
| Miscellaneous | Engine brake min vehicle speed | | | Upshift Indicator | | |
| Miscellaneous | Adaptive Cruise Control - Feature Option | | | | | |
| Miscellaneous | | Predictive Cruise Control Eng Brake Mode | | | | |
| Miscellaneous | | Predictive Cruise Control Lower Veh Spd Limit | | | | |
| Miscellaneous | | Predictive Cruise Control RSL Mode | | | | |
