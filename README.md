# SEACAR Water Quality Pilot Project

# Task 1a: Exploratory Analysis

The exploratory analyses include:
1.	Data slicing and outlier removal
2.	Line chart of weekly/monthly/yearly count of data records for each WQ parameter in each managed area.
3.	Line chart with error bars showing monthly and yearly mean and standard deviation of each WQ parameter in each managed area
4.	Line charts of monthly mean at different stations for each WQ parameter in each managed area.
5.	Box plots of WQ parameters aggregated by month of the year for each managed area.
6.	Plots in (4) and (5) of four managed areas with sufficient data are organized by rows (WQ parameter) and columns (managed areas) in order to determine seasonality (https://usf.box.com/s/x05pmyx686hwsp56km0ayelwf1uxxnu2)
7.	Histogram showing sampling frequencies at each observation locations for each WQ parameters
8.	Pie chart showing the ratios of random and fixed sampling locations for each WQ parameters
9.	Bar chart of ratios of fixed sampling locations for each WQ parameters

Maps are created to show the spatial distribution of the WQ samples.
1.	Bubble map of sampling points of WQ parameter in each managed area. The Github repository only shows dissolved oxygen and total nitrogen. Maps of other parameters can be easily generated by changing program parameters.
2.	Bubble maps of sampling points of WQ parameter in each month of a year in each managed area. The Github repository only shows dissolved oxygen and total nitrogen in 2019. Maps of other parameters and in other years can be easily generated by changing program parameters.
3.	Maps showing locations of different continuous stations are created for each managed area.

Analyses in Task 1a are shared in:
-	[SEACAR_WQ_Exploratory_Analysis_Dis.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/SEACAR_WQ_Exploratory_Analysis_Dis.ipynb): Temporal analysis of discrete data
-	[Sample_Location_Analysis_Dis.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/Sample_Location_Analysis_Dis.ipynb): Spatial locations of discrete data
-	[SEACAR_WQ_Exploratory_Analysis_Con.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/SEACAR_WQ_Exploratory_Analysis_Con.ipynb): Temporal analysis of continuous data: all stations
-	[Sample_Location_Analysis_Con.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/Sample_Location_Analysis_Con.ipynb): Spatial locations of discrete data: all stations
-	[SEACAR_WQ_Exploratory_Analysis_Con_Stations.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/SEACAR_WQ_Exploratory_Analysis_Con_Stations.ipynb): Temporal analysis of continuous data: separate by station
-	[Sample_Location_Analysis_Con_Stations.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/Sample_Location_Analysis_Con_Stations.ipynb): Spatial locations of discrete data: separate by station

# Task 1b: Spatial Interpolation

### 1b.1 Regression Analysis with WQ Parameters
Ordinary least square regression (OLS) and Pearson correlation analyses have been conducted to examine the relations between the potential covariates and water quality parameters. The analyses have been conducted with data from 2016 to 2018 in all managed areas and in separate managed areas. The general procedure is:
1. 	Preprocessing, including outlier removal, daytime data selection, combine continuous and discrete data, and select data in specific managed areas and years
2. 	Aggregate data in identical locations to mean values in wet and dry seasons (tentative dry season: Nov. – Apr., wet season: May to Oct.)
3. 	Extract values from covariate rasters to water quality locations
4.	Conduct Pearson correlation and OLS regression analysis in wet and dry seasons

Regression and correlation analysis are documented in:

- [Covariates_Analysis_All.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Covariates_Analysis/Covariates_Analysis_All.ipynb): Analysis with 2016-2018 data in all five managed areas
- [Covariates_Analysis_MA.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Covariates_Analysis/Covariates_Analysis_MA.ipynb): Analysis with 2016-2018 data in all five managed areas
- [Correlation_Covariates.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Covariates_Analysis/Correlation_Covariates.ipynb): Correlation between covariates with 2016-2018 data in all five managed areas

### 1b.2 Evaluation of Interpolation Methods
The following interpolation methods are selected for evaluation:
- <u>Inverse Distance Weighting (IDW)</u>: weighted average of observed data points in the neighborhood (simplest, fast)
- <u>Ordinary Kriging (OK)</u>: estimate values by fitting a theoretical variogram model (established method, proven performance)
- <u>Empirical Bayesian Kriging (EBK)</u>: estimate values by fitting a non-parametric variogram model (flexible, local model fitting, better suited for complex data pattern)
- <u>EBK Regression Prediction (Regression Kriging or RK)</u>: Extends EBK with explanatory variable that known to affect the predicted values (better suited if there are influential covariates)

The interpolation programs call functions from ArcGIS python interface (arcpy). The performance of these are evaluated through cross-validation. The purpose is to select the best performed method for batch production. The following metrics were derived to evaluate model performance:

- <u>Mean Error (ME)</u>: measures the average absolute difference between the observed and predicted values (measures biases)
- <u>Root Mean Square Error (RMSE)</u>: square root of average squared difference between observed and predicted values (measures accuracy)
- <u>Mean Standardized Error (MSE)</u> standardized by standard deviation of observed values (accuracy relative to data variability)

Performance evaluation of interpolation methods are documented in:

- [Interpolation_ArcGIS_CH.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/Interpolation_ArcGIS_CH.ipynb): Interpolation evaluation in Charlotte Harbor
- [Interpolation_ArcGIS_Estero_Bay.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/Interpolation_ArcGIS_Estero_Bay.ipynb): Interpolation evaluation in Estero Bay
- [Interpolation_ArcGIS_Big_Bend.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/Interpolation_ArcGIS_Big_Bend.ipynb): Interpolation evaluation in Big Bend
- [RK_Covariate_Assessment_CH.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_CH.ipynb): Evaluation of regression kriging with different covariates in Charlotte Harbor
- [RK_Covariate_Assessment_EB.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_EB.ipynb): Evaluation of regression kriging with different covariates in Estero Bay
- [RK_Covariate_Assessment_BB.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_EB.ipynb): Evaluation of regression kriging with different covariates in Big Bend
- [RK_Covariate_Assessment_Biscayne.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_Biscayne.ipynb): Evaluation of regression kriging with different covariates in Biscayne Bay
- [RK_Covariate_Assessment_GTM.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_Biscayne.ipynb): Evaluation of regression kriging with different covariates in Guana Tolomato Matanzas National Estuarine Research Reserve

### 1b.3 Automated Interpolation
Regression Kriging is applied to interpolate water quality parameters (Dissolved Oxygen, Total Nitrogen, Salinity, Secchi Depth, and Turbidity) in all seasons defined in this [table](https://github.com/FloridaSEACAR/WQ_Summaries/blob/gh-pages/OEATUSF_Geospatial_TempSeasons.csv) for all managed areas. The interpolation algorithm utilizes the optimal combination of covariates identified in Task 1b.2.

- [Automate_Interpolation.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Spatial_Interpolation/Automate_Interpolation.ipynb): The main function that calls the interpolation function in autointerpolation.py. The program loads preprocessed data to save computing time.

- [autointerpolation.py](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/misc/autointerpolation.py): The interpolation function that can be applied to selected data points. The interpolation function is used in the main function to interpolate maps in all seasons.

# Task 1c: Gap Analysis
### 1c.1 Overall Visualization
Kernel density estimation (KDE) maps and aggregated standard error of prediction (SEP) maps are created in pairs for visual detection of sampling gaps and redundancies. The KDE and SEP maps are created from all data points from 2015 to 2019 for each parameter and in each managed area.

[Gap_Analysis_Part1.ipynb](https://nbviewer.org/github/qiang-yi/SEACAR_WQ_Pilot/blob/main/Gap_Analysis/Gap_Analysis_Part1.html): Pairs of KDE and SEP maps for all sampling points from 2015 to 2019.

### 1c.2 Seasonal Visualization
Kernel density estimation (KDE) maps and aggregated standard error of prediction (SEP) maps are created for spring, summer, fall and winter from 2015 to 2019 for each parameter and in each managed area.

[Gap_Analysis_Part2.md](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Gap_Analysis/Gap_Analysis_Part2.md): Pairs of seasonal KDE and SEP maps. The executable Python codes to generate these maps can be found [here](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Gap_Analysis/Gap_Analysis_Part2.ipynb)

### 1c.3 Identify Redundancy and Gaps
The KDE and SEP maps generated in Task 1c.1 are reclassified into low, neutral and high using 25 and 75 percentile thresholds. Then, the KDE and SEP maps are overlaid to identify redundant sampling points and gap areas according to the table below.

| Kernel density | Standard error of prediction | Implication | Output |
| ----------- | ----------- | ----------- | ----------- |
| High      | High | Natural variation, potentially seasonal issue, or might be unexplained variation| Display seasonal maps for SEACAR team to consider explanation |
| High   | Low | Potential redundancy       | Identify specific sampling points within these areas |
| Low   | Low | No change needed / low priority       |Reference only |
| Low   | High | Potential need for more stations       | Identify areas on the map |
