# Crime prediction in US universities

Files included: 
1. All data files including original, cleaned, combined, and feature engineered across 3 years from 2020 to 2022, and data visualization and evaluation charts
3. SQL code to join csv files based on arrest numbers, crime numbers, disciplinary action numbers, and vawa numbers for reported data (on campus, off campus, public property and residence halls data have been combined in the same manner)
4. Data cleaning: dropping empty and irrelevant columns and saving them accordingly
5. Feature engineering: new features added - 
   a. Sum of each incident over 3 years,
   b. Average of each incident over 3 years,
   c. Differences between incident numbers between 2 particular years,
   d. Interaction terms between filters and incidents
6. Data visualization: the following charts have been generated -
   a. Trend Analysis: Line Plot for Each Incident Type
      i.  Sum over years,
      ii. Average over years,
   b. Distribution Analysis: Histogram for Each Incident Type, 
   c. Difference Analysis: Histograms for Differences Between Years,
   d. Relationship Analysis: Scatter Plots Between Different Incident Types,
   e. Correlation Heatmap for Incident Sums and Filters,
   d. Interaction Terms Visualization,
   e. Sum of All Incidents for Each Year
7. Model Building & Evaluation: The following data has been generated -
   a. Predictions of each incident at university and state level
   b. Top 20 universities and top 10 states for each incident
   c. Learning curve based on accuracy for each prediction type
   d. Evaluation metrics, confusion matrix, and ROC curve for each prediction type
8. Flask app and corresponding html templates
