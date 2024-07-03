# Crime prediction and forecasting in US universities

Files included: 
1. All data files including original, cleaned, combined, and feature engineered across 3 years from 2020 to 2022
2. SQL code to join csv files based on arrest numbers, crime numbers, disciplinary action numbers, and vawa numbers for reported data (on campus, off campus, public property and residence halls data have been combined in the same manner)
3. Data cleaning: dropping empty and irrelevant columns and saving them accordingly
4. Feature engineering: new features added - 
   a. Sum of each incident over 3 years,
   b. Average of each incident over 3 years,
   c. Differences between incident numbers between 2 particular years,
   d. Interaction terms between filters and incidents
