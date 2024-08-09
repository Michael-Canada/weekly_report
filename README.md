# weekly_report
This repo generates the quantitative part of the weekly report, analysing the performance of constraint forecasting.

To run on MISO:
1. Run the file: Run_MISO.py   (running requires about 20 minutes)
2. Open the file Run_analysis_MISO.py
3. Change the dates in line 69 as in the example below:
   currently, line 69 is:
   if date > '2024-08-01' and date <= '2024-08-08':
   Change it to
   if date > 'put_today_minus7_days_here' and date <= 'put_today_date_here':
4. Use line 69 to replace line 90
5. Run the file Run_analysis_MISO.py (running will take 20 seconds)
7. Report to Abbie the bottom line that appears on the scree, that looks like this:

Confusion_matrix:
FP: 45.3, FN: 11.3, TP: 43.4


To run on SPP:
Follow the instructions from running on MISO above, but work on the different files:
Run_SPP.py
Run_analysis_SPP.py


To run on ERCOT: work on the following files:
Run_2.py (for ERCOT)
run_analysis (for ERCOT)
