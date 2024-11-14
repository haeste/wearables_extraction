Two important files here.

GenerateSubjectReport 
GenerateSubjectReport will attempt to read all the sensor data in a subjects directory. It will resample all to once per minute and extract circadian rhythms using SSA. 
It will save the dataframe containing all data together to file and will save a number of figures and latex tables to be used in a report. Folder locations will need to be updated for new participants. 
A report can then be generated using the Latex template.

closed_loop_chrono_exercise
closed_loop_chrono_exercise will generate the physical activity schedule for participants in the scheduled PA study. Again URLs will need to be updated. It expects a single Lys file, a single CORE file (from the cloud), and an empatica folder. 
If this is to generate the scheduled based on week one only, set FIRST_WEEK_ONLY to True. 
If both weeks are to be analysed set to False but ensure that data files contain both weeks data. 
The start and end points of week one (observation week) and week two (schedule week) will need to be specified. These should start and end at exactly 00:00 on and should be exactly 7 days long.
Potential issues might arise if participant data is too short. 
