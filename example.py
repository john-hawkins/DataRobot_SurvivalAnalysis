# ###############################################################
# IMPORT THE FILE: SurvivalAnalysis.py
import SurvivalAnalysis as surv
from tabulate import tabulate
import datarobot as dr
import pandas as pd

df = pd.read_csv('data/Healthcare_example.csv')

# ###############################################################
# This is all of the configuration required to use the module
event_date = 'EVENT_DATE'
event_label_col = 'EVENT_TYPE'
event_censor_val = 'Censored'
scoring_date = 'TEST_DATE'
unique_index = 'ID'
proj_name = "CP_TEST_PROJECT"
target_name = "EVENT_OCCURED"
windows =  list(map(lambda x: x*28,[1,2,3,4,5,6,7,8,9,10,11,12])) 

full_df = surv.build_cumulative_probability_dataset(
    df,
    event_date,
    event_label_col,
    event_censor_val,
    scoring_date,
    unique_index,
    windows,
    target_name
)
 

hf_df = surv.build_hazard_function_dataset(    
    df,
    event_date,
    event_label_col,
    event_censor_val,
    scoring_date,
    unique_index,
    windows,
    28,
    target_name
)

# ###################################################################
# GENERATE a KAPLAN MEIER ESTIMATE OVER THE FULL TRAINING DATASET
kapm = surv.kaplan_meier_estimate(full_df, windows, target_name)


# ###################################################################
# LAUNCH THE DATAROBOT PROJECTS
# THIS CAN TAKE A WHILE AND WILL BLOCK EXECUTION
#

project = surv.run_cp_project( full_df, proj_name, target_name, unique_index, 'window', 20 )

project_hf = surv.run_hf_project( hf_df, proj_name, target_name, unique_index, 'window', 20 )

# ###################################################################
# Get the recommended model from each project
# 
#model = dr.ModelRecommendation.get(project.id).get_model()
#model_hf = dr.ModelRecommendation.get(project_hf.id).get_model()

# ###################################################################
# SCORE THE TEST SET
# We have two records with a single clinical test point
# One that should be much higher risk than the other
# - Note we are allowing DataRobot to perform model selection for us.
#
test = pd.read_csv('data/Healthcare_test_set.csv')
score_df = surv.build_scoring_dataset(test, scoring_date, unique_index, windows)

predictions = surv.score_data( score_df, project )
predictions_hf = surv.score_data( score_df, project_hf )

score_df['cumulative_probability'] = predictions['positive_probability']
score_df['survival_probability'] = predictions['class_0.0']
score_df['hazard_probability'] = predictions_hf['positive_probability']


# ###################################################################
# PLOT AND SAVE
# We plot the inferred survival curves according to the three approaches.
# A non-parametric Kaplan-Meier Estimate baseline
# A survival function generated using DataRobot AutoML to create a hazard function
# A survival function that is the inverse of the estimated cumulative probability function
#

import matplotlib.pyplot as plt

patient_x = score_df[ score_df['ID']==11991 ]
patient_y = score_df[ score_df['ID']==11999 ]
patient_x_surv = surv.survival_from_hazard(patient_x['hazard_probability'], windows )
patient_y_surv = surv.survival_from_hazard(patient_y['hazard_probability'], windows )

fig = plt.gcf()
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot( kapm['window'], kapm['survival'], label='Kaplan Meier') 
plt.plot( 
    patient_x['window'], 
    patient_x['survival_probability'], 
    label='DR - Cumulative') 
plt.plot( 
    patient_x_surv['window'], 
    patient_x_surv['survival'], 
    label='DR - Hazard Function') 
plt.title('Patient X')
plt.ylabel('Survival Probability')
plt.xlabel('Window (days)')
plt.legend(loc='lower left')
plt.ylim([0, 1.0])
plt.grid()


plt.subplot(1,2,2)
plt.plot( kapm['window'], kapm['survival'], label='Kaplan Meier')
plt.plot( 
    patient_y['window'], 
    patient_y['survival_probability'], 
    label='DR - Cumulative') 
plt.plot( 
    patient_y_surv['window'], 
    patient_y_surv['survival'], 
    label='DR - Hazard Function') 
plt.title('Patient Y')
plt.ylabel('')
plt.xlabel('Window (days)')
plt.legend(loc='lower left')
plt.ylim([0, 1.0])
plt.grid()

plt.savefig('example.png')

