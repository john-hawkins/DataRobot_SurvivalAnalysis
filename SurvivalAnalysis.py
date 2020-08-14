from datetime import timedelta
import datarobot as dr
import pandas as pd
import numpy as np

# #######################################################################################
#
# GENERATE A DATASET WITH A BINARY CLASSIFICATION TARGET THAT ACTS AS 
# TARGET FOR A CUMULATIVE PROBABILITY PROJECT
#
#   in_df:           The raw pandas dataframe
#   e_date:          The event date column
#   e_label_col:     The event label column
#   e_censor_val:    The text label value that idicates the event date is a censor point
#   s_date:          Scoring date column
#   u_index:         The name of the unque index column
#   windows:         The vector of window values in number of days
#   cp_target:       The name for the final target column to be generated
#
# #######################################################################################

def build_cumulative_probability_dataset(
        in_df, 
        e_date, 
        e_label_col,
        e_censor_val, 
        s_date, 
        u_index, 
        windows,
        cp_target
    ):

    df = in_df.copy()

    # ENSURE THAT THE DATETIME COLUMNS ARE DATETIMES
    df[e_date] = pd.to_datetime(df[e_date], errors='ignore')
    df[s_date] = pd.to_datetime(df[s_date], errors='ignore')

    # ADD A UNIQUE INDEX TO THE DATASET IF NECESSARY
    if(u_index not in df.columns):
        df[u_index] = pd.Series( range(1,len(df)+1), index=df.index )

    # JOIN WINDOWS AGAINST THE DATASET IN AN OUTER JOIN THAT EXHAUSTIVELY
    # GENERATES ALL COMBINATIONS OF DATA AND WINDOW FOR EVENT
    qqdf = pd.DataFrame(data={'window':windows, 'joinr':1})
    df['joinr'] = 1
    newdf = pd.merge(df, qqdf, on='joinr', how='outer') 

    # Create a temporary date column for the outer window of the target group.
    def add_window(date_val, window):
        return date_val + timedelta(days=window)

    t_date = 'temp_outer_window_date'
    newdf['temp_outer_window_date'] = newdf.apply(
        lambda x: add_window(x[s_date], x['window']), axis=1
    )

    # NOW GO THROUGH AND GENERATE A BINARY CLASSIFICATION TARGET THAT INDICATES
    # WHETHER THE EVENT HAPPENS WITHIN THE WINDOW
    # --- PRESUME THAT THE EVENT IS NEVER BEFORET THE SCORING DATE
    newdf[cp_target] = np.where( newdf[e_date] <= newdf['temp_outer_window_date'], 1, 0 )

    # FINALLY IN ORDER TO RESPECT CENSORING WE NEED TO REMOVE RECORDS WHERE THE
    # CENSORING OCCURED BEFORE THE OUTER WINDOW
    def mark_for_removal(label, event_d, temp_d):
        if label==e_censor_val and event_d<temp_d:
            return 1
        else:
            return 0

    newdf['removal'] = newdf.apply(
        lambda x: mark_for_removal(x[e_label_col], x[e_date], x[t_date]), 
        axis=1 
    )
    finaldf = newdf.drop( newdf[ newdf['removal']==1 ].index )

    # DROP INTERMEDIATE COLUMNS BEFORE RETURNING
    finaldf.drop([e_label_col, e_date, t_date, 'joinr', 'removal'], axis=1, inplace=True)

    return finaldf

# #######################################################################################
# 
# PREPARE A DATASET FOR SCORING WITH THE CUMULATIVE PROBABILITY PROJECT
#
#   in_df:           The raw pandas dataframe
#   s_date:          Scoring date column
#   u_index:         The name of the unque index column
#   windows:         The vector of window values in number of days
#
# #######################################################################################

def build_scoring_dataset(
        in_df,
        s_date,
        u_index,
        windows
):
    df = in_df.copy()

    # ENSURE THAT THE DATETIME COLUMN IS DATETIMES
    df[s_date] = pd.to_datetime(df[s_date], errors='ignore')

    # ADD A UNIQUE INDEX TO THE DATASET IF NECESSARY
    if(u_index not in df.columns):
        df[u_index] = pd.Series( range(1,len(df)+1), index=df.index )

    # JOIN WINDOWS AGAINST THE DATASET IN AN OUTER JOIN THAT EXHAUSTIVELY
    # GENERATES ALL COMBINATIONS OF DATA AND WINDOW FOR EVENT
    qqdf = pd.DataFrame(data={'window':windows, 'joinr':1})
    df['joinr'] = 1
    finaldf = pd.merge(df, qqdf, on='joinr', how='outer')

    # DROP INTERMEDIATE COLUMNS BEFORE RETURNING
    finaldf.drop(['joinr'], axis=1, inplace=True)

    return finaldf


# #######################################################################################
#
# GENERATE A DATASET WITH A BINARY CLASSIFICATION TARGET THAT ACTS AS
# TARGET FOR A THE HAZARD FUNCTION
#
#   in_df:           The raw pandas dataframe
#   e_date:          The event date column
#   e_label_col:     The event label column
#   e_censor_val:    The text label value that idicates the event date is a censor point
#   s_date:          Scoring date column
#   u_index:         The name of the unque index column
#   windows:         The vector of window values in number of days
#   timestep:        The number of days between the windows
#   cp_target:       The name for the final target column to be generated
#
# #######################################################################################

def build_hazard_function_dataset(
        in_df,
        e_date, 
        e_label_col,
        e_censor_val,
        s_date, 
        u_index, 
        windows,
        timestep,
        cp_target
    ):

    df = in_df.copy()

    # ENSURE THAT THE DATETIME COLUMNS ARE DATETIMES
    df[e_date] = pd.to_datetime(df[e_date], errors='ignore')
    df[s_date] = pd.to_datetime(df[s_date], errors='ignore')

    # ADD A UNIQUE INDEX TO THE DATASET IF NECESSARY
    if(u_index not in df.columns):
        df[u_index] = pd.Series( range(1,len(df)+1), index=df.index )

    # JOIN WINDOWS AGAINST THE DATASET IN AN OUTER JOIN THAT EXHAUSTIVELY
    # GENERATES ALL COMBINATIONS OF DATA AND WINDOW FOR EVENT
    qqdf = pd.DataFrame(data={'window':windows, 'joinr':1})
    df['joinr'] = 1
    newdf = pd.merge(df, qqdf, on='joinr', how='outer')

    # Create a temporary date column for the outer window of the target group.
    def add_outer_window(date_val, window):
        return date_val + timedelta(days=window)

    # Create a temporary date column for the inner window of the target group.
    def add_inner_window(date_val, window):
        return date_val + timedelta(days=(window-timestep))

    t_date_in = 'temp_inner_window_date'
    newdf[t_date_in] = newdf.apply(
        lambda x: add_inner_window(x[s_date], x['window']), axis=1
    )
    t_date_out = 'temp_outer_window_date'
    newdf[t_date_out] = newdf.apply(
        lambda x: add_outer_window(x[s_date], x['window']), axis=1
    )
    # NOW GO THROUGH AND GENERATE A BINARY CLASSIFICATION TARGET THAT INDICATES
    # WHETHER THE EVENT HAPPENS WITHIN THE WINDOW
    # --- PRESUME THAT THE EVENT IS NEVER BEFORE THE SCORING DATE
    newdf[cp_target] = np.where( 
       (newdf[e_date] > newdf['temp_inner_window_date']) & (newdf[e_date] <= newdf['temp_outer_window_date']), 
        1, 0 )

    # FINALLY IN ORDER TO RESPECT CENSORING WE NEED TO REMOVE RECORDS WHERE THE
    # CENSORING OCCURED BEFORE THE OUTER WINDOW OR THE EVENT OCCURED BEFORE THE INNER WINDOW
    def mark_for_removal(label, event_d, inner_d, outer_d):
        if label==e_censor_val and event_d<outer_d:
            return 1
        elif event_d<inner_d:
            return 1
        else:
            return 0

    newdf['removal'] = newdf.apply(
        lambda x: mark_for_removal(x[e_label_col], x[e_date], x[t_date_in], x[t_date_out]),
        axis=1
    )
    finaldf = newdf.drop( newdf[ newdf['removal']==1 ].index )
    # DROP INTERMEDIATE COLUMNS BEFORE RETURNING
    finaldf.drop([e_label_col, e_date, 
        t_date_in, t_date_out, 'joinr', 'removal'], axis=1, inplace=True)

    return finaldf

# #######################################################################################
#
# CREATE A DATASET CONTAINING THE RECORDS WITH JUST THE WINDOW IN WHICH THE EVENT OCCURED.
# -- USED FOR PLOTTING

def get_event_window_lookup(
        in_df,
        e_date,
        e_label_col,
        e_censor_val,
        s_date,
        u_index, 
        windows,
        timestep,
        cp_target
    ):

    df = in_df.copy()

    # ENSURE THAT THE DATETIME COLUMNS ARE DATETIMES
    df[e_date] = pd.to_datetime(df[e_date], errors='ignore')
    df[s_date] = pd.to_datetime(df[s_date], errors='ignore')

    # ADD A UNIQUE INDEX TO THE DATASET IF NECESSARY
    if(u_index not in df.columns):
        df[u_index] = pd.Series( range(1,len(df)+1), index=df.index )

    # JOIN WINDOWS AGAINST THE DATASET IN AN OUTER JOIN THAT EXHAUSTIVELY
    # GENERATES ALL COMBINATIONS OF DATA AND WINDOW FOR EVENT
    qqdf = pd.DataFrame(data={'window':windows, 'joinr':1})
    df['joinr'] = 1
    newdf = pd.merge(df, qqdf, on='joinr', how='outer')

    # Create a temporary date column for the outer window of the target group.
    def add_outer_window(date_val, window):
        return date_val + timedelta(days=window)

    # Create a temporary date column for the inner window of the target group.
    def add_inner_window(date_val, window):
        return date_val + timedelta(days=(window-timestep))

    t_date_in = 'temp_inner_window_date'
    newdf[t_date_in] = newdf.apply(
        lambda x: add_inner_window(x[s_date], x['window']), axis=1
    )
    t_date_out = 'temp_outer_window_date'
    newdf[t_date_out] = newdf.apply(
        lambda x: add_outer_window(x[s_date], x['window']), axis=1
    )
    # NOW GO THROUGH AND GENERATE A BINARY CLASSIFICATION TARGET THAT INDICATES
    # WHETHER THE EVENT HAPPENS WITHIN THE WINDOW
    # --- PRESUME THAT THE EVENT IS NEVER BEFORE THE SCORING DATE
    newdf[cp_target] = np.where(
       (newdf[e_date] > newdf['temp_inner_window_date']) & (newdf[e_date] <= newdf['temp_outer_window_date']),
        1, 0 )

    finaldf = newdf[ newdf[cp_target]==1 ] 
    events = finaldf.loc[:,[u_index, 'window']]
    lookup = pd.Series(events.window.values,index=events[u_index]).to_dict()
    return lookup


# #######################################################################################
#
# CREATE A DATAROBOT PROJECT TO BUILD THE CUMULATIVE PROBABILITY ESTIMATION 
# PROJECT ON A DATASET GENERATED BY THE FUNCTION ABOVE
#
# RETURN A REFERENCE TO THE PROJECT
#
# #######################################################################################
def run_cp_project( df, proj_name, target, unique_index, window_col, workers ):
    proj = dr.Project.create( df, proj_name, max_wait = 9999 )

    group_partition = dr.GroupTVH(holdout_pct=0, validation_pct=20, partition_key_cols = [unique_index])

    mono_up = [window_col]
    flist_mono_up = proj.create_featurelist( name='mono_up', features=mono_up ) 
    advanced_options = dr.AdvancedOptions(
        monotonic_increasing_featurelist_id=flist_mono_up.id,
        only_include_monotonic_blueprints=True
    )

    proj.set_target( 
        target = target, 
        positive_class=1,
        partitioning_method = group_partition, 
        mode = dr.AUTOPILOT_MODE.FULL_AUTO, 
        max_wait = 9999,
        advanced_options=advanced_options)
    proj.set_worker_count( workers )
    proj.wait_for_autopilot()
    return proj

# #######################################################################################
#
# CREATE A DATAROBOT PROJECT TO BUILD THE HAZARD FUNCTION ESTIMATION
# PROJECT ON A DATASET GENERATED BY THE FUNCTION ABOVE
#
# RETURN A REFERENCE TO THE PROJECT
#
# #######################################################################################
def run_hf_project( df, proj_name, target, unique_index, window_col, workers ):
    proj = dr.Project.create( df, proj_name, max_wait = 9999 )
    #partition = dr.StratifiedTVH(holdout_pct=0, validation_pct=20) 
    partition = dr.GroupTVH(holdout_pct=0, validation_pct=20, partition_key_cols = [unique_index])
    proj.set_target( 
        target = target, 
        positive_class=1,
        partitioning_method = partition,
        mode = dr.AUTOPILOT_MODE.FULL_AUTO,
        max_wait = 9999,
    )
    proj.set_worker_count( workers )
    proj.wait_for_autopilot()
    return proj

# #######################################################################################
# SCORE A NEW RAW DATA SET
# #######################################################################################
def score_raw_data( df, project, unique_index, windows ):
    newdf = prepare_scoring_data( df, unique_index, windows )
    return score_data( newdf, project )

# #######################################################################################
# SCORE A PREPARED DATA SET
# #######################################################################################
def score_data( df, project ):
    dataset = project.upload_dataset( sourcedata=df, max_wait=9999 )
    # WE NEED A TRY CATCH AROUND THIS BECAUSE 
    # DATAROBOT DOES NOT GUARANTEE
    # TO PROVIDE A RECOMMENDED MODEL
    try:
        model = dr.models.ModelRecommendation.get( project.id ).get_model()
    except (RuntimeError, TypeError, NameError, dr.errors.ClientError):
        model = project.get_models()[0]
    pred_job = model.request_predictions( dataset.id )
    preds = dr.models.predict_job.wait_for_async_predictions( 
        project.id, predict_job_id=pred_job.id, max_wait=9999 
    )
    return preds

# #######################################################################################
# ADD WINDOWS TO DATASET FOR SCORING
#
# df:           The raw pandas dataframe
# unique_index: The name of the column that will uniquely identify rows in the raw dataset
# windows:      Array containing the set of integer valued number of days for the set of
#               cumulative probability windows
#
# #######################################################################################
def prepare_scoring_data( in_df, unique_index, windows ):
    df = in_df.copy()

    # ADD A UNIQUE INDEX TO THE DATASET IF NECESSARY
    if(unique_index not in df.columns):
        df[unique_index] = pd.Series( range(1,len(df)+1), index=df.index )

    # JOIN WINDOWS AGAINST THE DATASET IN AN OUTER JOIN THAT EXHAUSTIVELY
    # GENERATES ALL COMBINATIONS OF DATA AND WINDOW FOR EVENT
    qqdf = pd.DataFrame(data={'window':windows, 'joinr':1})
    df['joinr'] = 1
    newdf = pd.merge(df, qqdf, on='joinr', how='outer')
    newdf.drop(['joinr'], inplace=True, axis=1, errors='ignore')
    return newdf


# #######################################################################################
# GENERATE A KAPLAN MEIER ESTIMATE FROM THE DATAFRAME GENERATED ABOVE
#
# Kaplan-Meier estimator is defined as:
#
# 𝑆̂ (𝑡)=∏𝑖:𝑡𝑖<𝑡(1−𝑑𝑖/𝑛𝑖) 
# where  𝑑𝑖  is a number of events in a given timepoints and  𝑛𝑖  
# is a number of survived units from the previous timepoint.

def kaplan_meier_estimate(df, windows, target_name):
    grpd = df.groupby('window')[target_name].mean().reset_index()
    grpd['survival'] = 1.0
    acc = 1.0
    for w in windows:
        acc = acc * (1.0 - grpd.loc[ grpd[grpd['window']==w].index,:][target_name].values[0])
        grpd.loc[grpd[ grpd['window']==w ].index,'survival'] = acc
    grpd.drop([target_name], inplace=True, axis=1)
    grpd['cumulative_probability'] = 1 - grpd['survival'] 
    return grpd

# #######################################################################################
# TURN A SERIES OF HAZARD FUNCTION PREDICTIONS INTO A SURVIVAL FUNCTION
def survival_from_hazard(hazard_prob, windows):
    rez = pd.DataFrame({"hazard":hazard_prob, "window":windows})
    rez['survival'] = 1.0
    acc = 1.0
    for w in windows:
        acc = acc * (1.0 - rez.loc[ rez[rez['window']==w].index,:]['hazard'].values[0])
        rez.loc[rez[ rez['window']==w ].index,'survival'] = acc
    return rez

