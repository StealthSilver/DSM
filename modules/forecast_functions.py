import pandas as pd
from prophet import Prophet

def generate_timestamps(date, freq='15T', days_ahead=0):
    """
    Generate a series of timestamps for a given date with a specified frequency,
    starting from the next multiple of the specified frequency after the given date,
    and optionally generate timestamps for additional days ahead.

    Args:
    - date (str): The date in 'DD-MM-YYYY' or 'DD-MM-YYYY %H:%M:%S' format.
    - freq (str): The frequency of timestamps, e.g., '5T' for 5 minutes, '15T' for 15 minutes, etc.
    - days_ahead (int): Number of additional days for which timestamps are generated. Default is 0.

    Returns:
    - pd.DataFrame: DataFrame with a column 'ds' containing timestamps in '%d-%m-%Y %H:%M:%S' format.
    """
    # Parse the date string
    try:
        start_time = pd.to_datetime(date, format='%d-%m-%Y')
    except ValueError:
        start_time = pd.to_datetime(date, format='%d-%m-%Y %H:%M:%S')

    # Adjust the start time to the next multiple of the specified frequency
    minutes = start_time.minute
    if minutes % int(freq[:-1]) != 0:
        minutes = minutes + (int(freq[:-1]) - minutes % int(freq[:-1]))
        start_time = start_time.replace(minute=minutes % 60, second=0)

    # Calculate the end time
    if ' ' not in date:
        end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(minutes=int(freq[:-1]))
    else:
        end_time = start_time.replace(hour=23, minute=59, second=59)

    # Generate the series of timestamps for the current date
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{freq[:-1]}min")

    if days_ahead > 0:
        # Calculate the end time for additional days ahead
        end_time_ahead = end_time + pd.Timedelta(days=days_ahead)

        # Generate the series of timestamps for additional days ahead
        timestamps_ahead = pd.date_range(start=end_time + pd.Timedelta(seconds=1),
                                          end=end_time_ahead, freq=f"{freq[:-1]}min")

        # Convert the DatetimeIndex to a DataFrame
        timestamps_ahead_df = pd.DataFrame({'ds': timestamps_ahead.strftime('%d-%m-%Y %H:%M:%S')})

        # Concatenate the current date timestamps with timestamps for additional days ahead
        timestamps_df = pd.concat([pd.DataFrame({'ds': timestamps.strftime('%d-%m-%Y %H:%M:%S')}), timestamps_ahead_df])
    else:
        timestamps_df = pd.DataFrame({'ds': timestamps.strftime('%d-%m-%Y %H:%M:%S')})

    return timestamps_df

def predict_using_prophet(df_data, 
                          features_to_predict=['poa','ambTemp','modTemp'], # Default these 3 are least required for DT. If bifacial, then poa_dw also.
                          date_to_predict='01-01-2030', 
                          gen_starttime='05:00:00', 
                          gen_endtime='19:00:00', 
                          freq='15T',
                          days_ahead = 0):
    
    datetime_column_name = find_datetime_column(df_data)

    df_predict = generate_timestamps(date_to_predict, freq=freq, days_ahead=days_ahead)

    for feature in features_to_predict:
        df_train = df_data.loc[:,[datetime_column_name, feature]].rename(columns={datetime_column_name:'ds', feature:'y'})
        
        m_ = Prophet()
        m_.fit(df_train)
        df_ = m_.predict(df_predict.loc[:,['ds']])

        if feature in {'poa', 'poa_dw'}:
            df_ = forcing_poa_tails_tozero(df_, generation_start=gen_starttime, generation_end=gen_endtime)

        df_predict[feature] = df_.loc[:,'yhat']

    df_predict = df_predict.rename(columns={'ds':datetime_column_name}) 

    return df_predict

def forcing_poa_tails_tozero(df_data, generation_start = '05:00:00', generation_end = '19:00:00'):
    # This function should be applied to Prophet predictions for POA and POA_DW
    mask = (df_data['ds'].dt.strftime('%H:%M:%S') < generation_start) | (df_data['ds'].dt.strftime('%H:%M:%S') >= generation_end)
    df_data.loc[mask, 'yhat'] = 0
    return df_data

# Function to find the column name containing datetime format
def find_datetime_column(df):
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
    return None
