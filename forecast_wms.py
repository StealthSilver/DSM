from modules.forecast_functions import predict_using_prophet, find_datetime_column
from datetime import datetime, timedelta

def wms_forecast(
        df_historical_data, 
        gen_starttime = '05:00:00', 
        gen_endtime = '19:00:00', 
        features_to_predict=['poa','ambTemp','modTemp'], # Default these 3 are least required for DT. If bifacial, then poa_dw also.
        freq_predict=15,
        days_ahead = 0):
    
    datetime_column_name = find_datetime_column(df_historical_data)
    
    datetime_to_predict = (df_historical_data[datetime_column_name].iloc[-1] + timedelta(minutes=1)).strftime("%d-%m-%Y %H:%M:%S")
    
    df_forecast = predict_using_prophet(df_data=df_historical_data, 
                                        features_to_predict=features_to_predict, 
                                        date_to_predict=datetime_to_predict, 
                                        gen_starttime=gen_starttime, 
                                        gen_endtime=gen_endtime, 
                                        freq=f"{freq_predict}T", 
                                        days_ahead=days_ahead)

    return df_forecast