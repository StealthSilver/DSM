import sys
from flask import Flask, jsonify, make_response, request
import traceback
import json
import pandas as pd
from forecast_wms import wms_forecast
import config

application = Flask(__name__)

@application.errorhandler(Exception)
def server_error(err):
    application.logger.exception(err)
    return (
        jsonify(
            {
                "error":True, 
                "errorMessage": err.args,
                "stack": traceback.format_exc(),
            }
        ),
        500,
    )

@application.route("/")
def root():
    return jsonify({"message":"Welcome to WMS Forecasting"})

@application.route("/wms_forecasting/forecast", methods=["POST"])
def forecast():
    try:
        input_data = json.loads(request.data)

        training_data = input_data["training_data"]
        prediction_frequency = input_data["frequency"]
        forecast_period = input_data["days_ahead"]

        try:
            gen_starttime = input_data["gen_starttime"]
            gen_endtime = input_data["gen_endtime"]
        except:
            gen_starttime = None
            gen_endtime = None
            print("Generation start and end times not provided. Using default values to forecast.")

        df_training = pd.DataFrame(training_data)
        for col in df_training.columns:
            if 'time' in col:
                datetime_column_name = col

        df_training[datetime_column_name] = pd.to_datetime(df_training[datetime_column_name], format="%d-%m-%Y %H:%M:%S")
        prediction_frequency = int(prediction_frequency)
        forecast_period = int(forecast_period)

        columns_forecasted = list(df_training.columns)
        columns_forecasted.remove(datetime_column_name)

        # print(df_training.dtypes)

        if (gen_starttime is None) and (gen_endtime is None):
            pass
            output = wms_forecast(df_training, freq_predict=prediction_frequency, days_ahead=forecast_period, features_to_predict=columns_forecasted)
        else:
            pass
            output = wms_forecast(df_training, freq_predict=prediction_frequency, days_ahead=forecast_period, gen_starttime=gen_starttime, gen_endtime=gen_endtime, features_to_predict=columns_forecasted)
        # print("\n output dtypes")
        # print(output.dtypes)
        # output['planttimestring'] = output['planttimestring'].dt.strftime("%d-%m-%Y %H:%M:%S")
        return jsonify(output.to_dict(orient="records"))

    except:
        print("error")
        raise Exception(sys.exc_info())
    
@application.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify({"error": True, "errorMessage": "Not found!"}), 404)

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=config.PORT, debug=True)



