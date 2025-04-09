from flask import Flask, request, jsonify, render_template
from market_forecast import forecast_industry, df_long

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    industry_name = data.get("industry_name")

    if not industry_name:
        return jsonify({"error": "Industry name is required"}), 400

    result = forecast_industry("All industry total")
    print("Forecasted GDP:", result)
    
    return jsonify({"forecasted_gdp": float(result)}) 
if __name__ == '__main__':
    app.run(debug=True)
