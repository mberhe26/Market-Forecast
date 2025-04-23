from flask import Flask, request, jsonify, render_template
from market_forecast import forecast_and_evaluate

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    industry_name = data.get("industry_name")
    target_year = data.get("target_year")
    

    if not industry_name or not target_year:
        return jsonify({"error": "Industry name and target year are required"}), 400

    try:
        result = forecast_and_evaluate(industry_name=industry_name)
        return jsonify({"forecasted_gdp": float(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
