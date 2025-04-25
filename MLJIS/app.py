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
    
    if not industry_name:
        return jsonify({"error": "Industry name is required"}), 400

    try:
        # Unpack all values returned by forecast_and_evaluate
        gdp, mape, label, insight = forecast_and_evaluate(industry_name=industry_name)
        
        return jsonify({
            "forecasted_gdp": float(gdp),
            "cv_mape": round(mape, 2),
            "forecasted_quarter": label,
            "forecast_insight": insight
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
