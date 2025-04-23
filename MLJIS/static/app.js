document.getElementById('forecastForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const industryName = document.getElementById('industry').value;
    const year = parseInt(document.getElementById('year').value);
    const quarter = document.getElementById('quarter').value;

    if (!industryName || !year || !quarter) {
        alert("Please select all required fields.");
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                industry_name: industryName,
                target_year: year,
                target_quarter: quarter
            }),
        });

        const data = await response.json();

        if (response.ok) {
            document.getElementById('result').innerText =
                `Forecasted GDP for ${industryName} in ${quarter} ${year}: ${data.forecasted_gdp.toFixed(3)}`;
        } else {
            document.getElementById('result').innerText = `Error: ${data.error}`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred while fetching the forecast.';
    }
});
