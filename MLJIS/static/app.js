document.getElementById("predict-btn").addEventListener("click", function () {
    const businessType = document.getElementById("business-type").value;
    const location = document.getElementById("location").value;

    if (!location) {
        alert("Please enter a location.");
        return;
    }

    document.getElementById("forecast-output").textContent = `Predicting trends for ${businessType} in ${location}...`;
});
