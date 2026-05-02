const fileInput = document.getElementById("fileInput")
const preview = document.getElementById("preview")

fileInput.addEventListener("change", function(){

const file = this.files[0]

if(file){

preview.src = URL.createObjectURL(file)

}

})

async function predict(){

if(fileInput.files.length === 0){

alert("Upload image first")
return

}

document.getElementById("spinner").style.display = "block"

const formData = new FormData()

formData.append("image", fileInput.files[0])

const response = await fetch("/predict", {

method: "POST",
body: formData

})

const data = await response.json()

document.getElementById("spinner").style.display = "none"

document.getElementById("prediction").innerText =
"Prediction: " + data.prediction

document.getElementById("confidenceText").innerText =
data.confidence + "%"

document.getElementById("confidenceBar").style.width =
data.confidence + "%"

document.getElementById("accuracy").innerText =
"Model Accuracy: " + data.accuracy + "%"

document.getElementById("heatmap").src =
"data:image/jpeg;base64," + data.heatmap

}