let img = document.querySelector("#imageSelected");
let model;



$("#selectImage").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#imageSelected").attr("src", dataURL);
		$("#predictionList").empty();
	}


	let file = $("#selectImage").prop("files")[0];
	reader.readAsDataURL(file);
	img.classList.remove("imgNone");
})



$(document).ready(async function() {

	$(".progressBar").show();
	console.log("Loading model...");

	model = await tf.loadGraphModel("http://localhost/model.json");

	console.log(model);
	console.log("Model loaded.");
	$(".progressBar").hide();
});


$("#buttonPredict").click(submitPicture);


async function submitPicture() {
	let image = $("#imageSelected").get(0);

	let tensor = tf.browser.fromPixels(image, 3)
		.resizeNearestNeighbor([224, 224])
		.expandDims()
		.toFloat()
		.reverse(-1);

	console.log(tensor);

	let predictions = await model.predict(tensor).data();
	let top2 = Array.from(predictions)
		.map(function (p, i) {
			return {
				probability: p,
				className: IMAGE_CLASSES[i]

			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 2);

	$("#predictionList").empty();
	top2.forEach(function (p) {
		$("#predictionList").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
	});
}
