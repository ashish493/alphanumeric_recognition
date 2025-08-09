let cvsIn = document.getElementById("inputimg");
let ctxIn = cvsIn.getContext('2d');
let divOut = document.getElementById("predictdigit");
let svgGraph = null;
let mouselbtn = false;


// initilize
window.onload = function(){

    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.lineWidth = 7;
    ctxIn.lineCap = "round";
}

// add cavas events
cvsIn.addEventListener("mousedown", function(e) {

    if(e.button == 0){
        let rect = e.target.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        mouselbtn = true;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    }
    else if(e.button == 2){
        onClear();  // right click for clear input
    }
});

cvsIn.addEventListener("mouseup", function(e) {
    if(e.button == 0){
        mouselbtn = false;
        onRecognition();
    }
});
cvsIn.addEventListener("mousemove", function(e) {
    let rect = e.target.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    if(mouselbtn){
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
    }
});

cvsIn.addEventListener("touchstart", function(e) {
    // for touch device
    if (e.targetTouches.length == 1) {
        let rect = e.target.getBoundingClientRect();
        let touch = e.targetTouches[0];
        let x = touch.clientX - rect.left;
        let y = touch.clientY - rect.top;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    }
});

cvsIn.addEventListener("touchmove", function(e) {
    // for touch device
    if (e.targetTouches.length == 1) {
        let rect = e.target.getBoundingClientRect();
        let touch = e.targetTouches[0];
        let x = touch.clientX - rect.left;
        let y = touch.clientY - rect.top;
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
        e.preventDefault();
    }
});

cvsIn.addEventListener("touchend", function(e) {
    // for touch device
    onRecognition();
});

// prevent display the contextmenu
cvsIn.addEventListener('contextmenu', function(e) {
    e.preventDefault();
});

document.getElementById("clearbtn").onclick = onClear;
function onClear(){
    mouselbtn = false;
    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.fillStyle = "black";
}

// post data to server for recognition
function onRecognition() {
    console.time("predict");

    const endpoint = './predict';
    const data = {img : cvsIn.toDataURL("image/png").replace('data:image/png;base64,','')};
    
    data.debug = 'false';

    $.ajax({
            url: endpoint,
            type:'POST',
            data: data,

        }).done(function(data) {

            showResult(JSON.parse(data))

        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            console.log(XMLHttpRequest);
            alert("error");
        })

    console.timeEnd("time");
}


function showResult(resultJson){

    // show predict digit
    divOut.textContent = resultJson.prediction;

    // show probability with confidence level
    let probabilityText = "Probability: " + resultJson.probability.toFixed(2) + "%";
    
    if (resultJson.confidence) {
        probabilityText += " (" + resultJson.confidence + " confidence)";
    }
    
    document.getElementById("probStr").innerHTML = probabilityText;

    // Add color coding based on confidence
    let probElement = document.getElementById("probStr");
    if (resultJson.probability > 80) {
        probElement.style.color = "green";
    } else if (resultJson.probability > 60) {
        probElement.style.color = "orange";
    } else {
        probElement.style.color = "red";
    }
    
    // Show debug information if available
    if (resultJson.debug) {
        console.log("Debug information:", resultJson.debug);
        
        // Show top predictions in console
        if (resultJson.debug.top_5_predictions) {
            console.log("Top 5 predictions:");
            resultJson.debug.top_5_predictions.forEach((pred, idx) => {
                console.log(`${idx + 1}. '${pred.character}': ${pred.probability.toFixed(2)}%`);
            });
        }
        
        if (resultJson.debug.message) {
            console.log(resultJson.debug.message);
        }
        
        // Update UI to show debug info
        let debugInfo = document.getElementById("debugInfo");
        if (!debugInfo) {
            debugInfo = document.createElement("div");
            debugInfo.id = "debugInfo";
            debugInfo.style.cssText = "margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; font-size: 12px;";
            document.getElementById("probStr").parentNode.appendChild(debugInfo);
        }
        
        let debugHtml = "<strong>Debug Info:</strong><br>";
        debugHtml += `Tensor stats: min=${resultJson.debug.tensor_min?.toFixed(3) || 'N/A'}, `;
        debugHtml += `max=${resultJson.debug.tensor_max?.toFixed(3) || 'N/A'}, `;
        debugHtml += `mean=${resultJson.debug.tensor_mean?.toFixed(3) || 'N/A'}<br>`;
        
        if (resultJson.debug.top_5_predictions) {
            debugHtml += "Top 5: ";
            resultJson.debug.top_5_predictions.forEach((pred, idx) => {
                debugHtml += `${pred.character}(${pred.probability.toFixed(1)}%) `;
            });
        }
        
        debugInfo.innerHTML = debugHtml;
    } else {
        // Hide debug info if not in debug mode
        let debugInfo = document.getElementById("debugInfo");
        if (debugInfo) {
            debugInfo.style.display = "none";
        }
    }
}


function drawImgToCanvas(canvasId, b64Img){
    let canvas = document.getElementById(canvasId);
    let ctx = canvas.getContext('2d');
    let img = new Image();
    img.src = "data:image/png;base64," + b64Img;
    img.onload = function(){
        ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height);
    }
}

