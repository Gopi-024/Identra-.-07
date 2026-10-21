```javascript
const modal = document.getElementById("scanModal");
const scanStatus = document.getElementById("scanStatus");


function startPalmScan() {

    modal.classList.add("active");

    scanStatus.textContent = "Initializing palm scanner...";

    setTimeout(() => {
        scanStatus.textContent = "Place your palm here";
    }, 1500);
}


function closePalmScan() {

    modal.classList.remove("active");

}


function scrollToHow() {

    document.getElementById("how").scrollIntoView({
        behavior: "smooth"
    });

}


/* Close modal when clicking outside */

modal.addEventListener("click", function(event) {

    if (event.target === modal) {
        closePalmScan();
    }

});


/* Scanner status animation */

const messages = [
    "Place your palm here",
    "Scanning palm...",
    "Detecting palm features...",
    "Ready for authentication"
];

let messageIndex = 0;

setInterval(() => {

    messageIndex++;

    if (messageIndex >= messages.length) {
        messageIndex = 0;
    }

    scanStatus.textContent = messages[messageIndex];

}, 3000);
```
