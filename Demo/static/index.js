const canvas = document.getElementById("gameCanvas");
const context = canvas.getContext("2d");
const rewardDisplay = document.getElementById("rewardDisplay");
const doneDisplay = document.getElementById("doneDisplay");
const modelSelect = document.getElementById("modelSelect");

const ip = "137.112.216.204"

let socket = null;
let isGameOver = true;
let isManualMode = false;

let modelNameMap ={
    "model1":"DQN 3000",
    "model2":"DQN 22200",
    "model3":"Double DQN 40800",
    "model4":"Maxmin Q Learning 40200",
    "model5":"Dueling DQN 32700",
}

const GameState = Object.freeze({
    SETUP: "setup",
    START: "start",
    STOP: "stop",
    GAME_OVER: "game_over",
});

let gameState = GameState.SETUP;

const updateGameState = (newState) => {
    if (!Object.values(GameState).includes(newState)) {
        console.error(`Invalid game state: ${newState}`);
        return;
    }
    gameState = newState;
    updateUIStatus(newState.replace(/_/g, " ").toUpperCase());
};

const updateUIStatus = (status) => (doneDisplay.textContent = status);
const updateReward = (reward) => (rewardDisplay.textContent = reward);

const renderInitialImage = () => {
    const img = new Image();
    img.onload = () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = "./background.png"; // Example image
};

const renderGame = (base64Image) => {
    const img = new Image();
    img.onload = () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = `data:image/jpeg;base64,${base64Image}`;
};

const setupWebSocket = () => {
    if (socket && socket.readyState !== WebSocket.CLOSED) return;
    socket = new WebSocket("ws://localhost:8000/ws/play");
    // socket = new WebSocket("ws://backend.dl.xiongy.pro/ws/play");
    // socket = new WebSocket("ws://"+ip+":8000/ws/play");

    socket.onopen = () => updateGameState(GameState.SETUP);
    socket.onmessage = handleSocketMessage;
    socket.onclose = () => console.log("WebSocket connection closed.");

    document.getElementById("startButton").textContent = "Start Game (Auto Mode)";
};

const handleSocketMessage = (event) => {
    const { next_observation, reward, done } = JSON.parse(event.data);
    updateReward(reward);
    renderGame(next_observation);
    if (done) endGame();
};

const startGame = () => {
    if ([GameState.SETUP, GameState.STOP, GameState.GAME_OVER].includes(gameState)) {
        isManualMode = false;
        isGameOver = false;
        const selectedModel = modelSelect.value;

        updateGameState(GameState.START);
        setupWebSocket();

        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ action: "start", model: selectedModel }));
        }
    }
};

const stopGame = () => {
    if ([GameState.START, GameState.GAME_OVER].includes(gameState)) {
        if (socket?.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ action: "stop" }));
            socket.close();
        }
        updateGameState(GameState.STOP);
        isGameOver = true;

        const currentReward = parseFloat(rewardDisplay.textContent); // Get the current reward
        const selectedModel = modelSelect.value;
        const mode = isManualMode ? "manual" : "auto";

        // Add the final reward to the table
        addFinalReward(selectedModel, currentReward, mode);
    }
};

const playManually = () => {
    if ([GameState.SETUP, GameState.STOP].includes(gameState)) {
        isManualMode = true;
        isGameOver = false;

        updateGameState(GameState.START);
        setupWebSocket();

        if (socket?.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ action: "manual" }));
        }
        document.addEventListener("keydown", handleKeyPress);
    }
};

const handleKeyPress = (event) => {
    if (event.code === "Space" && socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ action: "flap" }));
    }
};

const endGame = () => {
    if (gameState === GameState.START || isManualMode) {
        updateGameState(GameState.GAME_OVER);
        isGameOver = true;

        const currentReward = parseFloat(rewardDisplay.textContent); // Get the current reward
        const selectedModel = modelSelect.value;
        const mode = isManualMode ? "manual" : "auto";

        // Add the final reward to the table
        addFinalReward(selectedModel, currentReward, mode);

        if (socket) socket.close();
        console.log("Game has ended.");
        document.getElementById("startButton").textContent = "Restart Game";
    }
};

let topRewards = []; // Combined top 5 rewards

const updateTopRewardsTable = () => {
    const tableBody = document.getElementById("topRewardsTable").querySelector("tbody");
    tableBody.innerHTML = ""; // Clear the existing rows

    topRewards.forEach((rewardEntry, index) => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${rewardEntry.mode === "manual" ? "Manual" : modelNameMap[rewardEntry.model]}</td>
            <td>${rewardEntry.mode === "manual" ? "Manual" : "Auto"}</td>
            <td>${rewardEntry.reward}</td>
        `;
        tableBody.appendChild(row);
    });
};

const addFinalReward = (model, reward, mode) => {
    // Add the final reward to the array
    topRewards.push({ model, reward, mode });

    // Sort by reward descending
    topRewards.sort((a, b) => b.reward - a.reward);

    // Keep only the top 5
    if (topRewards.length > 5) {
        topRewards = topRewards.slice(0, 5);
    }

    // Save the updated topRewards to cookies
    saveTopRewardsToCookies();

    // Update the table
    updateTopRewardsTable();
};


const saveTopRewardsToCookies = () => {
    const jsonString = JSON.stringify(topRewards);
    document.cookie = `topRewards=${encodeURIComponent(jsonString)}; path=/; max-age=31536000`; // Save for 1 year
};

const loadTopRewardsFromCookies = () => {
    const cookies = document.cookie.split("; ");
    for (let cookie of cookies) {
        if (cookie.startsWith("topRewards=")) {
            const jsonString = decodeURIComponent(cookie.split("=")[1]);
            topRewards = JSON.parse(jsonString);
            updateTopRewardsTable(); // Update the UI table
            break;
        }
    }
};


// Initial render of the canvas image
window.onload = ()=>{
    renderInitialImage();
    loadTopRewardsFromCookies(); // Load saved data on page load
};

document.getElementById("startButton").addEventListener("click", startGame);
document.getElementById("stopButton").addEventListener("click", stopGame);
document.getElementById("manualButton").addEventListener("click", playManually);