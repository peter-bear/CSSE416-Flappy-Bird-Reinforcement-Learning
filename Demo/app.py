import time
import torch
import flappy_bird_gym
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from agent import Agent, DuelingAgent
from arguments import arguments
import base64
import cv2
from io import BytesIO
from PIL import Image
import asyncio
import threading

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load arguments globally
arg = arguments()

# Load multiple models and store them in a dictionary
model_paths = {
    "model1": "models/model_dqn_3000.pt",
    "model2": "models/model_dqn_22200.pt",
    "model3": "models/model_double_dqn_40800.pt",
    "model4": "models/model_maxmin_q_learning_40200.pt",
}
shared_agents = {
    model_name: Agent(None, arg) for model_name in model_paths
}

# add dueling agent
model_paths["model5"] = "models/model_dueling_dqn_32700.pt"
shared_agents["model5"] = DuelingAgent(None, arg)

for model_name, agent in shared_agents.items():
    agent.Net = torch.jit.load(model_paths[model_name])

# Lock for accessing the shared agents
agent_lock = threading.Lock()

# Convert image to binary for model input
def process_img_for_model(image):
    image = cv2.resize(image, (84, 84))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV)
    return binary_image

# Helper function to convert image to Base64 and send to frontend
async def send_image(websocket, image, reward, done):
    rotated_image = np.transpose(image, (1, 0, 2))
    pil_image = Image.fromarray(rotated_image)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=70)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    await websocket.send_json({
        "next_observation": img_str,
        "reward": reward,
        "done": done
    })


@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    await websocket.accept()

    # Initialize a new environment for each session
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")

    # Track the selected model and the associated agent
    selected_model = "model1"
    agent = shared_agents[selected_model]

    # Set up a queue to handle messages
    message_queue = asyncio.Queue()
    
    async def receive_messages():
        while True:
            try:
                data = await websocket.receive_json()
                await message_queue.put(data)
            except WebSocketDisconnect:
                print("WebSocket disconnected.")
                break

    # Start listening to messages in the background
    asyncio.create_task(receive_messages())
    
    # Reset environment and send initial observation
    rgb_obs = env.reset()
    await send_image(websocket, rgb_obs, reward=0, done=False)

    # Wait for "start" or "manual" signal
    manual_mode = False
    while True:
        data = await message_queue.get()
        action = data.get("action")
        if action == "start":
            selected_model = data.get("model", "model1")  # Get model from client, default to "model1"
            print(f"Selected model: {selected_model}")
            agent = shared_agents[selected_model]  # Select the appropriate agent
            print(f"Game started with model: {selected_model}")
            manual_mode = False
            break
        elif action == "manual":
            print("Switched to manual control mode.")
            manual_mode = True
            break
        elif action == "stop":
            await websocket.close()
            return

    # Prepare image for model input
    obs_for_model = np.repeat(np.expand_dims(process_img_for_model(rgb_obs), axis=0), 4, axis=0)
    total_reward = 0

    # Main game loop
    while True:
        try:
            await asyncio.sleep(0.05)  # Frame rate control

            if manual_mode:
                # Default action (no flap)
                action = 0

                # Process user input for manual control
                while not message_queue.empty():
                    data = await message_queue.get()
                    if data.get("action") == "flap":
                        # print("User triggered flap action.")
                        action = 1
                    elif data.get("action") == "stop":
                        # await websocket.close()
                        return

            else:
                # Lock access to the shared agent to prevent race conditions
                with agent_lock:
                    action = agent.greedy_action(obs_for_model)

                # Process messages for dynamic model switch
                while not message_queue.empty():
                    data = await message_queue.get()
                    if data.get("action") == "stop":
                        # await websocket.close()
                        return
                    elif data.get("action") == "play":
                        new_model = data.get("model", selected_model)
                        if new_model != selected_model:
                            selected_model = new_model
                            agent = shared_agents[selected_model]
                            print(f"Switched to model: {selected_model}")

            # Perform action and get next observation
            next_rgb_obs, reward, done, _ = env.step(action)
            total_reward += reward

            # Update the model input with the new observation
            next_obs_for_model = process_img_for_model(next_rgb_obs)
            obs_for_model = np.concatenate(
                (np.expand_dims(next_obs_for_model, 0), obs_for_model[:3, :, :]),
                axis=0
            )

            # Send frame to frontend
            await send_image(websocket, next_rgb_obs, reward=total_reward, done=done)

            if done:
                print("Game done. Waiting for frontend to restart.")
                break

        except WebSocketDisconnect:
            print("WebSocket disconnected.")
            break
