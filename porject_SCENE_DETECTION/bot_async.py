import os
import shutil
import time
from aiogram import Bot, Dispatcher, types
from aiogram import executor
from aiogram.types import ReplyKeyboardRemove
from aiogram.contrib.middlewares.logging import LoggingMiddleware
import pandas as pd
from PIL import Image
import torch
from scene_detection_datasets import CamSDD
from utils import load_model
from lavis.models import load_model_and_preprocess
import aiohttp
from aiogram.dispatcher.filters import MediaGroupFilter
from typing import List
from aiogram_media_group import media_group_handler
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import asyncio


# Define the path to the Excel file
EXCEL_FILE_PATH = "X:/bot/feedback_data.xlsx"
x = 1
# Check if the Excel file already exists
if os.path.exists(EXCEL_FILE_PATH):
    feedback_data = pd.read_excel(EXCEL_FILE_PATH)
else:
    feedback_data = pd.DataFrame(columns=["Selected Model", "Image Path", "Feedback"])

bot_token = ""  # the token is private
bot = Bot(token=bot_token)
dp = Dispatcher(bot, storage=MemoryStorage())
logging_middleware = LoggingMiddleware()
dp.middleware.setup(logging_middleware)

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = CamSDD(root_path='X:/datasets/CamSDD/test').prompts
model_blip_caption, vis_processors_blip_caption, _ = load_model_and_preprocess("blip_caption", "base_coco",
                                                                               is_eval=True, device=device)

# Dictionary to store the selected model for each user
user_selected_model = {}
user_image_paths = {}
user_images = {}
# Create an event to signal when feedback is received
event = {}
done = {}


# Initialize the three models at the beginning
def initialize_models():
    models = {}
    num_categories = 30

    # Model initialization for 'clip'
    model_clip, vis_processors_clip, _ = load_model('clip', 'ViT-B-32', 'linear_probe', True, num_categories)
    model_clip.load_state_dict(torch.load('checkpoints/20230909132644_clip_ViT-B-32_linear_probe_CamSDD.pth', map_location=torch.device(device)))
    models['clip'] = (model_clip, vis_processors_clip)

    # Model initialization for 'blip'
    model_blip, vis_processors_blip, _ = load_model('blip_feature_extractor', 'base', 'linear_probe', True, num_categories)
    model_blip.load_state_dict(torch.load('checkpoints/20230909140217_blip_feature_extractor_base_linear_probe_CamSDD.pth', map_location=torch.device(device)))
    models['blip'] = (model_blip, vis_processors_blip)

    # Model initialization for 'blip2'
    model_blip2, vis_processors_blip2, _ = load_model('blip2_feature_extractor', 'pretrain', 'linear_probe', True, num_categories)
    model_blip2.load_state_dict(torch.load('checkpoints/20230909160127_blip2_feature_extractor_pretrain_linear_probe_CamSDD.pth', map_location=torch.device(device)))
    models['blip2'] = (model_blip2, vis_processors_blip2)

    return models

# Initialize models at the beginning
model_dict = initialize_models()


# Start command handler
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    keyboard.add("Clip", "Blip", "Blip2")
    await message.answer("Hello! I am your Telegram bot. Please select a model:", reply_markup=keyboard)


# Handle the /changemodel command
@dp.message_handler(commands=['changemodel'])
async def change_model(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    keyboard.add("Clip", "Blip", "Blip2")
    await message.answer("Please select a new model:", reply_markup=keyboard)


# Handle model selection
@dp.message_handler(lambda message: message.text in ["Clip", "Blip", "Blip2"])
async def select_model(message: types.Message):
    user_id = message.from_user.id
    selected_model = message.text.lower()
    user_selected_model[user_id] = selected_model
    await message.answer(f'You selected {selected_model}. Now, please send an image.')


# Define a function to classify a single image
def classify_single_image(model, vis_processors, img):
    img = vis_processors["eval"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model.forward(img)
    values, indices = probs.topk(3)
    return values, indices


@dp.message_handler(MediaGroupFilter(is_media_group=True), content_types=types.ContentType.PHOTO)
@media_group_handler
async def album_handler(messages: List[types.Message]):
    user_id = messages[0].from_user.id
    if user_id not in user_selected_model:
        await messages[0].reply("Please select a model first.")
        return

    selected_model = user_selected_model[user_id]

    for message in messages:
        file_id = message.photo[-1].file_id
        image_path = await download_image(file_id)
        img = Image.open(image_path).convert("RGB")
        user_images[user_id] = img
        # Classify the image
        result = classify_single_image(model_dict[selected_model][0], model_dict[selected_model][1], img)

        # Process and send classification results as a plain text message
        values, indices = result
        response_text = "Classification results:\n"
        for i in range(3):  # Top 3 results
            class_index = indices[0, i].item()
            prob = values[0, i].item() * 100
            class_name = classes[class_index]
            response_text += f"{class_name}: {prob:.3f}%\n"

        # Remove the keyboard
        markup = ReplyKeyboardRemove()

        await message.reply(response_text, reply_markup=markup)
        # Define keyboard for feedback options
        feedback_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        feedback_keyboard.add("Yes", "No")
        await message.reply("Was the classification result accurate?", reply_markup=feedback_keyboard)

        # Define the directory where you want to save the image
        save_directory = "X:/bot/botImages"
        os.makedirs(save_directory, exist_ok=True)

        # Generate a unique filename for the saved image (e.g., using a timestamp)
        timestamp = str(int(time.time()))
        saved_image_path = os.path.join(save_directory, f"image_{timestamp}.jpg")
        user_image_paths[user_id] = saved_image_path

        # Move the downloaded image to the specified location using shutil.move
        shutil.move(image_path, saved_image_path)
        # Create an event for this user's feedback
        event[user_id] = asyncio.Event()
        # Wait for feedback
        await event[user_id].wait()
        await message.answer("Let's continue the classification:")
    await message.answer("Please upload another image.")


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_image(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_selected_model:
        await message.reply("Please select a model first.")
        return
    selected_model = user_selected_model[user_id]
    file_id = message.photo[-1].file_id
    image_path = await download_image(file_id)
    img = Image.open(image_path).convert("RGB")
    user_images[user_id] = img
    # Classify the image
    result = classify_single_image(model_dict[selected_model][0], model_dict[selected_model][1], img)

    # Process and send classification results as a plain text message
    values, indices = result
    response_text = "Classification results:\n"
    for i in range(3):  # Top 3 results
        class_index = indices[0, i].item()
        prob = values[0, i].item() * 100
        class_name = classes[class_index]
        response_text += f"{class_name}: {prob:.3f}%\n"

    # Remove the keyboard
    markup = ReplyKeyboardRemove()

    await message.answer(response_text, reply_markup=markup)

    # Define keyboard for feedback options
    feedback_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    feedback_keyboard.add("Yes", "No")

    await message.answer("Was the classification result accurate?", reply_markup=feedback_keyboard)

    # Define the directory where you want to save the image
    save_directory = "X:/bot/botImages"
    os.makedirs(save_directory, exist_ok=True)

    # Generate a unique filename for the saved image (e.g., using a timestamp)
    timestamp = str(int(time.time()))
    saved_image_path = os.path.join(save_directory, f"image_{timestamp}.jpg")
    user_image_paths[user_id] = saved_image_path

    # Move the downloaded image to the specified location using shutil.move
    shutil.move(image_path, saved_image_path)
    event[user_id] = asyncio.Event()
    # Wait for feedback
    await event[user_id].wait()
    await message.answer("Let's continue the classification:")
    await message.answer("Please upload another image.")


@dp.message_handler(lambda message: message.text.lower() in ["yes", "no"])
async def handle_feedback(message: types.Message):
    feedback = message.text

    # Convert "Yes" to 1 and "No" to 0
    feedback_value = 1 if feedback.lower() == "yes" else 0

    # Get user ID and selected model
    user_id = message.from_user.id
    if user_id in user_selected_model:
        selected_model = user_selected_model[user_id]
    else:
        selected_model = "Not Selected"  # Handle the case where the user hasn't selected a model

    # Get image path (assuming it's defined in the same way as in the handle_image function)
    image_path = "Not Available"  # Default value if image_path is not found
    if user_id in user_image_paths:
        image_path = user_image_paths[user_id]

    # Store feedback, selected_model, and image path as numerical values in Excel and DataFrame
    global feedback_data
    feedback_data = feedback_data.append({"Selected Model": selected_model, "Image Path": image_path, "Feedback": feedback_value}, ignore_index=True)

    # Save data to Excel (synchronous operation)
    feedback_data.to_excel(EXCEL_FILE_PATH, index=False)

    next_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    next_keyboard.add("get a caption", "continue")
    await message.answer("What would you like to do next?", reply_markup=next_keyboard)


@dp.message_handler(lambda message: message.text.lower() == "get a caption")
async def get_caption(message: types.Message):
    user_id = message.from_user.id
    # Assuming you have already loaded the image into the 'img' variable and processed it
    img = vis_processors_blip_caption["eval"](user_images[user_id]).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate captions for the entire batch
        caption = model_blip_caption.generate({"image": img})

    # Assuming 'caption' contains the generated caption
    generated_caption = ' '.join(caption)  # Join the list elements into a single string

    await message.answer(f"Here's a caption for your image:\n{generated_caption.strip('[]')}")
    event[user_id].set()


@dp.message_handler(lambda message: message.text.lower() == "continue")
async def continue_(message: types.Message):
    user_id = message.from_user.id
    event[user_id].set()


# Define a function to download the image from Telegram servers
async def download_image(file_id):
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path

    # Download the image file from Telegram servers
    image_url = f'https://api.telegram.org/file/bot{bot_token}/{file_path}'
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status == 200:
                image_data = await resp.read()
                # Define the path to save the image
                image_path = os.path.join('C:/Users/97252/Desktop/Master/Deep Learning/Scene detection', f'{file_id}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                return image_path
            else:
                raise Exception("Failed to download image")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
