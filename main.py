from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import shutil
import os
import cv2

from typing import List
import uuid
from fastapi import Form

app = FastAPI()

# 1. 配置模型路径 (请确保模型在 model 文件夹下)
#model = YOLO("model/yolo26x.pt")
# 预先定义模型路径字典
MODEL_PATHS = {
    "yolo26l": "model/yolo26l.pt",
    "yolo26n": "model/yolo26n.pt",
    "yolo26x": "model/yolo26x.pt"
}

# 用于存放已加载的模型，避免重复加载耗时
loaded_models = {
    "yolo26l": YOLO(MODEL_PATHS["yolo26l"]),
    "yolo26n": YOLO(MODEL_PATHS["yolo26n"]),
    "yolo26x": YOLO(MODEL_PATHS["yolo26x"])
}

# 2. 确保存放图片的目录存在
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 3. 挂载静态资源（CSS、图片等）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 4. 配置网页模板路径
templates = Jinja2Templates(directory="templates")


# 【路由 - 首页】
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 将 index.html 返回给浏览器
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )


# 【路由 - 预测】
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...),
                  model_name: str = Form(...)):  # 用 List 接收多文件
    # 检查模型是否已加载，没加载则现场加载
    if model_name not in loaded_models:
        path = MODEL_PATHS.get(model_name, MODEL_PATHS["yolo26l"])
        loaded_models[model_name] = YOLO(path)

    current_model = loaded_models[model_name]

    all_results = []

    for file in files:
        # 使用 UUID 生成唯一文件名，防止多图上传时冲突
        unique_id = uuid.uuid4().hex
        orig_filename = f"{unique_id}_orig.jpg"
        res_filename = f"{unique_id}_res.jpg"

        input_path = os.path.join(UPLOAD_DIR, orig_filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 识别
        results = current_model.predict(source=input_path)
        res_plotted = results[0].plot()

        # 保存结果图
        result_path = os.path.join(UPLOAD_DIR, res_filename)
        cv2.imwrite(result_path, res_plotted)

        # 将这一对路径存入列表
        all_results.append({
            "original_path": f"/static/uploads/{orig_filename}",
            "result_path": f"/static/uploads/{res_filename}"
        })

    return all_results  # 返回包含多个图片路径的数组

#terminal输入：uvicorn main:app --reload
