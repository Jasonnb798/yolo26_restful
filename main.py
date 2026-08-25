from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import shutil
import os
import cv2

app = FastAPI()

# 1. 配置模型路径 (请确保模型在 model 文件夹下)
model = YOLO("model/yolo26x.pt")

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
async def predict(file: UploadFile = File(...)):
    # 1. 保存原始图片
    input_path = os.path.join(UPLOAD_DIR, "original.jpg")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 调用模型识别
    # 注意：这里去掉了 save=True, project 和 name，改为手动处理结果
    results = model.predict(source=input_path)

    # 3. 【关键步骤】手动获取带框的图片
    # results[0].plot() 会返回一个包含检测框的 numpy 数组（BGR格式）
    res_plotted = results[0].plot()

    # 4. 【关键步骤】手动保存图片到你指定的路径
    result_filename = "result.jpg"
    result_path = os.path.join(UPLOAD_DIR, result_filename)
    cv2.imwrite(result_path, res_plotted)

    # 5. 返回路径（对应 FastAPI 挂载的 static 路径）
    return {
        "original_path": f"/static/uploads/original.jpg",
        "result_path": f"/static/uploads/{result_filename}"
    }