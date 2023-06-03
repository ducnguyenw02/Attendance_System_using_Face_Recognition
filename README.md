# Attendance_System_using_Face_Recognition
<p>
  <img src="figures/model_pipeline.png" alt="Pipeline" style="height: 100%; width: 100%;">
</p>


# Environment Setup
```python 
conda install faceid torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate faceid
pip install opencv-python
pip install onnxruntime==1.14.0
pip install onnxruntime-gpu==1.14.0
git clone https://github.com/ducnguyenw02/Attendance_System_using_Face_Recognition
cd Attendance_System_using_Face_Recognition
```
Ensuring the right data tree format

    Attendance_System_using_Face_Recognition
    ├── database_image
    │   ├── profile1.png
    |   ├── profile2.png
    |   ├── ...
    ├── database_tensor
    │   ├── profile1.npy
    |   ├── profile2.npy
    |   ├── ...

**database_image**: containing image for each profile

**database_tensor**: containing vector feature extracted by pretrained backbone for each profile

# Face Detection
Making a few key modifications to the YOLOv5 and optimize it for face detection.
 ```python
 yolov5m-face.onnx
 ```
 # Adding New Face
Pre-trained backbone ResNet100 weights on Glint360K which contains 17091657 images of 360232 individuals is available at : [weights](https://drive.google.com/drive/folders/1-DNgNFw-gQII1w0XK9hDBWaZsi4EEs-b?usp=share_link)

Manually add new face images to folder:
```python
database_image
```
For fast precomputation, pre-extract database images to .npy tensor:
```python
python feature_extraction.py --weight 'weights/backbone.pth' --path_database database_image
```
```python
database_tensor
```
Convert backbone weight to ONNX implementation:
```python
python converttoonnx.py
```
# Face Recognition
  It is implemented on ResNet100 backbone and SOTA ArcFace loss: [paper](https://arxiv.org/pdf/1801.07698.pdf)
  
  Following is the pipeline of ArcFace loss:
<p>
  <img src="figures/arcfaceloss.png" alt="arcface" style="height: 100%; width: 100%;">
</p>

# Webcam Real-time GPU inference
```python 
python detection_gpu.py
```