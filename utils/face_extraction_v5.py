import cv2
import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
import os
import requests
import traceback
import sys
# ===================================================================================
# Part 1: DECA 光照图生成模块 (来自您的第二个脚本)
# ===================================================================================
# 【重要】您需要确保 decalib 库已经正确安装，并且其依赖项也已就绪。
# 比如通过: pip install decalib
# 并且，DECA模型需要相关的数据文件，请务必根据下面的注释修改路径。
# try:
sys.path.append('models/Relightable-Portrait-Animation')
from src.decalib.utils import util
from src.decalib.utils.tensor_cropper import transform_points
from src.decalib.deca import DECA
from src.decalib.utils.config import cfg as deca_cfg
# except ImportError:
#     print("❌ 错误: 无法导入 decalib 库。")
#     print("请确保您已经正确安装了 DECA (decalib) 及其依赖项，并设置了正确的Python环境。")
#     exit()

class FaceImageRender:
    def __init__(self, deca_data_path: str) -> None:
        """
        初始化 DECA 模型。
        
        Args:
            deca_data_path (str): 存放 DECA 相关数据（如 mask 文件）的目录路径。
        """
        # --- 关键修改 ---
        # 移除了硬编码的路径，使其更加灵活。
        # 您需要提供一个包含 'FLAME_masks_face-id.pkl' 和 'FLAME_masks.pkl' 的路径。
        flame_masks_face_id_path = os.path.join(deca_data_path, 'FLAME_masks_face-id.pkl')
        flame_masks_path = os.path.join(deca_data_path, 'FLAME_masks.pkl')

        if not os.path.exists(flame_masks_face_id_path) or not os.path.exists(flame_masks_path):
            raise FileNotFoundError(
                f"DECA mask 文件未找到! 请检查路径: '{deca_data_path}'"
            )

        # 初始化 DECA 模型
        deca_cfg.model.use_tex = False 
        self.deca = DECA(config=deca_cfg)
        
        f_mask = np.load(flame_masks_face_id_path, allow_pickle=True, encoding='latin1')
        v_mask = np.load(flame_masks_path, allow_pickle=True, encoding='latin1')
        
        self.mask = {
            'v_mask': v_mask['face'].tolist(),
            'f_mask': f_mask['face'].tolist()
        }
        print("✅ DECA 模型初始化完成。")

    def image_to_3dcoeff(self, rgb_image):
        """从单张图像中提取DECA的3D系数"""
        with torch.no_grad():
            codedict, _ = self.deca.img_to_3dcoeff(rgb_image)
        return codedict

    def render_shape(self, shape, exp, pose, cam, light, tform, h, w):
        """使用给定的3D系数渲染光照图"""
        with torch.no_grad():
            verts, _, _ = self.deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_verts = util.batch_orth_proj(verts, cam)
            trans_verts[:,:,1:] = -trans_verts[:,:,1:]

            points_scale = [self.deca.image_size, self.deca.image_size]
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])

            shape_images, _, _, _, _ = self.deca.render.render_shape(
                verts, trans_verts, h=h, w=w, lights=light, images=None, return_grid=True, mask=self.mask
            )
            
            shape_images_np = shape_images.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy()[0] * 255
        return shape_images_np

class ShadingGenerator:
    """一个封装好的、用于从单个图像生成光照图的类"""
    def __init__(self, deca_data_path: str):
        self.fir = FaceImageRender(deca_data_path)

    def generate_from_image(self, face_image: np.ndarray):
        """
        直接从一个Numpy图像数组生成光照图。
        
        Args:
            face_image (np.ndarray): 输入的面部图像 (RGB, 512x512, np.uint8)。
            save_path (str): 生成的光照图保存路径。
        """
        # print("\n--- 开始生成光照图 ---")
        
        # 1. 从单个图像中提取所有需要的3D系数
        # print("正在提取3D系数...")
        codedict = self.fir.image_to_3dcoeff(face_image)
        neutral_expression = torch.zeros_like(codedict["exp"])
        # 2. 使用自身的系数进行渲染 (自己提供形状、姿态、表情和光照)
        # print("正在渲染光照图...")
        final_shading = self.fir.render_shape(
            shape=codedict["shape"],
            exp=codedict["exp"],
            pose=codedict["pose"],
            cam=codedict["cam"],
            light=codedict["light"],
            tform=codedict["tform"],
            h=codedict["height"],
            w=codedict["width"]
        )
        
        img = Image.fromarray(np.uint8(final_shading))
        return img

# ===================================================================================
# Part 2: 面部特征提取模块 (来自您的第一个脚本)
# ===================================================================================
def rtn_face_get(self, img, face, img_size_factor=3):
    """自定义面部对齐函数，裁剪后再扩大区域"""
    landmarks = face.kps
    min_x, min_y = np.min(landmarks, axis=0)
    max_x, max_y = np.max(landmarks, axis=0)
    
    h, w = img.shape[:2]
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    box_size = max(max_x - min_x, max_y - min_y) * img_size_factor
    
    x1 = max(0, int(center_x - box_size/2))
    y1 = max(0, int(center_y - box_size/2))
    x2 = min(w, int(center_x + box_size/2))
    y2 = min(h, int(center_y + box_size/2))
      
    expanded_face = img[y1:y2, x1:x2]
    face.crop_face = cv2.resize(expanded_face, (512, 512))
    
    # if len(face.crop_face.shape) == 3 and face.crop_face.shape[2] == 3:
    #     face.crop_face = cv2.cvtColor(face.crop_face, cv2.COLOR_BGR2RGB)
    
    return face

class FaceExtractor:
    def __init__(self, providers=['CUDAExecutionProvider','CPUExecutionProvider'], ctx_id=0):
        self.providers = providers
        self.ctx_id = ctx_id
        self.app = self._init_face_analysis("buffalo_l", det_size=(512, 512))
        # print("✅ InsightFace 模型初始化完成。")

    def _init_face_analysis(self, app_name, det_size=(512, 512)):
        app = FaceAnalysis(name=app_name, providers=self.providers)
        app.prepare(ctx_id=self.ctx_id, det_size=det_size)
        return app

    def extract_features(self, image_path, imgsize_scale_factor=3):
        """从输入图像中提取面部特征，返回面部图像和ID嵌入"""
        # print("\n--- 开始提取面部特征 ---")
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像 {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            faces = self.app.get(img_rgb)
            if not faces:
                raise ValueError("图像中未检测到面部")
            
            main_face = faces[0]
            
            # 1. 提取面部ID嵌入
            faceid_embeds = torch.from_numpy(main_face.normed_embedding).unsqueeze(0)
            
            # 2. 处理用于CLIP编码器的面部图像
            original_get = ArcFaceONNX.get
            try:
                ArcFaceONNX.get = lambda x, img, face: rtn_face_get(x, img, face, img_size_factor=imgsize_scale_factor)
                faces_crop = self.app.get(img_rgb)
                if faces_crop:
                    face_image = faces_crop[0].crop_face
                else:
                    face_image = face_align.norm_crop(img_rgb, landmark=main_face.kps, image_size=512)
            finally:
                ArcFaceONNX.get = original_get
            
            # print("✅ 面部特征提取成功!")
            return face_image, faceid_embeds
            
        except Exception as e:
            # print(f"❌ 面部提取错误: {e}")
            traceback.print_exc()
            return None, None

# ===================================================================================
# Part 3: 主函数入口 (合并后的执行流程)
# ===================================================================================
if __name__ == '__main__':
    # --- 1. 准备环境和参数 ---
    
    # 【【【*** 重要：请在此处配置您的路径 ***】】】
    # 这是 DECA 模型所需的数据文件（如 FLAME_masks.pkl）所在的文件夹路径。
    # 您需要将 'PATH_TO_YOUR_DECA_PROJECT/src/decalib/data' 替换为您的实际路径。
    DECA_DATA_FOLDER = "models/Relightable-Portrait-Animation/src/decalib/data/" 
    
    # 检查 DECA 路径是否已配置
    if DECA_DATA_FOLDER == "PATH_TO_YOUR_DECA_PROJECT/src/decalib/data":
        print("❌ 警告: 您尚未配置 DECA 数据文件夹的路径。")
        print(f"请修改脚本中的 `DECA_DATA_FOLDER` 变量，使其指向正确的位置。")
        exit()

    # 定义测试图片和输出文件的名称
    image_url = "https://images.unsplash.com/photo-1669455715139-086b58707bff?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    test_image_path = "test_face_input.jpg"
    extracted_face_path = "extracted_face.jpg"
    shading_map_path = "shading_map.jpg"

    # 下载测试图片
    if not os.path.exists(test_image_path):
        try:
            print(f"正在下载测试图片: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status()
            with open(test_image_path, 'wb') as f:
                f.write(response.content)
            print(f"测试图片已保存至: {test_image_path}")
        except Exception as e:
            print(f"下载测试图片失败: {e}")
            exit()
    
    # --- 2. 初始化模型 ---
    # 根据您的硬件选择 'CPUExecutionProvider' 或 'CUDAExecutionProvider'
    try:
        # 第一次运行时，模型文件会自动下载，请耐心等待
        face_extractor = FaceExtractor(providers=['CPUExecutionProvider'])
        shading_generator = ShadingGenerator(deca_data_path=DECA_DATA_FOLDER)
    except Exception as e:
        print(f"在初始化过程中发生严重错误: {e}")
        print("请确保 onnxruntime, insightface, decalib 等库已正确安装。")
        exit()

    # --- 3. 执行面部提取 ---
    face_image, faceid_embeds = face_extractor.extract_features(test_image_path,imgsize_scale_factor=3.0)
    face_image_shading, _ = face_extractor.extract_features(test_image_path,imgsize_scale_factor=6.0)
    # --- 4. 如果提取成功，则生成光照图 ---
    if face_image is not None and faceid_embeds is not None:
        # 保存提取出的面部图像，方便查看
        pil_img = Image.fromarray(face_image)
        pil_img.save(extracted_face_path)
        print(f"提取出的面部图像已保存至: {extracted_face_path}")
        
        # 生成光照图
        img = shading_generator.generate_from_image(
            face_image=face_image_shading
        )
        img.save(shading_map_path)
        print("\n🎉 全部流程执行完毕!")
        print(f" - 您可以查看输入图片: {test_image_path}")
        print(f" - 提取出的标准脸: {extracted_face_path}")
        print(f" - 生成的光照图: {shading_map_path}")

    else:
        print("\n❌ 流程中断: 未能成功提取面部特征，无法继续生成光照图。")