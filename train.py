import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func

# ======================================================
# 1. 全局配置项 (配置中心化，易于调参)
# ======================================================
@dataclass
class Config:
    image_size: int = 256
    channels: int = 1
    timesteps: int = 1000
    batch_size: int = 4
    epochs: int = 50
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "I:/outputs"

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)

# ======================================================
# 2. 核心网络：双重注意力机制与 U-Net
# ======================================================
class TimeEmbedding(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class SpatialChannelAttention(nn.Module):
    """优化的自适应门控：通道与空间双重注意力"""
    def __init__(self, channels):
        super().__init__()
        # 通道注意力 (简化版)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力 (简化版)
        self.sa = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch)
        )
        self.attn = SpatialChannelAttention(out_ch)
        self.relu = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv(x)
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = self.attn(h + t_emb) # 结合时间信息后进行自适应注意力筛选
        return self.relu(h + self.shortcut(x))

class AdaptiveInfraredUNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.time_dim = base_ch * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU()
        )
        
        # 输入：噪声图(1) + 原始红外图(1) + 自适应权重图(1) = 3
        self.inc = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)
        
        self.down1 = ResidualBlock(base_ch, base_ch * 2, self.time_dim)
        self.down2 = ResidualBlock(base_ch * 2, base_ch * 4, self.time_dim)
        
        self.up1 = ResidualBlock(base_ch * 4 + base_ch * 2, base_ch * 2, self.time_dim)
        self.up2 = ResidualBlock(base_ch * 2 + base_ch, base_ch, self.time_dim)
        
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x, low_img, ada_map, t):
        t_emb = self.time_embed(t)
        
        # 将结构条件与当前状态拼接 [B, 3, H, W]
        x_in = torch.cat([x, low_img, ada_map], dim=1)
        
        e1 = self.inc(x_in)                        # [B, 64, H, W]
        e2 = self.down1(F.avg_pool2d(e1, 2), t_emb) # [B, 128, H/2, W/2]
        e3 = self.down2(F.avg_pool2d(e2, 2), t_emb) # [B, 256, H/4, W/4]
        
        d1 = self.up1(torch.cat([F.interpolate(e3, scale_factor=2), e2], dim=1), t_emb)
        d2 = self.up2(torch.cat([F.interpolate(d1, scale_factor=2), e1], dim=1), t_emb)
        
        return self.outc(d2)

# ======================================================
# 3. 完整的扩散过程 (含正向加噪与逆向推理)
# ======================================================
class DiffusionPipeline:
    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        
        # 预计算方差表 (Linear Schedule)
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """正向扩散：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model, low_img, ada_map):
        """逆向推理：从纯噪声生成增强后的清晰图像"""
        model.eval()
        b, c, h, w = low_img.shape
        img = torch.randn((b, 1, h, w), device=self.device) # 初始纯噪声
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="去噪采样中", leave=False):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            # 预测当前步的噪声
            pred_noise = model(img, low_img, ada_map, t)
            
            # 根据公式剔除噪声
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            
            # 计算无噪图像的均值
            model_mean = (1.0 / torch.sqrt(alpha_t)) * (
                img - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * pred_noise
            )
            
            if i > 0:
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(beta_t) * noise
            else:
                img = model_mean # 最后一步不加随机噪声
                
        return torch.clamp(img, 0.0, 1.0) # 确保像素值在合法范围内

# ======================================================
# 4. 辅助函数与评估指标
# ======================================================
def generate_adaptive_map(x):
    """计算红外图像特征权重分布"""
    mean_filter = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)
    local_mean = mean_filter(x)
    # 对比度 + 暗部补偿
    ada_weight = torch.abs(x - local_mean) + (1.0 - local_mean) * 0.4
    return torch.clamp(ada_weight, 0.0, 1.0)

class Evaluator:
    @staticmethod
    def calc_metrics(pred, target):
        """安全计算 PSNR 与 SSIM (处理 Batch)"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        psnr_list, ssim_list = [], []
        for i in range(pred_np.shape[0]):
            p, t = pred_np[i, 0], target_np[i, 0]
            mse = np.mean((p - t) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
            ssim = ssim_func(p, t, data_range=1.0)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
        return np.mean(psnr_list), np.mean(ssim_list)

    @staticmethod
    def export_deployment_model(model, filepath):
        model.eval()
        dummy_noise = torch.randn(1, 1, cfg.image_size, cfg.image_size).to(cfg.device)
        dummy_low = torch.randn(1, 1, cfg.image_size, cfg.image_size).to(cfg.device)
        dummy_ada = torch.randn(1, 1, cfg.image_size, cfg.image_size).to(cfg.device)
        dummy_t = torch.tensor([500]).to(cfg.device)
        
        torch.onnx.export(
            model, (dummy_noise, dummy_low, dummy_ada, dummy_t), filepath,
            export_params=True, opset_version=14,
            input_names=['x_t', 'low_quality', 'ada_map', 'timestep'],
            output_names=['predicted_noise'],
            dynamic_axes={'x_t': {0: 'batch_size'}, 'low_quality': {0: 'batch_size'}}
        )
        print(f"\n✅ 模型已成功编译导出至: {filepath} (支持动态 Batch)")

# ======================================================
# 5. 主程序入口
# ======================================================
import glob
import cv2
from torch.utils.data import Dataset, DataLoader

# --- 新增：真实数据集加载类 ---
class InfraredDataset(Dataset):
    def __init__(self, low_dir, high_dir, image_size=256):
        self.low_paths = sorted(glob.glob(os.path.join(low_dir, "*.*")))
        self.high_paths = sorted(glob.glob(os.path.join(high_dir, "*.*")))
        self.image_size = image_size
        assert len(self.low_paths) == len(self.high_paths), "low和high图片数量不一致！"

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img = cv2.imread(self.low_paths[idx], cv2.IMREAD_GRAYSCALE)
        high_img = cv2.imread(self.high_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        low_img = cv2.resize(low_img, (self.image_size, self.image_size))
        high_img = cv2.resize(high_img, (self.image_size, self.image_size))
        
        low_tensor = torch.from_numpy(low_img).float().unsqueeze(0) / 255.0
        high_tensor = torch.from_numpy(high_img).float().unsqueeze(0) / 255.0
        
        return {"low": low_tensor, "high": high_tensor}

# --- 修改：主程序入口 ---
def main():
    print(f"🚀 初始化自适应扩散系统 (设备: {cfg.device})...")
    model = AdaptiveInfraredUNet().to(cfg.device)
    pipeline = DiffusionPipeline(cfg.timesteps, cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    
    # ⚠️ 请确认这里的路径是你存放 LLVIP 图片的实际路径！
    train_low_dir = "I:/data2/dataset/train/low"
    train_high_dir = "I:/data2/dataset/train/high"
    
    dataset = InfraredDataset(train_low_dir, train_high_dir, image_size=cfg.image_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    print(f"✅ 成功加载数据集！共 {len(dataset)} 对图片。")
    
    best_psnr = 0.0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{cfg.epochs}]")
        
        for batch in pbar:
            # 取出真实的低质图和高质图
            real_low = batch["low"].to(cfg.device)
            real_high = batch["high"].to(cfg.device)
            b_size = real_low.shape[0]
            
            ada_map = generate_adaptive_map(real_low)
            t = torch.randint(0, cfg.timesteps, (b_size,), device=cfg.device).long()
            noisy_img, true_noise = pipeline.q_sample(real_high, t)
            
            pred_noise = model(noisy_img, real_low, ada_map, t)
            loss = F.mse_loss(pred_noise, true_noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # 周期性评估与保存
        if epoch % 5 == 0 or epoch == cfg.epochs:
            enhanced_img = pipeline.p_sample_loop(model, real_low, ada_map)
            psnr, ssim = Evaluator.calc_metrics(enhanced_img, real_high)
            print(f"🌟 Epoch [{epoch:03d}] PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f}")
            
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_weight.pth"))

    print("\n🎉 训练周期结束！正在打包部署文件...")
    onnx_path = os.path.join(cfg.save_dir, "adaptive_infrared_enhancer.onnx")
    Evaluator.export_deployment_model(model, onnx_path)

if __name__ == "__main__":
    main()