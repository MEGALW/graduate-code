# ======================================================
# 0. 环境变量设置 (必须放在最前面，解决 OMP DLL 冲突报错)
# ======================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import glob
from dataclasses import dataclass

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func
from torch.utils.data import Dataset, DataLoader

# 导入 AMD 显卡加速核心库
import torch_directml

# ======================================================
# 1. 全局配置项 (已深度适配 AMD DirectML)
# ======================================================
@dataclass
class Config:
    image_size: int = 256
    channels: int = 1
    timesteps: int = 1000
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    # 自动检测并调用 AMD 显卡，如果检测不到才会退回 CPU
    device: str = torch_directml.device() if torch_directml.is_available() else "cpu"
    save_dir: str = "I:/outputs"

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)

# ======================================================
# 2. 真实数据集加载器 (直接读取你做好的 LLVIP 图像)
# ======================================================
class InfraredDataset(Dataset):
    def __init__(self, low_dir, high_dir, image_size=256):
        # 搜索目录下所有图片并排序，确保 low 和 high 一一对应
        self.low_paths = sorted(glob.glob(os.path.join(low_dir, "*.*")))
        self.high_paths = sorted(glob.glob(os.path.join(high_dir, "*.*")))
        self.image_size = image_size
        
        assert len(self.low_paths) > 0, f"❌ 在 {low_dir} 找不到图片！请检查路径。"
        assert len(self.low_paths) == len(self.high_paths), "❌ low 和 high 文件夹里的图片数量不一致！"

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        # 以灰度图模式读取
        low_img = cv2.imread(self.low_paths[idx], cv2.IMREAD_GRAYSCALE)
        high_img = cv2.imread(self.high_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 统一缩放尺寸
        low_img = cv2.resize(low_img, (self.image_size, self.image_size))
        high_img = cv2.resize(high_img, (self.image_size, self.image_size))
        
        # 转换为 Tensor，归一化到 [0, 1]，增加通道维度 [1, H, W]
        low_tensor = torch.from_numpy(low_img).float().unsqueeze(0) / 255.0
        high_tensor = torch.from_numpy(high_img).float().unsqueeze(0) / 255.0
        
        return {"low": low_tensor, "high": high_tensor}

# ======================================================
# 3. 核心网络：双重注意力机制与 U-Net
# ======================================================
class TimeEmbedding(nn.Module):
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
    def __init__(self, channels):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
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
        h = self.attn(h + t_emb) 
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
        
        self.inc = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_ch, base_ch * 2, self.time_dim)
        self.down2 = ResidualBlock(base_ch * 2, base_ch * 4, self.time_dim)
        self.up1 = ResidualBlock(base_ch * 4 + base_ch * 2, base_ch * 2, self.time_dim)
        self.up2 = ResidualBlock(base_ch * 2 + base_ch, base_ch, self.time_dim)
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x, low_img, ada_map, t):
        t_emb = self.time_embed(t)
        # 核心创新：拼接噪声图、原图和自适应权重图
        x_in = torch.cat([x, low_img, ada_map], dim=1)
        
        e1 = self.inc(x_in)                        
        e2 = self.down1(F.avg_pool2d(e1, 2), t_emb) 
        e3 = self.down2(F.avg_pool2d(e2, 2), t_emb) 
        
        d1 = self.up1(torch.cat([F.interpolate(e3, scale_factor=2), e2], dim=1), t_emb)
        d2 = self.up2(torch.cat([F.interpolate(d1, scale_factor=2), e1], dim=1), t_emb)
        
        return self.outc(d2)

# ======================================================
# 4. 完整的扩散过程 (物理引擎)
# ======================================================
class DiffusionPipeline:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model, low_img, ada_map):
        model.eval()
        b, c, h, w = low_img.shape
        img = torch.randn((b, 1, h, w), device=self.device) 
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="去噪采样中", leave=False):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            pred_noise = model(img, low_img, ada_map, t)
            
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            
            model_mean = (1.0 / torch.sqrt(alpha_t)) * (
                img - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * pred_noise
            )
            
            if i > 0:
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(beta_t) * noise
            else:
                img = model_mean 
                
        return torch.clamp(img, 0.0, 1.0) 

# ======================================================
# 5. 辅助功能与评估
# ======================================================
def generate_adaptive_map(x):
    mean_filter = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)
    local_mean = mean_filter(x)
    ada_weight = torch.abs(x - local_mean) + (1.0 - local_mean) * 0.4
    return torch.clamp(ada_weight, 0.0, 1.0)

class Evaluator:
    @staticmethod
    def calc_metrics(pred, target):
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
# 6. 主程序入口 (调度总指挥)
# ======================================================
def main():
    print(f"🚀 初始化自适应扩散系统...")
    print(f"🖥️  当前正在调用的计算设备: {cfg.device}")
    
    model = AdaptiveInfraredUNet().to(cfg.device)
    pipeline = DiffusionPipeline(cfg.timesteps, cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    
    # ---------------------------------------------------------
    # ⚠️ 【重要核对】请确认你的退化数据集路径是否存放在这里：
    # ---------------------------------------------------------
    train_low_dir = "I:/data2/dataset/train/low"
    train_high_dir = "I:/data2/dataset/train/high"
    
    dataset = InfraredDataset(train_low_dir, train_high_dir, image_size=cfg.image_size)
    
    # Windows 环境下如果多进程读取数据卡死，强制设置 num_workers=0 (主进程读取)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    print(f"✅ 成功加载数据集！共 {len(dataset)} 对图片，分为 {len(dataloader)} 个 Batch。")
    
    best_psnr = 0.0
    
    # 开始炼丹循环
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{cfg.epochs}]")
        for batch in pbar:
            # 1. 获取真实数据并丢到 AMD 显卡上
            real_low = batch["low"].to(cfg.device)
            real_high = batch["high"].to(cfg.device)
            b_size = real_low.shape[0] 
            
            # 2. 生成自适应条件图
            ada_map = generate_adaptive_map(real_low)
            
            # 3. 随机选时间步并加噪
            t = torch.randint(0, cfg.timesteps, (b_size,), device=cfg.device).long()
            noisy_img, true_noise = pipeline.q_sample(real_high, t)
            
            # 4. 网络预测噪声
            pred_noise = model(noisy_img, real_low, ada_map, t)
            loss = F.mse_loss(pred_noise, true_noise)
            
            # 5. 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # 周期性评估与保存模型 (每 5 个 Epoch 执行一次)
        if epoch % 5 == 0 or epoch == cfg.epochs:
            # 推理阶段从纯噪声开始，进行完整的去噪生成
            enhanced_img = pipeline.p_sample_loop(model, real_low, ada_map)
            psnr, ssim = Evaluator.calc_metrics(enhanced_img, real_high)
            
            print(f"🌟 Epoch [{epoch:03d}] 平均 Loss: {epoch_loss/len(dataloader):.4f} | PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f}")
            
            # 如果指标创新高，保存最佳权重
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_weight.pth"))

    print("\n🎉 训练周期结束！正在打包部署文件...")
    onnx_path = os.path.join(cfg.save_dir, "adaptive_infrared_enhancer.onnx")
    Evaluator.export_deployment_model(model, onnx_path)

if __name__ == "__main__":
    main()