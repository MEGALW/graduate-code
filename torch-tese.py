import torch
import torch_directml

print("========== AMD 显卡 (DirectML) 测试 ==========")

# 1. 检查 DirectML 是否可用
is_available = torch_directml.is_available()
print(f"1. DirectML 可用状态: {is_available}")

if is_available:
    # 2. 获取 DirectML 设备名称 (通常会输出 privateuseone:0)
    dml_device = torch_directml.device()
    print(f"2. 当前分配的设备代号: {dml_device}")

    # 3. 终极测试：把数据传给 AMD 显卡并进行计算
    print("\n3. 正在将张量推送到显卡进行计算...")
    tensor_a = torch.tensor([10.0, 20.0, 30.0]).to(dml_device)
    tensor_b = torch.tensor([5.0, 5.0, 5.0]).to(dml_device)
    
    result = tensor_a + tensor_b

    print(f"   -> 计算结果: {result}")
    print(f"   -> 数据目前所在的硬件: {result.device}")
    print("\n✅ 如果你看到了这行字，并且上面没有报错，说明你的 AMD 显卡已经准备就绪！")
else:
    print("\n❌ 糟糕，PyTorch 还是找不到你的 AMD 显卡，请检查 torch-directml 是否安装在当前环境中。")