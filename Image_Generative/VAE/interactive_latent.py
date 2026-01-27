import torch
import matplotlib.pyplot as plt
import numpy as np
from models.vae import ConvVAE
import os
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches

class InteractiveLatentVisualizer:
    def __init__(self, model_path='checkpoints/vae_final.pth', latent_dim=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.model = self.load_model(model_path)
        self.current_z = torch.zeros(1, latent_dim).to(self.device)
        
        # 设置交互式界面
        self.setup_figure()
        self.setup_controls()
        self.update_image()
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        model = ConvVAE(latent_dim=self.latent_dim).to(self.device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型已加载: {model_path}")
        else:
            print(f"模型文件不存在: {model_path}")
            return None
        model.eval()
        return model
    
    def setup_figure(self):
        """设置图形界面"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.suptitle('交互式潜在空间可视化', fontsize=16)
        
        # 左侧：潜在空间2D投影（使用前两个维度）
        self.ax_latent = plt.subplot(121)
        self.ax_latent.set_xlim(-4, 4)
        self.ax_latent.set_ylim(-4, 4)
        self.ax_latent.set_xlabel('潜在维度 1')
        self.ax_latent.set_ylabel('潜在维度 2')
        self.ax_latent.set_title('潜在空间（前两个维度）')
        self.ax_latent.grid(True, alpha=0.3)
        
        # 当前点
        self.current_point, = self.ax_latent.plot([], [], 'ro', markersize=10, label='当前位置')
        
        # 点击历史点
        self.history_points, = self.ax_latent.plot([], [], 'bo', markersize=6, alpha=0.5, label='历史位置')
        self.history_z = []
        
        self.ax_latent.legend()
        
        # 右侧：生成的图像
        self.ax_image = plt.subplot(122)
        self.ax_image.set_title('解码器输出')
        self.ax_image.axis('off')
        
        # 连接鼠标点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def setup_controls(self):
        """设置控制按钮和滑块"""
        # 重置按钮
        ax_reset = plt.axes([0.45, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, '重置')
        self.btn_reset.on_clicked(self.reset)
        
        # 随机按钮
        ax_random = plt.axes([0.55, 0.02, 0.08, 0.04])
        self.btn_random = Button(ax_random, '随机')
        self.btn_random.on_clicked(self.random_sample)
        
        # 清除历史按钮
        ax_clear = plt.axes([0.65, 0.02, 0.08, 0.04])
        self.btn_clear = Button(ax_clear, '清除历史')
        self.btn_clear.on_clicked(self.clear_history)
        
        # 为其他维度添加滑块（显示前6个额外维度）
        self.sliders = []
        slider_positions = [0.15, 0.10, 0.05]  # y位置
        slider_labels = ['维度3', '维度4', '维度5']
        
        for i, (pos, label) in enumerate(zip(slider_positions, slider_labels)):
            ax_slider = plt.axes([0.05 + i*0.03, pos, 0.02, 0.03])
            slider = Slider(ax_slider, label, -3.0, 3.0, valinit=0.0, valstep=0.1, orientation='vertical')
            slider.on_changed(lambda val, idx=i+2: self.update_dimension(idx, val))
            self.sliders.append(slider)
        
    def on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes == self.ax_latent:
            # 获取点击位置
            x, y = event.xdata, event.ydata
            
            # 更新潜在向量（前两个维度）
            self.current_z[0, 0] = x
            self.current_z[0, 1] = y
            
            # 添加到历史记录
            self.history_z.append(self.current_z.clone())
            if len(self.history_z) > 50:  # 限制历史记录数量
                self.history_z.pop(0)
            
            # 更新显示
            self.update_display()
            
    def update_dimension(self, dim_idx, value):
        """更新指定维度的值"""
        if dim_idx < self.latent_dim:
            self.current_z[0, dim_idx] = value
            self.update_image()
            
    def update_display(self):
        """更新整个显示"""
        # 更新潜在空间中的点
        self.current_point.set_data([self.current_z[0, 0].cpu().item()], [self.current_z[0, 1].cpu().item()])
        
        # 更新历史点
        if self.history_z:
            history_x = [z[0, 0].cpu().item() for z in self.history_z]
            history_y = [z[0, 1].cpu().item() for z in self.history_z]
            self.history_points.set_data(history_x, history_y)
        
        # 更新滑块位置
        for i, slider in enumerate(self.sliders):
            dim_idx = i + 2
            if dim_idx < self.latent_dim:
                slider.set_val(self.current_z[0, dim_idx].cpu().item())
        
        # 更新图像
        self.update_image()
        
        # 刷新画布
        self.fig.canvas.draw_idle()
        
    def update_image(self):
        """更新生成的图像"""
        if self.model is not None:
            with torch.no_grad():
                generated = self.model.decode(self.current_z)
                generated = generated.cpu().squeeze().numpy()
                
                self.ax_image.clear()
                self.ax_image.imshow(generated, cmap='gray', vmin=0, vmax=1)
                self.ax_image.set_title(f'解码器输出\nz=[{self.current_z[0, 0].cpu().item():.2f}, {self.current_z[0, 1].cpu().item():.2f}, ...]')
                self.ax_image.axis('off')
                
    def reset(self, event):
        """重置到原点"""
        self.current_z = torch.zeros(1, self.latent_dim).to(self.device)
        self.update_display()
        
    def random_sample(self, event):
        """随机采样"""
        self.current_z = torch.randn(1, self.latent_dim).to(self.device)
        self.update_display()
        
    def clear_history(self, event):
        """清除历史记录"""
        self.history_z = []
        self.history_points.set_data([], [])
        self.fig.canvas.draw_idle()
        
    def save_current(self, filename='current_sample.png'):
        """保存当前生成的图像"""
        self.fig.savefig(f'results/{filename}', dpi=150, bbox_inches='tight')
        print(f"图像已保存: results/{filename}")
        
    def show(self):
        """显示交互式界面"""
        plt.show()

def main():
    """主函数"""
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 检查可用的模型文件
    model_path = 'checkpoints/vae_final.pth'
    if not os.path.exists(model_path):
        # 尝试找到最新的检查点
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('vae_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = f'checkpoints/{latest_checkpoint}'
            print(f"使用最新的检查点: {model_path}")
        else:
            print("没有找到可用的模型文件")
            return
    
    print("启动交互式潜在空间可视化...")
    print("使用说明:")
    print("- 点击左侧潜在空间区域来选择位置")
    print("- 使用滑块调整其他维度的值")
    print("- 点击'随机'按钮生成随机样本")
    print("- 点击'重置'按钮回到原点")
    print("- 点击'清除历史'按钮清除历史轨迹")
    
    # 创建并显示交互式可视化器
    visualizer = InteractiveLatentVisualizer(model_path, latent_dim=20)
    visualizer.show()

if __name__ == "__main__":
    main()
