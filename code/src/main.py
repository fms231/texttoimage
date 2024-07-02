import torch
from PIL import Image
import clip
import os.path as osp
import sys
import torchvision.utils as vutils
sys.path.insert(0, '../')

from models.GALIP import NetG, CLIP_TXT_ENCODER

# 加载模型权重
def load_model_weights(model, weights, train=True):
    state_dict = weights
    model.load_state_dict(state_dict)
    return model

# 准备模型
device = 'cpu' # 'cpu' # 'cuda:0'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()

# 加载模型
text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
netG = NetG(64, 100, 512, 256, 3, clip_model).to(device)
path = '../saved_models/bird/state_epoch_150.pth'
# path = '../saved_models/pretrained/pre_cc12m.pth'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG = load_model_weights(netG, checkpoint['model']['netG'])

# 每次生成6张图片
batch_size = 6
noise = torch.randn((batch_size, 100)).to(device)

# 定义界面
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6 import QtGui

class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.resize(600, 400)
        
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 取消自带的标题栏
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("Window{background-color: rgb(255, 255, 255); border-radius: 10px;}")

        self.centralWidget().layout().setContentsMargins(0, 0, 0, 0)

        # 自定义标题栏
        hlayout = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("文本到图像生成器")
        self.title.setStyleSheet("font-size: 20px; font-weight: bold; color: red;")
        self.minimizeButton = QtWidgets.QPushButton("最小化")
        self.minimizeButton.setStyleSheet("background-color: rgb(0, 170, 0);")
        self.minimizeButton.setMaximumSize(100, 30)
        self.closeButton = QtWidgets.QPushButton("关闭")
        self.closeButton.setStyleSheet("background-color: rgb(170, 0, 0);")
        self.closeButton.setMaximumSize(100, 30)

        # 设置槽函数
        self.minimizeButton.clicked.connect(self.showMinimized)
        self.closeButton.clicked.connect(self.close)

        # 应用到布局
        hlayout.addWidget(self.title)
        hlayout.addWidget(self.minimizeButton)
        hlayout.addWidget(self.closeButton)
        
        # 界面的上半部分
        self.topLayout = QtWidgets.QHBoxLayout()
        # 文本
        self.textLabel = QtWidgets.QPlainTextEdit(self)
        self.textLabel.setMinimumSize(448, 200)
        self.textLabel.setStyleSheet("border: 6px solid green;")
        # 图像
        self.gird = QtWidgets.QGridLayout()
        for i in range(2):
            for j in range(3):
                imagelabel = QtWidgets.QLabel(self)
                imagelabel.setMinimumSize(224, 224)
                imagelabel.setStyleSheet("border: 4px solid red;")
                self.gird.addWidget(imagelabel, i, j)
        # self.imageLabel = QtWidgets.QLabel(self)
        # self.imageLabel.setMinimumSize(300, 200)
        # self.imageLabel.setStyleSheet("border: 1px solid black;")

        # 应用到布局
        self.topLayout.addWidget(self.textLabel)
        self.topLayout.addLayout(self.gird)

        # 界面的下半部分
        self.bottomLayout = QtWidgets.QHBoxLayout()
        # 生成按钮
        self.generateButton = QtWidgets.QPushButton(self)
        self.generateButton.setText("✔生成图片")

        # 应用到布局
        self.bottomLayout.addWidget(self.generateButton)

        # 整体布局
        mainLayout.addLayout(hlayout)
        mainLayout.addLayout(self.topLayout)
        mainLayout.addLayout(self.bottomLayout)

        # 设置槽函数
        self.generateButton.clicked.connect(self.generateImage)

    # 拖动顶部标题栏
    def mousePressEvent(self, event):
        self.windowHandle().startSystemMove()  

    # 生成图片
    def generateImage(self):
        text = self.textLabel.toPlainText()
        text = text.strip()
        if text == "":
            return
        tokenized_text = clip.tokenize([text]).to(device)
        sent_emb,word_emb = text_encoder(tokenized_text)
        # 重复batch_size次
        sent_emb = sent_emb.repeat(batch_size,1)
        # 获取batch_size个生成图片
        fake_imgs = netG(noise,sent_emb,eval=True).float()
        # 保存6张图片
        for i in range(fake_imgs.size(0)):
            vutils.save_image(fake_imgs[i], './samples/%s_%d.png'%(text,i))
            pic = QtGui.QPixmap('./samples/%s_%d.png'%(text,i)).scaled(224, 224)
            self.gird.itemAt(i).widget().setPixmap(pic)
        QtWidgets.QMessageBox.information(self, "提示", "图片生成成功！已保存在samples文件夹下！")
        
from qt_material import apply_stylesheet

# 构建应用
app = QtWidgets.QApplication()
Window = MWindow()

# 设置界面
apply_stylesheet(app, theme='dark_teal.xml')
# 启动
Window.show()
sys.exit(app.exec())
