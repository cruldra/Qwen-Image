<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>
<p align="center">&nbsp&nbsp💜 <a href="https://chat.qwen.ai/">通义千问聊天</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(文生图)</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(图像编辑)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">魔搭社区-文生图</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">魔搭社区-图像编辑</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">技术报告</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">博客(文生图)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">博客(图像编辑)</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">文生图演示</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">图像编辑演示</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">微信群</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

## 简介
我们很高兴发布 **Qwen-Image**，这是一个200亿参数的MMDiT图像基础模型，在**复杂文本渲染**和**精确图像编辑**方面取得了显著进展。实验表明，该模型在图像生成和编辑方面都具有强大的通用能力，在文本渲染方面表现卓越，特别是对中文的支持。

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/bench.png#center)

## 最新动态
- 2025.08.19: 我们观察到Qwen-Image-Edit的性能不一致问题。为确保最佳效果，请更新到最新的diffusers提交版本。预期在身份保持和指令遵循方面会有改进。
- 2025.08.18: 我们很兴奋地宣布开源Qwen-Image-Edit！🎉 您可以通过下面的快速开始指南在本地环境中试用，或者前往[通义千问聊天](https://chat.qwen.ai/)或[Huggingface演示](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit)立即体验在线演示！如果您喜欢我们的工作，请为我们的仓库点个星，您的鼓励对我们意义重大！
- 2025.08.09: Qwen-Image现在支持多种LoRA模型，如MajicBeauty LoRA，能够生成高度逼真的美女图像。查看[魔搭社区](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary)上的可用权重。
<p align="center">
    <img src="assets/magicbeauty.png"/>
<p>
    
- 2025.08.05: Qwen-Image现在原生支持ComfyUI，请参阅[ComfyUI中的Qwen-Image：图像文本生成的新时代！](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)
- 2025.08.05: Qwen-Image现已在通义千问聊天中上线。点击[通义千问聊天](https://chat.qwen.ai/)并选择"图像生成"。
- 2025.08.05: 我们在Arxiv上发布了[技术报告](https://arxiv.org/abs/2508.02324)！
- 2025.08.04: 我们发布了Qwen-Image权重！请查看[Huggingface](https://huggingface.co/Qwen/Qwen-Image)和[魔搭社区](https://modelscope.cn/models/Qwen/Qwen-Image)！
- 2025.08.04: 我们发布了Qwen-Image！查看我们的[博客](https://qwenlm.github.io/blog/qwen-image)了解更多详情！

> [!NOTE]
> 由于访问量较大，如果您想在线体验我们的演示，我们也推荐访问DashScope、WaveSpeed和LibLib。请在下面的社区支持部分找到相关链接。

## 快速开始

1. 确保您的transformers>=4.51.3（支持Qwen2.5-VL）

2. 安装最新版本的diffusers
```
pip install git+https://github.com/huggingface/diffusers
```

### 文本生成图像

以下代码片段展示了如何使用模型根据文本提示生成图像：

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# 加载管道
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # 英文提示词
    "zh": ", 超清，4K，电影级构图." # 中文提示词
}

# 生成图像
prompt = '''一个咖啡店入口有一个黑板标牌写着"Qwen Coffee 😊 $2 per cup"，旁边有一个霓虹灯显示"通义千问"。旁边挂着一张美丽中国女性的海报，海报下方写着"π≈3.1415926-53589793-23846264-33832795-02384197"。'''

negative_prompt = " " # 如果不使用负面提示词，建议使用空格。

# 使用不同的宽高比生成
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["zh"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
```

### 图像编辑

```python
import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("管道已加载")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("./input.png").convert("RGB")
prompt = "将兔子的颜色改为紫色，背景加上闪光灯效果。"

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("图像已保存至", os.path.abspath("output_image_edit.png"))
```

> [!NOTE]
> 我们强烈建议使用提示词重写来提高编辑案例的稳定性。作为参考，请查看我们的官方[演示脚本](src/examples/edit_demo.py)，其中包含示例系统提示词。Qwen-Image-Edit正在积极发展中，持续开发中。敬请期待未来的增强功能！

## 展示案例

### 通用案例
其突出能力之一是在各种图像中进行高保真文本渲染。无论是英语等字母语言还是中文等汉字文字，Qwen-Image都能以惊人的准确性保持字体细节、布局连贯性和上下文和谐。文本不仅仅是叠加的，而是无缝集成到视觉结构中。

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center)

除了文本，Qwen-Image在通用图像生成方面表现出色，支持广泛的艺术风格。从逼真的场景到印象派绘画，从动漫美学到极简设计，该模型能够流畅地适应创意提示，使其成为艺术家、设计师和故事讲述者的多功能工具。

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center)

在图像编辑方面，Qwen-Image远超简单的调整。它支持高级操作，如风格转换、对象插入或移除、细节增强、图像内文本编辑，甚至人体姿态操作——所有这些都具有直观的输入和连贯的输出。这种控制水平使专业级编辑触手可及。

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center)

但Qwen-Image不仅仅是创建或编辑，它还能理解。它支持一套图像理解任务，包括对象检测、语义分割、深度和边缘（Canny）估计、新视角合成和超分辨率。这些能力虽然在技术上不同，但都可以看作是由深度视觉理解驱动的智能图像编辑的专门形式。

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center)

这些功能共同使Qwen-Image不仅仅是一个生成漂亮图片的工具，而是一个用于智能视觉创作和操作的综合基础模型——语言、布局和图像在此汇聚。

### 图像编辑教程

Qwen-Image-Edit的亮点之一在于其强大的语义和外观编辑能力。语义编辑是指在保持原始视觉语义的同时修改图像内容。为了直观地展示这种能力，让我们以通义千问的吉祥物——水豚为例：
![水豚](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片3.JPG#center)
如图所示，尽管编辑后的图像中大部分像素与输入图像（最左边的图像）不同，但水豚的角色一致性得到了完美保持。Qwen-Image-Edit强大的语义编辑能力使原创IP内容的轻松多样化创作成为可能。
此外，在通义千问聊天中，我们设计了一系列围绕16种MBTI人格类型的编辑提示词。利用这些提示词，我们成功地基于我们的吉祥物水豚创建了一套MBTI主题表情包，轻松扩展了IP的影响力和表达方式。
![MBTI表情包系列](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片4.JPG#center)
此外，新视角合成是语义编辑中的另一个关键应用场景。如下面两个示例图像所示，Qwen-Image-Edit不仅可以将对象旋转90度，还可以执行完整的180度旋转，让我们直接看到对象的背面：
![视角变换90度](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片12.JPG#center)
![视角变换180度](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片13.JPG#center)
语义编辑的另一个典型应用是风格转换。例如，给定一个输入肖像，Qwen-Image-Edit可以轻松将其转换为各种艺术风格，如宫崎骏风格。这种能力在虚拟头像创建等应用中具有重要价值：
![风格转换](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片1.JPG#center)
除了语义编辑，外观编辑是另一个常见的图像编辑需求。外观编辑强调保持图像的某些区域完全不变，同时添加、移除或修改特定元素。下图展示了在场景中添加标牌的案例。
如图所示，Qwen-Image-Edit不仅成功插入了标牌，还生成了相应的反射，展现了对细节的卓越关注。
![添加标牌](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片6.JPG#center)
下面是另一个有趣的例子，展示了如何从图像中移除细发丝和其他小物体。
![移除细发丝](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片7.JPG#center)
此外，可以将图像中特定字母"n"的颜色修改为蓝色，实现对特定元素的精确编辑。
![修改文本颜色](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片8.JPG#center)
外观编辑在调整人物背景或更换服装等场景中也有广泛应用。下面三张图像分别展示了这些实际用例。
![修改背景](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片11.JPG#center)
![修改服装](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片5.JPG#center)
Qwen-Image-Edit的另一个突出特点是其准确的文本编辑能力，这源于Qwen-Image在文本渲染方面的深厚专业知识。如下所示，以下两个案例生动地展示了Qwen-Image-Edit在编辑英文文本方面的强大性能：
![编辑英文文本1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片15.JPG#center)
![编辑英文文本2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片16.JPG#center)
Qwen-Image-Edit还可以直接编辑中文海报，不仅能够修改大标题文本，还能精确调整甚至小而复杂的文本元素。
![编辑中文海报](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片17.JPG#center)
最后，让我们通过一个具体的图像编辑示例来演示如何使用链式编辑方法逐步纠正Qwen-Image生成的书法作品中的错误：
![书法作品](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片18.JPG#center)
在这幅作品中，几个中文字符包含生成错误。我们可以利用Qwen-Image-Edit逐步纠正它们。例如，我们可以在原始图像上绘制边界框来标记需要纠正的区域，指示Qwen-Image-Edit修复这些特定区域。在这里，我们希望字符"稽"在红框内正确书写，字符"亭"在蓝色区域内准确渲染。
![纠正字符](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片19.JPG#center)
然而，在实践中，字符"稽"相对较为生僻，模型无法一步正确纠正。"稽"的右下部分应该是"旨"而不是"日"。此时，我们可以进一步用红框突出显示"日"部分，指示Qwen-Image-Edit微调这个细节并将其替换为"旨"。
![微调字符](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片20.JPG#center)
是不是很神奇？通过这种链式、逐步编辑的方法，我们可以持续纠正字符错误，直到达到期望的最终结果。
![最终版本1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片21.JPG#center)
![最终版本2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片22.JPG#center)
![最终版本3](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片23.JPG#center)
![最终版本4](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片24.JPG#center)
![最终版本5](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片25.JPG#center)
最终，我们成功获得了完全正确的《兰亭集序》书法版本！
总之，我们希望Qwen-Image-Edit能够进一步推动图像生成领域的发展，真正降低视觉内容创作的技术门槛，并激发更多创新应用。

### 高级用法

#### 提示词增强
为了增强提示词优化和多语言支持，我们推荐使用由Qwen-Plus驱动的官方提示词增强工具。

您可以直接将其集成到您的代码中：
```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

或者，从命令行运行示例脚本：

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

## 部署Qwen-Image

Qwen-Image支持多GPU API服务器进行本地部署：

### 多GPU API服务器管道和用法

多GPU API服务器将启动一个基于Gradio的Web界面，具有：
- 多GPU并行处理
- 高并发队列管理
- 自动提示词优化
- 支持多种宽高比

通过环境变量配置：
```bash
export NUM_GPUS_TO_USE=4          # 使用的GPU数量
export TASK_QUEUE_SIZE=100        # 任务队列大小
export TASK_TIMEOUT=300           # 任务超时时间（秒）
```

```bash
# 启动gradio演示服务器，提示词增强的API密钥
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

## AI竞技场

为了全面评估Qwen-Image的通用图像生成能力并客观地与最先进的闭源API进行比较，我们推出了[AI竞技场](https://aiarena.alibaba-inc.com)，这是一个基于Elo评级系统构建的开放基准测试平台。AI竞技场为模型评估提供了公平、透明和动态的环境。

在每一轮中，两张图像——由从同一提示词随机选择的模型生成——匿名呈现给用户进行成对比较。用户为更好的图像投票，结果用于通过Elo算法更新个人和全球排行榜，使开发者、研究人员和公众能够以稳健和数据驱动的方式评估模型性能。AI竞技场现已公开可用，欢迎大家参与模型评估。

![AI竞技场](assets/figure_aiarena_website.png)

最新的排行榜排名可在[AI竞技场排行榜](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image)查看。

如果您希望在AI竞技场上部署您的模型并参与评估，请联系weiyue.wy@alibaba-inc.com。

## 社区支持

### Huggingface

Diffusers从第一天就支持Qwen-Image。对LoRA和微调工作流程的支持目前正在开发中，很快就会推出。

### 魔搭社区
* **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)** 为Qwen-Image提供全面支持，包括低GPU内存逐层卸载（在4GB显存内推理）、FP8量化、LoRA/全量训练。
* **[DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)** 为Qwen-Image推理和部署提供高级优化，包括基于FBCache的加速、无分类器引导（CFG）并行等。
* **[魔搭社区AIGC中心](https://www.modelscope.cn/aigc)** 提供Qwen Image的实践体验，包括：
    - [图像生成](https://www.modelscope.cn/aigc/imageGeneration)：使用Qwen Image模型生成高保真图像。
    - [LoRA训练](https://www.modelscope.cn/aigc/modelTraining)：轻松训练Qwen Image LoRA以实现个性化概念。

### WaveSpeedAI

WaveSpeed从第一天就在其平台上部署了Qwen-Image，访问他们的[模型页面](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image)了解更多详情。

### LiblibAI

LiblibAI从第一天就原生支持Qwen-Image。访问他们的[社区](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c)页面了解更多详情和讨论。

### 推理加速方法：cache-dit

cache-dit为Qwen-Image提供DBCache、TaylorSeer和Cache CFG的缓存加速支持。访问他们的[示例](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py)了解更多详情。

## 许可协议

Qwen-Image采用Apache 2.0许可证。

## 引用

如果您发现我们的工作有用，我们诚挚地鼓励您引用我们的工作。

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report},
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324},
}
```

## 联系我们和加入我们

如果您想与我们的研究团队取得联系，我们很乐意听到您的声音！加入我们的[Discord](https://discord.gg/z3GAxXZ9Ce)或扫描二维码通过我们的[微信群](assets/wechat.png)联系——我们始终欢迎讨论和合作。

如果您对这个仓库有疑问、想要分享反馈或直接贡献，我们欢迎您在GitHub上提出问题和拉取请求。您的贡献有助于让Qwen-Image对每个人都更好。

如果您对基础研究充满热情，我们正在招聘全职员工（FTE）和研究实习生。不要等待——请联系我们：fulai.hr@alibaba-inc.com

## Star历史

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)
