import os

from openai import OpenAI


def api(prompt, model, kwargs={}):
    """调用OpenAI API"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.zhizengzeng.com/v1"
    )
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    response_format = kwargs.get('response_format', None)
    
    completion_kwargs = {
        'model': model,
        'messages': messages,
    }
    
    if response_format:
        completion_kwargs['response_format'] = response_format

    response = client.chat.completions.create(**completion_kwargs)
    
    return response.choices[0].message.content


def generate_creative_prompt(subject, simple_prompt):
    """
    基于主体和简单提示词生成创意提示词
    
    Args:
        subject: 图片中的主体，如"小熊"、"猫"等
        simple_prompt: 简单的提示词，如"画画"、"弹吉他"等
        
    Returns:
        str: 生成的详细英文提示词
    """
    
    # 构建系统提示词
    SYSTEM_PROMPT = f'''
你是一个专业的原创IP角色设计和图像编辑提示词生成器。你的任务是根据用户提供的主体和简单动作，生成详细、具体、富有创意和想象力的英文提示词，让AI能够自由发挥创造出独特的原创IP角色形象。

## 核心要求：
1. 生成的提示词必须是英文
2. 提示词要包含丰富的细节描述：服装、道具、姿势、表情、环境、色彩、材质等
3. 鼓励创意和想象，不要局限于现实，可以加入奇幻、科幻、卡通等元素
4. 描述要生动具体，让AI有足够的创作空间
5. 主体必须是用户指定的"{subject}"
6. 提示词长度要充实，包含多个维度的描述

## 详细描述维度：
- **服装造型**：详细的服装描述，包括颜色、材质、风格、配饰
- **道具物品**：相关的工具、装备、物品，描述其外观和使用方式
- **动作姿势**：具体的身体姿态、手势、表情
- **环境背景**：适合的场景设置，可以是抽象或具体的
- **视觉效果**：光影、色彩、特效、氛围
- **风格特色**：艺术风格、主题色调、整体感觉

## 参考示例（注意这些示例相对简单，你需要生成更丰富的描述）：

- 输入：主体="小熊"，动作="画画"
  期望输出风格：Add a vibrant artist's easel with a large canvas showing colorful abstract paintings, place a wooden palette with rainbow paint blobs in the bear's left paw, a fine-tipped paintbrush with paint dripping in the right paw, dress the bear in a paint-splattered blue artist's smock with rolled-up sleeves, add a red beret tilted stylishly on its head, position the bear standing confidently with one paw on hip, surrounded by scattered paint tubes, brushes in a ceramic jar, and colorful paint splatters on the ground, with warm golden lighting creating an inspiring creative atmosphere

- 输入：主体="小熊"，动作="弹吉他"
  期望输出风格：Transform the bear into a folk musician with a vintage acoustic guitar featuring intricate wood grain patterns and mother-of-pearl inlays, position the bear sitting cross-legged on a colorful woven rug, wearing a cozy knitted sweater in earth tones with rolled sleeves, add a leather guitar strap across the shoulder, place the bear's paws in realistic guitar-playing position with fingers on frets and strumming motion, include a harmonica holder around the neck, scatter sheet music and a leather-bound songbook nearby, add warm campfire lighting with soft shadows, creating a cozy evening atmosphere with musical notes floating in the air

- 输入：主体="小熊"，动作="宇航员"
  期望输出风格：Dress the bear in a futuristic white and silver spacesuit with glowing blue LED strips along the seams, add a transparent helmet with heads-up display reflections, include oxygen tubes and control panels on the chest, position the bear floating in zero gravity with arms outstretched pointing toward distant stars, add rocket boosters on the back with subtle flame effects, surround with floating space debris and colorful nebula clouds, include Earth visible in the background, create dramatic lighting with starlight reflections on the helmet visor

请根据用户输入的主体和动作，生成比上述示例更加丰富详细的英文提示词。要充分发挥想象力，添加更多创意元素和视觉细节。只返回提示词内容，不要包含其他解释。
'''

    # 构建用户输入
    user_input = f"主体：{subject}\n动作：{simple_prompt}"
    
    # 调用API生成提示词
    try:
        generated_prompt = api(SYSTEM_PROMPT + "\n\n" + user_input, model='gemini-2.5-pro')
        generated_prompt = generated_prompt.strip()
        
        # 清理可能的多余内容
        if generated_prompt.startswith('"') and generated_prompt.endswith('"'):
            generated_prompt = generated_prompt[1:-1]
            
        return generated_prompt
        
    except Exception as e:
        print(f"生成提示词时出错: {e}")
        # 返回一个基础的提示词作为备选
        return f"Make the {subject} {simple_prompt}"


def get_subject_suggestions():
    """
    获取主体建议列表
    
    Returns:
        list: 常见主体的建议列表
    """
    return [
        "小熊", "猫", "狗", "兔子", "熊猫", 
        "小鸟", "企鹅", "狐狸", "老虎", "狮子",
        "人物", "小孩", "女孩", "男孩", "机器人"
    ]


def get_action_suggestions():
    """
    获取动作建议列表
    
    Returns:
        list: 常见动作的建议列表
    """
    return [
        "画画", "弹吉他", "唱歌", "跳舞", "读书",
        "写字", "做饭", "运动", "睡觉", "思考",
        "宇航员", "魔法师", "医生", "老师", "警察",
        "超级英雄", "公主", "国王", "忍者", "海盗"
    ]


def validate_inputs(subject, simple_prompt):
    """
    验证输入参数
    
    Args:
        subject: 主体
        simple_prompt: 简单提示词
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not subject or not subject.strip():
        return False, "请输入图片主体"
    
    if not simple_prompt or not simple_prompt.strip():
        return False, "请输入动作或角色"
    
    if len(subject.strip()) > 20:
        return False, "主体名称过长，请控制在20个字符以内"
    
    if len(simple_prompt.strip()) > 50:
        return False, "动作描述过长，请控制在50个字符以内"
    
    return True, ""


def create_prompt_with_examples(subject, simple_prompt):
    """
    使用示例模式生成提示词（备用方法）

    Args:
        subject: 主体
        simple_prompt: 简单提示词

    Returns:
        str: 生成的提示词
    """
    # 预定义的丰富模板映射
    templates = {
        "画画": f"Transform the {subject} into an artistic creator wearing a paint-splattered apron in vibrant colors, add a large wooden easel with a canvas showing abstract colorful brushstrokes, place a wooden palette with rainbow paint blobs in one hand and a fine-tipped brush with dripping paint in the other, surround with scattered paint tubes, ceramic brush jars, and colorful paint splatters on the ground, add a stylish beret and rolled-up sleeves, create warm golden lighting with an inspiring creative atmosphere",

        "弹吉他": f"Style the {subject} as a folk musician with a vintage acoustic guitar featuring intricate wood grain and mother-of-pearl inlays, dress in a cozy knitted sweater with earth tones and rolled sleeves, position sitting cross-legged on a colorful woven rug with realistic guitar-playing hand positions, add a leather guitar strap, harmonica holder around neck, scattered sheet music and leather songbook nearby, create warm campfire lighting with musical notes floating in the air",

        "宇航员": f"Dress the {subject} in a futuristic white and silver spacesuit with glowing blue LED strips along seams, add a transparent helmet with heads-up display reflections, include oxygen tubes and control panels on chest, position floating in zero gravity with arms pointing toward distant stars, add rocket boosters with subtle flame effects, surround with space debris and colorful nebula clouds, include Earth in background with dramatic starlight reflections",

        "魔法师": f"Transform the {subject} into a mystical wizard wearing an elaborate dark robe with golden star patterns, add a tall pointed hat with crescent moon emblem, place an ornate magic wand with glowing crystal tip in hand, surround with floating magical orbs and sparkling particles, add ancient spellbooks and potion bottles, create mysterious purple and blue lighting with magical energy swirling around",

        "读书": f"Style the {subject} as a scholarly reader wearing cozy reading glasses and a comfortable cardigan, position sitting in a plush armchair with an oversized open book, surround with towering bookshelves filled with colorful volumes, add a warm reading lamp casting golden light, include scattered papers, bookmarks, and a steaming cup of tea on a side table, create a cozy library atmosphere",

        "做饭": f"Transform the {subject} into a master chef wearing a pristine white chef's hat and apron with embroidered details, add professional cooking utensils including a gleaming knife and wooden spoon, position standing at a modern kitchen counter with fresh ingredients scattered around, include steaming pots, colorful vegetables, herbs, and cooking flames, create warm kitchen lighting with aromatic steam effects",

        "睡觉": f"Style the {subject} in comfortable pajamas with cute patterns, position lying peacefully on a cloud-like bed with fluffy pillows and a soft blanket, add a crescent moon and twinkling stars in the background, include a bedside lamp with warm glow, stuffed animals, and dream bubbles floating above, create a serene nighttime atmosphere with gentle blue and purple lighting",

        "跳舞": f"Transform the {subject} into an energetic dancer wearing flowing, colorful dance attire with ribbons and accessories, position in a dynamic pose with arms gracefully extended and one leg lifted, add motion blur effects and musical notes floating around, include a dance floor with spotlights, colorful stage lighting creating dramatic shadows and highlights, surround with swirling fabric and movement trails"
    }

    # 如果有直接匹配的模板，使用丰富模板
    if simple_prompt in templates:
        return templates[simple_prompt]

    # 否则生成基础但仍然详细的提示词
    return f"Transform the {subject} into a {simple_prompt} character with appropriate detailed costume, props, and atmospheric setting, including rich colors, textures, and dramatic lighting to create an engaging and creative scene"
