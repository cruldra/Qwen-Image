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
你是一个专业的图像编辑提示词生成器。你的任务是根据用户提供的主体和简单动作，生成详细、具体、富有创意的英文提示词。

## 规则：
1. 生成的提示词必须是英文
2. 提示词要具体描述动作、物品、姿势、位置等细节
3. 保持自然和合理的场景设置
4. 提示词长度适中，不要过于冗长
5. 主体必须是用户指定的"{subject}"

## 参考示例：
- 输入：主体="小熊"，动作="画画" 
  输出：Add a colorful art board and paintbrush in the bear's hands, position the bear standing in front of the art board as if painting

- 输入：主体="小熊"，动作="弹吉他"
  输出：Add a brown acoustic guitar in the bear's arms, positioned horizontally across its lap, with fingers gently plucking the strings; keep the bear's posture seated on the ground, facing forward, wearing the same white T-shirt with 'Qwen' logo

- 输入：主体="小熊"，动作="宇航员"
  输出：This bear is wearing a spacesuit and pointing towards the distance

- 输入：主体="小熊"，动作="魔法师"
  输出：This bear is wearing a tuxedo, a magician's hat, and holding a magic wand, performing a magic trick gesture

请根据用户输入的主体和动作，生成类似风格的英文提示词。只返回提示词内容，不要包含其他解释。
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
    # 预定义的模板映射
    templates = {
        "画画": f"Add a colorful art board and paintbrush in the {subject}'s hands, position the {subject} standing in front of the art board as if painting",
        "弹吉他": f"Add a brown acoustic guitar in the {subject}'s arms, positioned horizontally across its lap, with fingers gently plucking the strings; keep the {subject}'s posture seated on the ground, facing forward",
        "宇航员": f"This {subject} is wearing a spacesuit and pointing towards the distance",
        "魔法师": f"This {subject} is wearing a tuxedo, a magician's hat, and holding a magic wand, performing a magic trick gesture",
        "读书": f"Add an open book in the {subject}'s hands, position the {subject} sitting comfortably while reading with focused expression",
        "做饭": f"Add a chef's hat on the {subject}'s head and cooking utensils in hands, position the {subject} standing in front of a stove",
        "睡觉": f"Position the {subject} lying down with closed eyes, add a pillow under the head and a blanket covering the body",
        "跳舞": f"Position the {subject} in a dynamic dancing pose with arms raised and one leg lifted, showing movement and rhythm"
    }
    
    # 如果有直接匹配的模板，使用模板
    if simple_prompt in templates:
        return templates[simple_prompt]
    
    # 否则生成基础提示词
    return f"Make the {subject} {simple_prompt} with appropriate props and posture"
