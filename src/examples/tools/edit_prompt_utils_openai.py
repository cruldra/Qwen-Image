import os
from openai import OpenAI

def api(prompt, model, kwargs={}):
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


def get_caption_language(prompt):
    ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
        # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
        # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
    ]
    for char in prompt:
        if any(start <= char <= end for start, end in ranges):
            return 'zh'
    return 'en'

def polish_edit_prompt_en(original_prompt):
    SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.  

## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.  
- **For text replacement tasks, always use the fixed template:**
    - `Replace "xx" to "yy"`.  
    - `Replace the xx bounding box to "yy"`.  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image's context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text \"LIMITED EDITION\" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  

### 3. Attribute Modification Tasks
- For color changes, specify the exact color name or description.  
- For size changes, use relative terms (larger, smaller, double size, half size).  
- For position changes, use clear directional terms (left, right, center, top, bottom).  

### 4. Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.  
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  

Below is the edit instruction to be rewritten. Please directly rewrite it into a clear, concise, and actionable edit instruction:
    '''
    original_prompt = original_prompt.strip()
    prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\nRewritten Edit Instruction:"
    success=False
    while not success:
        try:
            polished_prompt = api(prompt, model='gemini-2.5-pro')
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"Error during API call: {e}")
    return polished_prompt

def polish_edit_prompt_zh(original_prompt):
    SYSTEM_PROMPT = '''
# 编辑指令重写器
你是一位专业的编辑指令重写师。你的任务是根据用户提供的指令，生成精确、简洁、视觉上可实现的专业级编辑指令。

请严格遵循以下重写规则：

## 1. 通用原则
- 保持重写后的提示词**简洁**。避免过长的句子，减少不必要的描述性语言。
- 如果指令矛盾、模糊或无法实现，优先进行合理推断和修正，必要时补充细节。
- 保持原始指令的核心意图不变，只增强其清晰度、合理性和视觉可行性。
- 所有添加的对象或修改必须与编辑输入图像的整体场景逻辑和风格保持一致。

## 2. 任务类型处理规则
### 1. 添加、删除、替换任务
- 如果指令清晰（已包含任务类型、目标实体、位置、数量、属性），保留原意并仅完善语法。
- 如果描述模糊，补充最少但足够的细节（类别、颜色、大小、方向、位置等）。例如：
    > 原始："添加一个动物"
    > 重写："在右下角添加一只浅灰色小猫，坐姿面向镜头"
- 移除无意义指令：例如"添加0个对象"应被忽略或标记为无效。
- 对于替换任务，指定"将Y替换为X"并简要描述X的关键视觉特征。

### 2. 文本编辑任务
- 所有文本内容必须用英文双引号`" "`括起来。不要翻译或改变文本的原始语言，不要改变大小写。
- **对于文本替换任务，始终使用固定模板：**
    - `将"xx"替换为"yy"`
    - `将xx边界框替换为"yy"`
- 如果用户未指定文本内容，根据指令和输入图像的上下文推断并添加简洁文本。例如：
    > 原始："添加一行文字"（海报）
    > 重写："在顶部中央添加文字\"LIMITED EDITION\"，带轻微阴影"
- 简洁地指定文本位置、颜色和布局。

### 3. 属性修改任务
- 对于颜色变化，指定确切的颜色名称或描述。
- 对于大小变化，使用相对术语（更大、更小、双倍大小、一半大小）。
- 对于位置变化，使用清晰的方向术语（左、右、中心、顶部、底部）。

### 4. 风格转换或增强任务
- 如果指定了风格，用关键视觉特征简洁描述。例如：
    > 原始："迪斯科风格"
    > 重写："1970年代迪斯科：闪烁灯光、迪斯科球、镜面墙、彩色调"
- 如果指令说"使用参考风格"或"保持当前风格"，分析输入图像，提取主要特征（颜色、构图、纹理、光照、艺术风格），并将其整合到提示词中。
- **对于着色任务，包括修复老照片，始终使用固定模板：**"修复老照片，去除划痕，降低噪声，增强细节，高分辨率，真实感，自然肤色，清晰面部特征，无失真，复古照片修复"

下面是要重写的编辑指令。请直接将其重写为清晰、简洁、可操作的编辑指令：
    '''
    original_prompt = original_prompt.strip()
    prompt = f'''{SYSTEM_PROMPT}\n\n用户输入：{original_prompt}\n重写的编辑指令：'''
    success=False
    while not success:
        try:
            polished_prompt = api(prompt, model='gemini-2.5-pro')
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"Error during API call: {e}")
    return polished_prompt


def rewrite_edit_instruction(input_prompt):
    """
    重写图像编辑指令
    
    Args:
        input_prompt: 原始编辑指令
        
    Returns:
        str: 重写后的编辑指令
    """
    lang = get_caption_language(input_prompt)
    if lang == 'zh':
        return polish_edit_prompt_zh(input_prompt)
    elif lang == 'en':
        return polish_edit_prompt_en(input_prompt)
