import os
import sys
import time
import json
import streamlit as st
from dotenv import load_dotenv

# 引入 LangChain 核心依赖组件
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. 环境密钥池加载与深度清洗净化
# ==========================================
load_dotenv()

API_KEY_POOL = [
    os.environ.get("GOOGLE_API_KEY1"),
    os.environ.get("GOOGLE_API_KEY2"),
    os.environ.get("GOOGLE_API_KEY3"),
    os.environ.get("GOOGLE_API_KEY4"),
    os.environ.get("GOOGLE_API_KEY5"), 
]

active_google_keys = []
for k in API_KEY_POOL:
    if k:
        clean_k = str(k).strip().replace('"', '').replace("'", "")
        if clean_k and clean_k.upper() != "NONE" and clean_k != "":
            active_google_keys.append(clean_k)

# 全局初始化密钥轮询索引（存储在 Streamlit 会话状态中，防止页面重刷重置）
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

def get_current_key_and_rotate():
    """获取当前就绪的 API Key，并将指针严格推向下一个（实现接力赛跑机制）"""
    if not active_google_keys:
        return None, -1
    key = active_google_keys[st.session_state.key_index]
    current_used_idx = st.session_state.key_index
    # 强制指针强制递增轮转
    st.session_state.key_index = (st.session_state.key_index + 1) % len(active_google_keys)
    return key, current_used_idx

# ==========================================
# 2. Streamlit 页面前端配置与渲染
# ==========================================
st.set_page_config(page_title="Gemini 工业级高可用字幕转换器", layout="centered")
st.title("🛡️ Gemini 工业级高可用字幕转换器")
st.caption("最新全量完备版：JSON 规约双向绑定 + 强制 Key 轮询接力 + 10秒冷却 + 超级防御解包")

st.sidebar.header("🔑 密钥池审计状态")
if not active_google_keys:
    st.sidebar.error("❌ 熔断：未在环境配置文件中发现任何可用的 GOOGLE_API_KEY！")
    print("❌ [CONSOLE ERROR] 铁血审计熔断：未发现任何可用的 GOOGLE_API_KEY！")
    st.stop()
else:
    st.sidebar.success(f"🟢 发现 {len(active_google_keys)} 个可用 Key 正在交替待命")
    with st.sidebar.expander("查看当前就绪的 Key 状态"):
        for idx, k in enumerate(active_google_keys):
            st.write(f"Key {idx+1}: ...{k[-6:] if len(k)>6 else '***'}")

target_lang = st.sidebar.text_input("请输入目标翻译语言:", value="英文")
batch_size = st.sidebar.slider("单次合并翻译的字幕条数（建议50行）:", min_value=10, max_value=100, value=50)

# ==========================================
# 3. Python 结构化解析 SRT
# ==========================================
def parse_srt(srt_text):
    """用纯 Python 将 SRT 解析为高度结构化的内部字典列表，将时间轴锁定在本地"""
    blocks = srt_text.replace('\r\n', '\n').strip().split('\n\n')
    parsed_items = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0].strip()
            time_axis = lines[1].strip()
            text = "\n".join(lines[2:]).strip()
            parsed_items.append({"index": index, "time": time_axis, "text": text})
    return parsed_items

# ==========================================
# 4. JSON 级原子化高可用翻译核心
# ==========================================
def translate_pure_texts_with_json_format(text_list, target_language, batch_round):
    """
    通过将数据转化为标准的 JSON 数组，逼迫大模型一对一返回，拒绝 \n 干扰。
    内置全量输入输出调试日志打印，并具备超级防 list 报错机制。
    """
    if not text_list:
        return []
    
    expected_count = len(text_list)
    print("\n" + "="*60)
    print(f"🛰️  [ROUND {batch_round}] 开始处理新批次 (包含 {expected_count} 行字幕)")
    print("="*60)
    
    # 将 50 行纯文字打包成标准的 JSON 对象，绝不发任何时间轴给大模型
    input_data_dict = {"source_texts": text_list}
    input_prompt = json.dumps(input_data_dict, ensure_ascii=False, indent=2)
    
    system_prompt = (
        "你是一个精通影视字幕的专业翻译官。用户会提供一个包含多行文本的 JSON 对象。\n"
        f"请把 `source_texts` 数组中的每一行纯文本逐行翻译成【{target_language}】。\n"
        "【严格输出规范】:\n"
        "1. 必须返回一个符合标准 JSON 格式的对象，键名为 \"translated_texts\"，其值是一个字符串数组。\n"
        f"2. \"translated_texts\" 数组的元素数量必须严格等于输入数组的数量，当前必须为 【{expected_count}】 个元素。\n"
        "3. 数组中的每一个翻译结果必须与输入数组的索引一一对应。\n"
        "4. 不要返回任何行号、时间轴、合并文本或解释性文字。\n\n"
        "【输出 JSON 示例】:\n"
        "{\n"
        "  \"translated_texts\": [\n"
        "    \"Translation for line 1\",\n"
        "    \"Translation for line 2\"\n"
        "  ]\n"
        "}"
    )

    # 🛠️ 控制台高亮 debug 打印：全量打印大模型输入 Prompt 结构
    print(f"📤 [LLM INPUT PROMPT - SYSTEM_PROMPT]:\n{system_prompt}\n")
    print(f"📤 [LLM INPUT PROMPT - USER_JSON_DATA] (共 {expected_count} 条):\n{input_prompt}")
    print("-" * 60)

    max_attempts = len(active_google_keys)
    
    for attempt in range(max_attempts):
        gemini_key, key_id = get_current_key_and_rotate()
        print(f"👉 [尝试 {attempt + 1}/{max_attempts}] 正在启动备用接力，调用 GOOGLE_API_KEY{key_id + 1} ...")
        
        try:
            # 严格配置调用指定的 1.5 flash lite 模型，并硬编码锁定输出为 JSON 对象
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite", 
                temperature=0.1, 
                google_api_key=gemini_key,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input_prompt)
            ]
            
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed = time.time() - start_time
            
            # ==========================================
            # ✨ 【核心超级防御性洗涤】绝对打碎并清洗掉返回的 list 数据结构
            # ==========================================
            raw_content = response.content
            
            # 情况 A：如果新版 LangChain 框架返回的是元素列表 [TextMessage(...)] 
            if isinstance(raw_content, list):
                clean_text_list = []
                for block in raw_content:
                    if hasattr(block, "text"):
                        clean_text_list.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        clean_text_list.append(block["text"])
                    else:
                        clean_text_list.append(str(block))
                final_text = "".join(clean_text_list)
            # 情况 B：如果是常规的纯文本字符串
            else:
                final_text = str(raw_content)
                
            # 去除可能混入的外层 Markdown ```json 标志包裹
            final_text = final_text.strip()
            if final_text.startswith("```json"):
                final_text = final_text.split("```json", 1)[1]
            if final_text.endswith("```"):
                final_text = final_text.rsplit("```", 1)[0]
            final_text = final_text.strip()
            # ===================================================================
            
            # 此时解析 JSON 字符串绝对安全稳定，彻底消灭 'list' object has no attribute 'strip' 报错
            response_json = json.loads(final_text)
            translated_lines = response_json.get("translated_texts", [])
            actual_count = len(translated_lines)
            
            print(f"✅ [SUCCESS] KEY{key_id + 1} 完美响应成功! 本轮请求耗时: {elapsed:.2f}秒")
            print(f"📊 [LINE CHECK] 预期解析数组长度: {expected_count} | 模型实际返回数组长度: {actual_count}")
            
            # 极高级别防漏行、防合并对齐机制
            if actual_count != expected_count:
                print(f"⚠️ [WARN] 返回数组长度不绝对对齐！正在启动本地 Python 自动裁切补白对齐机制...")
                if actual_count < expected_count:
                    translated_lines += [""] * (expected_count - actual_count)
                else:
                    translated_lines = translated_lines[:expected_count]
            
            # 控制台打印前两条数据对照，供肉眼直观 Debug
            # ========== ✨ 替换为：全量打印 50 行对照，方便您在终端肉眼核对 ==========
            print(f"📝 [全量译文对照输出 - 共 {expected_count} 行]:")
            for idx, (orig, trans) in enumerate(zip(text_list, translated_lines)):
                print(f"  [{idx + 1}] 原文: \"{orig}\" --> 译文: \"{trans}\"")
            print("-" * 50)
            # ===================================================================
            return translated_lines
            
        except Exception as e:
            print(f"❌ [API ERROR] KEY{key_id + 1} 发生异常。原因分析: {str(e)[:150]}")
            if attempt < max_attempts - 1:
                print("🔄 正在丢弃坏 Key 并触发指针轮询，动用下一个备用 Key 重试当前批次...")
                time.sleep(1.0)
                continue
            else:
                print("🚨 [FATAL] 密钥池内所有可用密钥在当前批次全部遭遇限流熔断！将使用空字符串保护填补。")
                return [""] * expected_count

# ==========================================
# 5. 整体工作流装配
# ==========================================
uploaded_file = st.file_uploader("请上传您的长单语 .srt 文件", type=["srt"])

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    srt_content = bytes_data.decode("utf-8")
    
    # 纯本地 Python 结构化提取，死锁时间轴
    parsed_srt_list = parse_srt(srt_content)
    total_lines = len(parsed_srt_list)
    st.info(f"💾 解析成功！共提取出 {total_lines} 条结构化字幕段落。")

    if st.button("🚀 开始 50行原子化接力翻译", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        translated_srt_result = []
        
        batch_round = 1
        
        # 按照步长（例如 50 行）进行块滑窗迭代处理
        for i in range(0, total_lines, batch_size):
            current_batch = parsed_srt_list[i : i + batch_size]
            status_text.text(f"⏳ 正在执行第 {batch_round} 轮接力，处理第 {i+1} 到 {min(i + batch_size, total_lines)} 行...")
            
            # 仅仅提取纯文字
            pure_texts = [item['text'] for item in current_batch]
            
            # 执行原子化翻译（全量控制台日志托管）
            translated_texts = translate_pure_texts_with_json_format(pure_texts, target_lang, batch_round)
            
            # 本地利用 Python 将原序号、原时间轴、原有语种文字与全新译文缝合在一起
            for item, trans_txt in zip(current_batch, translated_texts):
                bilingual_block = (
                    f"{item['index']}\n"
                    f"{item['time']}\n"
                    f"{item['text']}\n"       # 第一行：原有语种
                    f"{trans_txt}"            # 第二行：新语种译文文字
                )
                translated_srt_result.append(bilingual_block)
            
            # 更新 Streamlit 前端进度条
            progress_bar.progress(min((i + batch_size) / total_lines, 1.0))
            
            # ⏳ 严格的 10 秒物理冷却停顿逻辑（防止免费 Tier API 短时间过度充能触发 429 报错）
            if i + batch_size < total_lines:
                print(f"⏱️ [COOL DOWN] 第 {batch_round} 轮大功告成。为让密钥池回血，强制进入 10 秒物理冷却...")
                for countdown in range(10, 0, -1):
                    status_text.text(f"⏱️ 保护机制生效中：每轮结束强制冷却中... 还剩 {countdown} 秒")
                    time.sleep(1.0)
            
            batch_round += 1
            
        status_text.text("✨ 全量接力翻译圆满完成！")
        print("\n🎉 ====== [ALL FINISHED] 恭喜！全量长字幕双语拼装完毕 ====== 🎉\n")
        
        # 拼回标准的双回车换行 SRT 文本格式
        final_srt_output = "\n\n".join(translated_srt_result)
        
        # 提供成果导出
        output_filename = uploaded_file.name.replace(".srt", f".bilingual.srt")
        st.download_button(
            label="💾 下载最新机制生成的双语 SRT 文件",
            data=final_srt_output.encode("utf-8"),
            file_name=output_filename,
            mime="text/srt"
        )