import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gradio as gr
from l0_engine.atomizer import atomize
from l0_engine.noise_engine.pipeline import filter_noise


def demo(text: str) -> dict:
    """Atomize + noise filter, return structured results."""
    if not text.strip():
        return {"error": "请输入文本"}

    claims = atomize(text)
    clean = filter_noise(claims, text)

    results = []
    for c in clean:
        results.append({
            "claim": c.get("claim_text", ""),
            "noise_score": c.get("noise_score", 0),
            "hollowness": round(c.get("structural_hollowness", 0), 4),
            "causal_risk": round(c.get("causal_risk", 0), 4),
        })

    return {
        "input_length": len(text),
        "total_claims": len(claims),
        "clean_claims": len(clean),
        "noise_filtered": len(claims) - len(clean),
        "noise_rate": round((len(claims) - len(clean)) / max(len(claims), 1), 4),
        "results": results,
    }


if __name__ == "__main__":
    gr.Interface(
        fn=demo,
        inputs=gr.Textbox(
            label="输入中文新闻文本",
            placeholder="粘贴一段中文新闻，例如：恒大集团负债2.4万亿元，会议指出要加强金融监管……",
            lines=6,
        ),
        outputs=gr.JSON(label="去噪结果"),
        title="Verdict — 中文信息去噪引擎",
        description="零 LLM、纯 CPU、<300MB 内存。输入一段中文新闻，自动拆分原子事实并过滤噪声。",
        examples=[
            ["恒大地产集团有限公司负债2.4万亿元，注册资本为150亿元人民币。会议指出，要坚持以经济建设为中心。"],
            ["5月28日，工信部发布公告称，我国5G基站数量突破500万个，覆盖率达98%。华为公司表示将继续加大研发投入。"],
            ["据了解，该项目建设周期为3年，预计总投资超过50亿元。张三表示，公司将继续推进数字化转型战略。此次会议特别强调要加强金融监管。"],
        ],
        theme="soft",
    ).launch()