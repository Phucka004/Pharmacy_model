from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark.ml_baseline.inference import MLBaselinePredictor, REQUIRES_MEDICAL_VISIT

ROOT_DIR = Path(__file__).resolve().parents[3]
DISEASE_MAP_PATH = ROOT_DIR / "data" / "silver" / "disease_category_map.csv"


def _print_header() -> None:
    print("=" * 90)
    print("ML BASELINE DEBUG TEST")
    print(f"Project root: {ROOT_DIR}")
    print(f"Disease map:   {DISEASE_MAP_PATH}")
    print("=" * 90)


def _load_disease_map() -> pd.DataFrame:
    df_map = pd.read_csv(DISEASE_MAP_PATH)
    required_columns = {"disease_code", "category_key"}
    missing = required_columns - set(df_map.columns)
    if missing:
        raise ValueError(f"disease_category_map.csv thiếu cột: {', '.join(sorted(missing))}")
    df_map["disease_code"] = df_map["disease_code"].astype(str).str.strip()
    return df_map


def _debug_single_query(predictor: MLBaselinePredictor, df_map: pd.DataFrame, title: str, query_text: str) -> None:
    q_text = query_text.lower().strip()

    print(f"\n[{title}]")
    print(f"Query: {q_text}")

    pred_code = predictor.pipeline.predict([q_text])[0]
    matched_rows = df_map.loc[df_map["disease_code"] == str(pred_code).strip()]
    category_key = matched_rows.iloc[0]["category_key"] if not matched_rows.empty else ""
    category_key_text = "" if pd.isna(category_key) else str(category_key).strip()

    final_result = predictor.predict_products_for_query(q_text)

    print(f"Disease Code: {pred_code}")
    print(f"Mapped category_key: {category_key_text}")
    print(f"Final Result: {final_result}")

    if final_result == REQUIRES_MEDICAL_VISIT:
        print("Note: Model triggered the medical-visit safety branch.")


def debug_test() -> None:
    _print_header()
    print("Initializing MLBaselinePredictor...")
    predictor = MLBaselinePredictor()
    df_map = _load_disease_map()

    # queries = [
    # (
    #     "Test 1 (Alzheimer)",
    #     "dạo này tôi vừa nhớ chuyện đó xong quay đi đã quên, nhiều lúc còn không nhớ hôm nay là thứ mấy và hay lạc đường dù ở nơi quen thuộc",
    # ),
    # (
    #     "Test 2 (An Khong Tieu)",
    #     "ăn xong bụng cứ đầy hơi khó tiêu, ợ chua liên tục và nóng rát lên tận cổ họng",
    # )
# ]
    queries = [
    (
        "Test 3 (Benh Tri)",
        "mỗi lần đi đại tiện tôi đau rát hậu môn, có máu dính trên giấy và sờ thấy búi thịt nhỏ lòi ra",
    ),
    (
        "Test 4 (Benh Gout Cap Tinh)",
        "khớp ngón chân cái của tôi sưng đỏ nóng và đau dữ dội sau khi ăn nhiều hải sản với uống bia",
    ),
    (
        "Test 5 (Xuat Tinh Som)",
        "mỗi lần quan hệ tôi chỉ mới bắt đầu chưa được một phút đã xuất tinh khiến cả hai đều không hài lòng",
    ),
    (
        "Test 6 (Yeu Sinh Ly O Nu)",
        "gần đây vùng kín bị khô rát, quan hệ rất đau và tôi gần như không còn ham muốn",
    ),
    (
        "Test 7 (Yeu Sinh Ly)",
        "tôi rất khó cương cứng, có cương được cũng nhanh mềm xuống nên không thể quan hệ trọn vẹn",
    ),
    (
        "Test 8 (Hoi Chung Fournier)",
        "vùng bìu của tôi sưng đau dữ dội, da chuyển màu tím đen và có mùi hôi rất khó chịu",
    ),
    (
        "Test 9 (Dau Nua Dau)",
        "đầu tôi đau giật từng cơn ở một bên thái dương, nhìn ánh sáng là càng đau hơn",
    ),
    (
        "Test 10 (Viem Phe Quan)",
        "tôi ho có đờm vàng xanh nhiều ngày liền, cổ họng rát và tức ngực khi ho",
    ),
    (
        "Test 11 (Mun Trung Ca)",
        "mặt tôi nổi đầy mụn bọc mụn mủ đỏ, da rất nhiều dầu và sờ vào đau",
    ),
    (
        "Test 12 (Nhiet Mien)",
        "trong miệng tôi xuất hiện mấy vết loét trắng nhỏ, ăn đồ cay hay mặn là xót vô cùng",
    ),
    (
        "Test 13 (Gau Da Dau)",
        "da đầu bong rất nhiều vảy trắng, ngứa liên tục và vai áo lúc nào cũng đầy gàu",
    ),
    (
        "Test 14 (Viem Mui Di Ung)",
        "mỗi lần trời trở lạnh tôi hắt hơi liên tục, chảy nước mũi trong và ngứa mũi ngứa mắt",
    ),
    (
        "Test 15 (Dai Trang Co That)",
        "bụng tôi đau quặn từng cơn, lúc táo bón lúc tiêu chảy khiến đi ngoài thất thường",
    ),
    (
        "Test 16 (Cum Chay Mua)",
        "tôi sốt cao đột ngột, đau nhức toàn thân, đau họng và rất mệt mỏi",
    ),
    (
        "Test 17 (Viem Amidan)",
        "cổ họng sưng đau, nuốt nước bọt cũng khó và tôi thấy amidan có nhiều chấm mủ trắng",
    ),
    (
        "Test 18 (Me Day Di Ung)",
        "sau khi ăn tôm tôi nổi mề đay khắp người, ngứa dữ dội và các mảng đỏ xuất hiện liên tục",
    ),
    (
        "Test 19 (Thieu Mau)",
        "tôi thường xuyên chóng mặt, da xanh xao, đứng lên là hoa mắt và rất dễ mệt",
    ),
    (
        "Test 20 (Viem Ket Mac)",
        "mắt tôi đỏ, chảy ghèn vàng nhiều và sáng ngủ dậy hai mí mắt dính chặt vào nhau",
    ),
    (
        "Test 21 (Rung Toc)",
        "mỗi lần gội đầu tóc rụng từng nắm, tóc thưa đi thấy rõ và tôi rất lo",
    ),
    (
        "Test 22 (Roi Loan Kinh Nguyet)",
        "kinh nguyệt của tôi lúc có lúc không, chu kỳ thất thường và mỗi lần hành kinh đau bụng dữ dội",
    ),
    (
        "Test 23 (U Tai)",
        "tai tôi cứ ù ù như có tiếng ve kêu, nghe kém hơn trước và cảm giác bị nghẹt",
    ),
    (
        "Test 24 (Thieu Canxi)",
        "ban đêm tôi hay bị chuột rút ở chân, móng tay giòn dễ gãy và đôi khi tê bì",
    ),
    (
        "Test 25 (Nut Ne Got Chan)",
        "gót chân tôi nứt sâu, đi lại rất đau và có chỗ còn rớm máu",
    ),
    (
        "Test 26 (Mat Ngu)",
        "đêm nào tôi cũng trằn trọc mãi mới ngủ được, ngủ chập chờn nên sáng dậy rất mệt",
    ),
    (
        "Test 27 (Dau Co)",
        "sau buổi tập nặng hôm qua cơ đùi và bắp tay đau nhức cứng đờ, cử động rất khó",
    ),
    (
        "Test 28 (Say Tau Xe)",
        "chỉ cần lên ô tô là tôi chóng mặt buồn nôn, người vã mồ hôi và nhiều lần phải nôn",
    ),
    (
        "Test 29 (E Buot Rang)",
        "mỗi khi uống nước đá hoặc ăn kem thì răng tôi buốt nhói lên rất khó chịu",
    ),
    (
        "Test 30 (Cam Lanh)",
        "hôm qua dính mưa về giờ tôi sổ mũi, hắt hơi liên tục, hơi sốt và người mệt lả",
    ),
]
    for title, query_text in queries:
        _debug_single_query(predictor, df_map, title, query_text)

    print("\nDone.")


if __name__ == "__main__":
    debug_test()
