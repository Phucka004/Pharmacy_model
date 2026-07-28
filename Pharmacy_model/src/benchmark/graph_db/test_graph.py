from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.graph_db.inference import GraphKnowledgePredictor, REQUIRES_MEDICAL_VISIT


predictor = GraphKnowledgePredictor()


def _print_header() -> None:
    print("=" * 90)
    print("GRAPH KNOWLEDGE DEBUG TEST")
    print(f"Project root: {ROOT_DIR}")
    print("=" * 90)


def _resolve_result(input_text: str):
    cleaned = input_text.strip()
    if cleaned.upper().startswith("BENH:"):
        return predictor.get_products_by_disease(cleaned), cleaned
    return predictor.predict_products_for_query(cleaned), predictor.detect_disease_code(cleaned)


def _run_case(title: str, query_text: str) -> None:
    print("\n" + "=" * 90)
    print(f"[{title}]")
    print("-" * 90)
    print(f"Input: {query_text}")

    result, detected_code = _resolve_result(query_text)
    print(f"Detected Disease Code: {detected_code if detected_code else 'None'}")

    if result == REQUIRES_MEDICAL_VISIT:
        print(f"Final Result: {REQUIRES_MEDICAL_VISIT}")
        print("Guardrail: bệnh nguy hiểm hoặc không có đường đi tới danh mục -> kích hoạt an toàn.")
        return

    print(f"Final Result Count: {len(result)}")
    print(f"First 3 Products: {result[:3]}")


def main() -> None:
    _print_header()

    _run_case(
    title="Test 1 (traonguocdaday)",
    query_text="Tôi bị trào ngược dạ dày"
    )

    # _run_case(
    #     title="Test 2 (An Khong Tieu)",
    #     query_text="Mỗi lần ăn cơm xong là bụng tôi lại ậm ạch, cứ liên tục ợ hơi với ợ lên vị chua rất khó chịu ở cổ họng.",
    # )

    # _run_case(
    #     title="Test 3 (Benh Tri)",
    #     query_text="Mỗi lần đi vệ sinh tôi thấy đau nhói ở vùng hậu môn, sờ vào thấy có cục thịt nhỏ sưng lộ ra ngoài.",
    # )

    # _run_case(
    #     title="Test 4 (Benh Gout Cap Tinh)",
    #     query_text="Khớp ngón chân cái của tôi đột nhiên sưng tấy đỏ và đau dữ dội sau một buổi tối đi nhậu ăn nhiều đồ hải sản.",
    # )

    # _run_case(
    #     title="Test 5 (Xuat Tinh Som)",
    #     query_text="Tôi gặp tình trạng chưa đi đến chợ đã tiêu hết tiền, thời gian quan hệ quá ngắn chỉ dưới 1 phút là đã ra rồi.",
    # )

    # _run_case(
    #     title="Test 6 (Yeu Sinh Ly O Nu)",
    #     query_text="Vùng kín của tôi dạo này rất khô rát, mỗi lần gần gũi chồng đều cảm thấy đau và không còn ham muốn.",
    # )

    # _run_case(
    #     title="Test 7 (Yeu Sinh Ly Nam)",
    #     query_text="Cậu nhỏ của tôi rất khó cương lên, hoặc có cứng được một chút thì lại xìu ngay, không duy trì được khi quan hệ.",
    # )

    # _run_case(
    #     title="Test 8 (Hoi Chung Fournier)",
    #     query_text="Vùng bìu và tầng sinh môn của tôi bị sưng đau dữ dội, da chuyển sang màu thâm đen và có mùi hôi rất khó chịu.",
    # )

    # _run_case(
    #     title="Test 9 (Dau Nua Dau)",
    #     query_text="Đầu tôi đau như búa bổ, cảm giác như có tiếng trống đập ong ong ở một bên thái dương.",
    # )

    # _run_case(
    #     title="Test 10 (Viem Phe Quan)",
    #     query_text="Tôi bị ho khục khặc kéo dài, cổ họng ngứa rát và có đờm đặc màu vàng xanh.",
    # )

    # _run_case(
    #     title="Test 11 (Mun Trung Ca)",
    #     query_text="Da mặt tôi đổ dầu nhiều và nổi rất nhiều nốt mụn bọc, mụn mủ sưng đỏ gây đau.",
    # )

    # _run_case(
    #     title="Test 12 (Nhiet Mien)",
    #     query_text="Tôi hay bị loét những nốt trắng nhỏ ở niêm mạc miệng, mỗi lần ăn đồ mặn hay cay là xót rát vô cùng.",
    # )

    # _run_case(
    #     title="Test 13 (Gau Da Dau)",
    #     query_text="Da đầu tôi bong tróc rất nhiều vảy trắng nhỏ, bám đầy trên vai áo và ngứa ngáy liên tục.",
    # )

    # _run_case(
    #     title="Test 14 (Viem Mui Di Ung)",
    #     query_text="Tôi bị hắt hơi liên tục, chảy nước mũi trong và ngứa mắt mỗi khi thời tiết thay đổi đột ngột.",
    # )

    # _run_case(
    #     title="Test 15 (Dai Trang Co That)",
    #     query_text="Bụng tôi đau quặn từng cơn dọc theo đại tràng, lúc thì táo bón lúc lại tiêu chảy phân lỏng.",
    # )

    # _run_case(
    #     title="Test 16 (Cum Chay Mua)",
    #     query_text="Tôi bị sốt cao đột ngột, người đau nhức mỏi các cơ và đau rát họng dữ dội.",
    # )

    # _run_case(
    #     title="Test 17 (Viem Amidan)",
    #     query_text="Cổ họng tôi sưng to, nuốt nước bọt cũng thấy đau và hai bên amidan có chấm mủ trắng.",
    # )

    # _run_case(
    #     title="Test 18 (Me Day Di Ung)",
    #     query_text="Sau khi ăn tôm biển, người tôi nổi đầy các nốt sẩn đỏ thành từng mảng và ngứa ngáy dữ dội.",
    # )

    # _run_case(
    #     title="Test 19 (Thieu Mau)",
    #     query_text="Tôi thường xuyên cảm thấy mệt mỏi, hụt hơi, chóng mặt khi đứng lên và da dẻ xanh xao.",
    # )

    # _run_case(
    #     title="Test 20 (Viem Ket Mac)",
    #     query_text="Mắt tôi đỏ ngầu, chảy nhiều nước mắt và có dịch vàng bết chặt hai mi mắt khi ngủ dậy.",
    # )

    # _run_case(
    #     title="Test 21 (Rung Toc)",
    #     query_text="Tôi bị rụng tóc rất nhiều, mỗi lần gội đầu hay chải đầu là tóc rụng thành từng búi.",
    # )

    # _run_case(
    #     title="Test 22 (Roi Loan Kinh Nguyet)",
    #     query_text="Kinh nguyệt của em không đều, tháng có tháng không và mỗi lần đến kỳ là đau bụng quằn quại.",
    # )

    # _run_case(
    #     title="Test 23 (U Tai)",
    #     query_text="Tai tôi cứ có tiếng ve kêu e e bên trong, đôi lúc cảm thấy hơi nghẹt và khó nghe.",
    # )

    # _run_case(
    #     title="Test 24 (Thieu Canxi)",
    #     query_text="Tôi thường xuyên bị chuột rút ở bắp chân vào ban đêm, móng tay thì giòn và rất dễ gãy.",
    # )

    # _run_case(
    #     title="Test 25 (Nut Ne Got Chan)",
    #     query_text="Gót chân của tôi bị nứt nẻ sâu, rớm máu và rất đau mỗi khi đi bộ.",
    # )

    # _run_case(
    #     title="Test 26 (Mat Ngu)",
    #     query_text="Tôi ngủ không sâu giấc, đêm nào cũng trằn trọc mất ngủ và ban ngày thì cơ thể rất mệt mỏi.",
    # )

    # _run_case(
    #     title="Test 27 (Dau Co)",
    #     query_text="Sau khi tập gym quá nặng, toàn bộ vùng cơ đùi và bắp tay của tôi bị căng cứng, đau nhức không duỗi thẳng được.",
    # )

    # _run_case(
    #     title="Test 28 (Say Tau Xe)",
    #     query_text="Tôi bị say tàu xe rất nặng, cứ bước lên ô tô là đầu óc quay cuồng, buồn nôn và nôn.",
    # )

    # _run_case(
    #     title="Test 29 (E Buot Rang)",
    #     query_text="Răng tôi buốt nhói lên tận óc mỗi khi uống nước đá lạnh hoặc ăn kem.",
    # )

    # _run_case(
    #     title="Test 30 (Cam Lanh)",
    #     query_text="Tôi bị dính nước mưa xong giờ người lờ đờ, sụt sịt mũi và hơi nhức đầu sốt nhẹ.",
    # )
    print("\nDone.")


if __name__ == "__main__":
    main()
