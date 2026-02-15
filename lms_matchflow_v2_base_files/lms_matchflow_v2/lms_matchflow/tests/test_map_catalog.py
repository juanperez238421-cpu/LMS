from lms_matchflow.map_catalog import match_map_name_from_ocr, DEFAULT_MAPS


def test_match_map_name_from_ocr_direct():
    m = match_map_name_from_ocr("Mapa: CASTILLOS DESERTICOS", DEFAULT_MAPS)
    assert m is not None
    assert m.map_id == "castillos_deserticos"


def test_match_map_name_from_ocr_overlap():
    m = match_map_name_from_ocr("Estas en jardines congelados", DEFAULT_MAPS)
    assert m is not None
    assert m.map_id == "jardines_congelados"
