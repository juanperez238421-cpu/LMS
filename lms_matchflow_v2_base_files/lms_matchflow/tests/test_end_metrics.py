from lms_matchflow.end_metrics import parse_mmss_to_seconds, parse_first_int, extract_end_of_combat_metrics


def test_parse_mmss_to_seconds_ok():
    assert parse_mmss_to_seconds("Tiempo con vida 01:34") == 94
    assert parse_mmss_to_seconds("01:00") == 60
    assert parse_mmss_to_seconds("9:59") == 599


def test_parse_mmss_to_seconds_bad():
    assert parse_mmss_to_seconds("") is None
    assert parse_mmss_to_seconds("abc") is None
    assert parse_mmss_to_seconds("01:99") is None


def test_parse_first_int_ok():
    assert parse_first_int("Eliminaciones 1") == 1
    assert parse_first_int("Objetos construidos: 12") == 12


def test_extract_end_of_combat_metrics():
    m = extract_end_of_combat_metrics({
        "end_time_alive": "Tiempo con vida 01:34",
        "end_eliminations": "Eliminaciones 1",
        "end_objects_built": "5 elementos construidos",
    })
    assert m.time_alive_sec == 94
    assert m.eliminations == 1
    assert m.objects_built == 5
